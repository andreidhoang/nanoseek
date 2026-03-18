"""
Monitored training wrapper — runs pre_train.py as a subprocess,
captures output to a log file, and writes a machine-readable status
file that an AI agent (or human) can check at any time.

The status file (training_status.json) contains:
- PID of the training process
- Current state (running/completed/failed/killed)
- Last N log lines
- All health alerts detected
- The exact command that was run

Usage:
    # Start a monitored training run
    python -m nanoseek.scripts.monitored_train \
        --run gate1-smoke --scale anchor \
        --num-iterations 100 --eval-every 50 --save-every 100

    # All args after the script are forwarded to pre_train.py
    # Output goes to: logs/{run_name}.log
    # Status goes to: logs/{run_name}_status.json

    # Check status from another terminal (or from Claude Code agent):
    cat logs/gate1-smoke_status.json | python -m json.tool

    # Or use the built-in checker:
    python -m nanoseek.scripts.monitored_train --check gate1-smoke
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


LOG_DIR = "logs"
STATUS_SUFFIX = "_status.json"


def get_log_path(run_name: str) -> str:
    return os.path.join(LOG_DIR, f"{run_name}.log")


def get_status_path(run_name: str) -> str:
    return os.path.join(LOG_DIR, f"{run_name}{STATUS_SUFFIX}")


def write_status(run_name: str, status: dict):
    """Atomically write status file."""
    path = get_status_path(run_name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2, default=str)
    os.replace(tmp, path)


def parse_alerts_from_line(line: str) -> list:
    """Extract health monitor alerts from a log line."""
    alerts = []
    # Match [CRITICAL] or [WARNING] from TrainingHealthMonitor
    for severity in ["CRITICAL", "WARNING"]:
        if f"[{severity}]" in line:
            alerts.append({"severity": severity, "message": line.strip()})
    return alerts


def parse_step_from_line(line: str) -> dict:
    """Extract step number and metrics from a training log line."""
    # Match: step 00050/4194 (1.2%) | loss: 8.1234 (main:7.9000 mtp:0.2234 aux:0.000100) | H_load: 5.82 | lr×: 0.025 | γ: 0.0010 | grad: 0.45 | dt: 1.23s | mfu: 42.1% | tok/s: 125,000
    m = re.search(r"step (\d+)/(\d+).*loss: ([\d.]+) \(main:([\d.]+) mtp:([\d.]+) aux:([\d.]+)\).*H_load: ([\d.]+).*lr×: ([\d.]+).*γ: ([\d.]+).*grad: ([\d.]+).*dt: ([\d.]+)s.*mfu: ([\d.]+)", line)
    if m:
        return {
            "step": int(m.group(1)),
            "total_steps": int(m.group(2)),
            "loss": float(m.group(3)),
            "main_loss": float(m.group(4)),
            "mtp_loss": float(m.group(5)),
            "aux_loss": float(m.group(6)),
            "h_load": float(m.group(7)),
            "lr_mult": float(m.group(8)),
            "gamma": float(m.group(9)),
            "grad_norm": float(m.group(10)),
            "step_time": float(m.group(11)),
            "mfu": float(m.group(12)),
        }
    # Fallback: try simpler pattern for backward compat
    m2 = re.search(r"step (\d+)/(\d+).*loss: ([\d.]+).*H_load: ([\d.]+).*grad: ([\d.]+).*mfu: ([\d.]+)", line)
    if m2:
        return {
            "step": int(m2.group(1)),
            "total_steps": int(m2.group(2)),
            "loss": float(m2.group(3)),
            "h_load": float(m2.group(4)),
            "grad_norm": float(m2.group(5)),
            "mfu": float(m2.group(6)),
        }
    return {}


def run_training(run_name: str, extra_args: list):
    """Run pre_train.py as a subprocess with full monitoring."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = get_log_path(run_name)

    cmd = [
        sys.executable, "-m", "nanoseek.scripts.pre_train",
        *extra_args,
    ]

    # Initial status
    status = {
        "run_name": run_name,
        "pid": None,
        "state": "starting",
        "command": " ".join(cmd),
        "log_file": log_path,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "return_code": None,
        "last_step": None,
        "last_loss": None,
        "last_h_load": None,
        "last_grad_norm": None,
        "last_mfu": None,
        "alerts": [],
        "critical_count": 0,
        "warning_count": 0,
        "last_lines": [],
        "error_output": None,
    }
    write_status(run_name, status)

    print(f"Starting monitored training: {run_name}")
    print(f"  Log: {log_path}")
    print(f"  Status: {get_status_path(run_name)}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Check status: python -m nanoseek.scripts.monitored_train --check {run_name}")
    print()

    # Open log file and start subprocess
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

        status["pid"] = proc.pid
        status["state"] = "running"
        write_status(run_name, status)

        # Forward SIGTERM/SIGINT to child process
        def forward_signal(signum, frame):
            if proc.poll() is None:
                proc.send_signal(signum)
        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        # Read output line by line, parse, log, update status
        last_lines_buffer = []
        try:
            for line in proc.stdout:
                # Write to log file
                log_file.write(line)
                log_file.flush()

                # Print to terminal
                sys.stdout.write(line)
                sys.stdout.flush()

                # Keep last 50 lines for status
                last_lines_buffer.append(line.rstrip())
                if len(last_lines_buffer) > 50:
                    last_lines_buffer.pop(0)

                # Parse alerts
                alerts = parse_alerts_from_line(line)
                for alert in alerts:
                    status["alerts"].append({
                        **alert,
                        "time": time.strftime("%H:%M:%S"),
                    })
                    if alert["severity"] == "CRITICAL":
                        status["critical_count"] += 1
                    else:
                        status["warning_count"] += 1

                # Parse step metrics
                step_info = parse_step_from_line(line)
                if step_info:
                    status["last_step"] = step_info.get("step")
                    status["last_loss"] = step_info.get("loss")
                    status["last_h_load"] = step_info.get("h_load")
                    status["last_grad_norm"] = step_info.get("grad_norm")
                    status["last_mfu"] = step_info.get("mfu")

                # Update status file every 10 lines (not every line — disk I/O)
                status["last_lines"] = last_lines_buffer[-20:]
                if len(last_lines_buffer) % 10 == 0:
                    write_status(run_name, status)

        except Exception as e:
            status["error_output"] = str(e)

        # Wait for process to finish
        proc.wait()

        status["return_code"] = proc.returncode
        status["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["last_lines"] = last_lines_buffer[-20:]

        if proc.returncode == 0:
            status["state"] = "completed"
        else:
            status["state"] = "failed"

        write_status(run_name, status)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Run: {run_name} — {status['state'].upper()}")
    print(f"  Return code: {proc.returncode}")
    print(f"  Duration: {status['start_time']} → {status['end_time']}")
    print(f"  Last step: {status['last_step']}")
    print(f"  Alerts: {status['critical_count']} critical, {status['warning_count']} warnings")
    if status["alerts"]:
        print(f"  Last alert: {status['alerts'][-1]['message']}")
    print(f"{'=' * 60}")

    return proc.returncode


def check_status(run_name: str):
    """Print the current status of a training run."""
    status_path = get_status_path(run_name)
    if not os.path.exists(status_path):
        print(f"No status file found for run: {run_name}")
        print(f"  Expected at: {status_path}")

        # List available runs
        if os.path.exists(LOG_DIR):
            available = [f.replace(STATUS_SUFFIX, "")
                        for f in os.listdir(LOG_DIR) if f.endswith(STATUS_SUFFIX)]
            if available:
                print(f"  Available runs: {', '.join(sorted(available))}")
        return

    with open(status_path, "r") as f:
        status = json.load(f)

    # Check if process is still alive
    pid = status.get("pid")
    actually_running = False
    if pid and status["state"] == "running":
        try:
            os.kill(pid, 0)  # signal 0 = check if alive
            actually_running = True
        except ProcessLookupError:
            status["state"] = "crashed (PID gone)"
        except PermissionError:
            actually_running = True  # alive but different user

    print(f"{'=' * 60}")
    print(f"  Run: {status['run_name']}")
    print(f"  State: {status['state']}" + (" (PID {pid} alive)".format(pid=pid) if actually_running else ""))
    print(f"  PID: {pid}")
    print(f"  Started: {status.get('start_time', '?')}")
    if status.get("end_time"):
        print(f"  Ended: {status['end_time']}")
    print(f"  Return code: {status.get('return_code', 'still running')}")
    print(f"  Last step: {status.get('last_step', '?')}")
    print(f"  Last loss: {status.get('last_loss', '?')}")
    print(f"  Last H_load: {status.get('last_h_load', '?')}")
    print(f"  Last grad_norm: {status.get('last_grad_norm', '?')}")
    print(f"  Last MFU: {status.get('last_mfu', '?')}%")
    print(f"  Alerts: {status.get('critical_count', 0)} critical, {status.get('warning_count', 0)} warnings")
    print(f"  Log: {status.get('log_file', '?')}")
    print(f"  Command: {status.get('command', '?')}")

    # Show alerts
    alerts = status.get("alerts", [])
    if alerts:
        print(f"\n  Recent alerts (last 10):")
        for a in alerts[-10:]:
            print(f"    [{a.get('time', '?')}] {a['message']}")

    # Show last log lines
    last_lines = status.get("last_lines", [])
    if last_lines:
        print(f"\n  Last log output:")
        for line in last_lines[-10:]:
            print(f"    {line}")

    print(f"{'=' * 60}")

    # Return diagnosis hints for the agent
    if status["state"] == "failed":
        print(f"\n  DIAGNOSIS HINTS:")
        print(f"  1. Check full log: cat {status.get('log_file', '?')}")
        print(f"  2. Check last 50 lines: tail -50 {status.get('log_file', '?')}")
        if status.get("critical_count", 0) > 0:
            print(f"  3. {status['critical_count']} CRITICAL alerts — likely cause of failure")
        if status.get("return_code") == -9:
            print(f"  3. Return code -9 = OOM killed. Reduce --device-batch-size")
        elif status.get("return_code") == -15:
            print(f"  3. Return code -15 = SIGTERM (preemption?). Resume from checkpoint.")
        print(f"  4. Re-run with fix: python -m nanoseek.scripts.monitored_train [same args]")


def list_runs():
    """List all monitored training runs."""
    if not os.path.exists(LOG_DIR):
        print("No logs directory found.")
        return

    status_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(STATUS_SUFFIX)])
    if not status_files:
        print("No monitored runs found.")
        return

    print(f"{'Run':<30} {'State':<12} {'Step':<8} {'Loss':<10} {'Alerts':<10}")
    print("-" * 70)

    for sf in status_files:
        with open(os.path.join(LOG_DIR, sf), "r") as f:
            s = json.load(f)
        name = s.get("run_name", "?")
        state = s.get("state", "?")
        step = s.get("last_step", "?")
        loss = s.get("last_loss", "?")
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        alerts = f"{s.get('critical_count', 0)}C/{s.get('warning_count', 0)}W"
        print(f"{name:<30} {state:<12} {step:<8} {loss:<10} {alerts:<10}")


def kill_run(run_name: str):
    """Kill a running training process."""
    status_path = get_status_path(run_name)
    if not os.path.exists(status_path):
        print(f"No status file for: {run_name}")
        return

    with open(status_path, "r") as f:
        status = json.load(f)

    pid = status.get("pid")
    if not pid:
        print("No PID recorded.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid} ({run_name})")
        print("Process will save emergency checkpoint before exiting.")
        # Update status
        status["state"] = "killed"
        status["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_status(run_name, status)
    except ProcessLookupError:
        print(f"PID {pid} not found (already exited).")
    except PermissionError:
        print(f"Permission denied to kill PID {pid}.")


if __name__ == "__main__":
    # Check for meta-commands first
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--check":
            run_name = sys.argv[2] if len(sys.argv) > 2 else None
            if run_name:
                check_status(run_name)
            else:
                list_runs()
            sys.exit(0)

        elif sys.argv[1] == "--list":
            list_runs()
            sys.exit(0)

        elif sys.argv[1] == "--kill":
            if len(sys.argv) < 3:
                print("Usage: --kill <run_name>")
                sys.exit(1)
            kill_run(sys.argv[2])
            sys.exit(0)

    # Otherwise, forward all args to pre_train.py
    # Extract --run name for our tracking
    run_name = "unknown"
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--run" and i + 1 < len(args):
            run_name = args[i + 1]
            break

    exit_code = run_training(run_name, args)
    sys.exit(exit_code)
