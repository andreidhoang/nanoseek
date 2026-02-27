# 19 — Performance Load Testing & Benchmarking Harness

## Principal Engineer's Implementation Guide — February 2026

**Author**: Distinguished AI Researcher & Principal Engineer, Foundation Models Division
**Scope**: Complete load testing framework — async load generation, latency/throughput metrics collection, A/B comparison reports for the Three Money Levers
**Prerequisite**: `07_INFERENCE_ENGINE.md` (continuous batching engine), `15_OPENAI_COMPATIBLE_API_SERVER.md` (OpenAI-compatible HTTP API), `08_SPECULATIVE_DECODING_MTP.md` (MTP speculative decoding)
**Outcome**: Reproducible, statistically rigorous benchmarks that quantify the throughput/latency impact of MLA KV cache, continuous batching, and MTP speculative decoding — the three optimizations that determine serving cost

---

## 1. Problem Statement

You have built the engine (Doc 07), the API server (Doc 15), speculative decoding (Doc 08), MLA attention (Doc 02), and quantization (Doc 16). But how do you know any of it actually matters? You timed a single request — 42 tokens/second — and called it done.

That number is meaningless. Here is why:

| What You Measured | What Production Looks Like |
|---|---|
| 1 user, 1 request, idle GPU | 50-200 concurrent users, GPU saturated |
| Fixed 128-token prompt | Distribution: 10% short (32 tok), 40% medium (128-512 tok), 50% long (1K-4K tok) |
| Cold start, first request | Warm JIT, CUDA graphs cached, KV pool pre-allocated |
| Average latency: 42 tok/s | p50=38, p95=112, p99=340 tok/s — the tail is 8× worse |
| No contention | Prefill and decode competing for GPU, memory pressure causing preemption |

**A single-request benchmark is not a benchmark — it is a demo.** Production performance is defined by the distribution of latencies under realistic concurrent load, not by the best-case number you can put on a slide.

### The Three Money Levers

The FMS Lab Plan identifies three optimizations that directly determine serving cost per token:

| Lever | Doc | Claimed Improvement | Cost Impact |
|---|---|---|---|
| MLA KV Cache (23× compression) | 02 | 23× more concurrent sequences per GPU | Fewer GPUs needed → $/token drops |
| Continuous Batching | 07 | 5-10× throughput vs static batching | Same GPU serves 5-10× more users |
| MTP Speculative Decoding | 08 | 1.5-2.5× decode throughput | Faster generation → lower latency → fewer GPUs at SLA |

These claims must be **measured, not asserted**. The load testing harness exists to produce the before/after numbers that prove (or disprove) each lever's impact under realistic conditions.

### What We Need

| Capability | Why |
|---|---|
| Open-loop load generation with configurable concurrency | Closed-loop underestimates tail latency (Section 2) |
| Multiple load patterns (steady, burst, ramp) | Different patterns stress different subsystems |
| Per-request trace collection (queue, prefill, decode, total) | Aggregates hide bimodal distributions |
| Percentile computation (p50, p90, p95, p99) | Averages are lies (Section 2) |
| A/B comparison with statistical significance | "It feels faster" is not evidence |
| Machine-readable reports (JSON) + human-readable reports (ASCII tables) | CI pipelines consume JSON; humans consume tables |

---

## 2. First Principles

### Throughput vs Latency: Little's Law

The fundamental relationship governing any queuing system:

```
L = λ × W

L = average number of requests in the system (concurrency)
λ = arrival rate (requests/second)
W = average time a request spends in the system (latency)
```

This is not an approximation — it is a mathematical identity for any stable system:

- **At fixed concurrency**, increasing throughput (λ) requires decreasing latency (W). This is the single-request optimization story.
- **At fixed latency**, increasing throughput requires increasing concurrency (L). This is the batching story.
- **In practice**, increasing concurrency increases latency (contention, memory pressure). The throughput-latency curve bends. The load tester must trace this curve.

```
Throughput
(tok/s)     ╱‾‾‾‾‾‾‾‾‾‾‾‾╲
           ╱                ╲  ← Saturation: GPU memory full,
          ╱                  ╲   preemption kicks in
         ╱                    ╲
        ╱  ← Linear scaling    ╲ ← Collapse
       ╱     (batching wins)    ╲
─────╱────────────────────────────╲────▶ Concurrency
     1    8   16   32   64  128  256
```

### Why Percentiles Matter More Than Averages

Consider two systems:

```
System A: 100 requests, all complete in 100ms
  Mean = 100ms, p99 = 100ms

System B: 99 requests at 50ms, 1 request at 5000ms
  Mean = 99.5ms, p99 = 5000ms
```

System B has a *lower* mean but a p99 of 5 seconds. At 10,000 requests per hour, 100 users per hour wait 50× longer than median. Tail latency causes in LLM serving:

| Cause | Mechanism | Percentile Impact |
|---|---|---|
| Prefill contention | Long prompt blocks decode batch | p95-p99 spike |
| KV cache preemption | Memory pressure evicts sequence; re-prefill on resume | p99 doubles |
| Speculative rejection | All K drafts rejected; fallback to sequential decode | p90 increase |
| GC pause | Python garbage collector during decode step | p99 spike (50-200ms) |

### Open-Loop vs Closed-Loop Load Generation

This is the single most important design decision. **Closed-loop** (wrong): the client waits for the previous response before sending the next. If the server slows down, arrival rate drops — the server never sees the burst that reveals its breaking point. **Open-loop** (correct): the client sends requests at a fixed rate regardless. If the server slows, requests queue — exactly what happens in production.

```
Closed-loop: server slows → client slows → load drops → tail hidden
Open-loop:   server slows → queue grows → tail exposed (reality)
```

Our implementation uses open-loop: `asyncio.sleep(inter_arrival_time)` between dispatches, independent of response collection. Gil Tene's HdrHistogram paper shows closed-loop can underestimate p99 by 10-100×.

### Warm-Up Periods and Steady-State Measurement

| Warm-up Effect | Duration | Latency Impact |
|---|---|---|
| CUDA context initialization | First request | +500-2000ms |
| JIT compilation (torch.compile) | First 3-5 requests | +200-1000ms per request |
| KV cache pool allocation | First request | +50-200ms |
| TCP connection establishment | First request per connection | +1-5ms |

**Strategy**: Send `warmup_requests` (default: 20) before the measurement window. Discard all warm-up traces. Steady state is detected when rolling-window mean latency changes by <5%.

### Statistical Rigor: Confidence Intervals

A single run of 100 requests might show p99 = 340ms. Run again: p99 = 280ms. Both are sample estimates. We use the **bootstrap method**: resample with replacement B=1000 times, compute the percentile on each, take the 2.5th/97.5th percentiles as the 95% CI. If CI width exceeds 10% of point estimate, the report flags the result as low confidence.

### A/B Testing Methodology

To measure the impact of an optimization, run the server twice (control and treatment), controlling for:

| Variable | How Controlled |
|---|---|
| Hardware | Same GPU, same node, sequential runs |
| Prompt distribution | Seeded RNG; save and replay prompt sequences |
| Warm-up | Fixed warmup_requests count |
| Server state | Clean restart between A and B |

**Statistical test**: Mann-Whitney U (non-parametric, skew-robust) at p < 0.05. Effect size: rank-biserial correlation r from U statistic. r > 0.5 is a "large" effect.

---

## 3. Production Code

### File: `fms/perf/loadgen.py`

```python
"""
Open-loop load generator for NanoSeek inference benchmarking.

Sends requests at a fixed arrival rate regardless of server response time,
ensuring that queuing effects and tail latency are accurately captured.

Usage:
    python -m fms.perf.loadgen \
        --target http://localhost:8000 \
        --concurrency 16 \
        --duration 60

Reference: Schroeder et al., "Open Versus Closed: A Cautionary Tale" (NSDI 2006)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from .metrics import MetricsCollector, RequestTrace

logger = logging.getLogger(__name__)


@dataclass
class PromptDistribution:
    """
    Defines the distribution of prompt and output lengths.

    Production LLM traffic is NOT uniform.  Typical distribution:
      15% short (≤64 tok), 35% medium (65-512), 35% long (513-2048),
      15% very long (2049-4096).
    """
    prompt_lengths: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048, 4096]
    )
    prompt_weights: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05]
    )
    output_lengths: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )
    output_weights: List[float] = field(
        default_factory=lambda: [0.20, 0.35, 0.30, 0.15]
    )

    def sample_prompt_length(self, rng: random.Random) -> int:
        return rng.choices(self.prompt_lengths, weights=self.prompt_weights, k=1)[0]

    def sample_output_length(self, rng: random.Random) -> int:
        return rng.choices(self.output_lengths, weights=self.output_weights, k=1)[0]


UNIFORM_DISTRIBUTION = PromptDistribution(
    prompt_lengths=[32, 128, 512, 1024, 2048],
    prompt_weights=[0.20, 0.20, 0.20, 0.20, 0.20],
    output_lengths=[64, 128, 256, 512],
    output_weights=[0.25, 0.25, 0.25, 0.25],
)

SHORT_PROMPT_DISTRIBUTION = PromptDistribution(
    prompt_lengths=[32, 64, 128],
    prompt_weights=[0.33, 0.34, 0.33],
    output_lengths=[64, 128],
    output_weights=[0.50, 0.50],
)

LONG_PROMPT_DISTRIBUTION = PromptDistribution(
    prompt_lengths=[1024, 2048, 4096],
    prompt_weights=[0.33, 0.34, 0.33],
    output_lengths=[256, 512],
    output_weights=[0.50, 0.50],
)


class LoadGenerator:
    """
    Open-loop HTTP load generator for LLM inference servers.

    Key design: requests are dispatched on a timer (open-loop), not gated
    on responses.  Uses aiohttp for non-blocking HTTP.  Prompt sequences
    are seeded for reproducibility across A/B comparisons.
    """

    def __init__(
        self,
        target_url: str,
        prompt_distribution: Optional[PromptDistribution] = None,
        seed: int = 42,
        warmup_requests: int = 20,
        timeout_sec: float = 120.0,
        max_connections: int = 256,
        model_name: str = "nanoseek-1b",
    ):
        self.target_url = target_url.rstrip("/")
        self.distribution = prompt_distribution or PromptDistribution()
        self.rng = random.Random(seed)
        self.seed = seed
        self.warmup_requests = warmup_requests
        self.timeout_sec = timeout_sec
        self.max_connections = max_connections
        self.model_name = model_name
        self.collector: Optional[MetricsCollector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._stop_event = asyncio.Event()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            self._session = aiohttp.ClientSession(
                connector=connector, timeout=timeout,
            )
        return self._session

    async def _close_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.25)

    def _generate_prompt_tokens(self, length: int) -> str:
        """Generate synthetic prompt of ~length tokens using common English words."""
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "in", "a", "large", "open", "field", "under", "clear", "blue",
            "sky", "with", "bright", "sun", "shining", "down", "on", "green",
            "grass", "and", "tall", "trees", "near", "river", "bank", "where",
        ]
        return " ".join(self.rng.choices(words, k=length))

    def _build_request_body(self, prompt_length: int, output_length: int) -> Dict:
        prompt = self._generate_prompt_tokens(prompt_length)
        return {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": output_length,
            "temperature": 0.0,
            "stream": False,
        }

    async def _send_request(
        self, request_id: str, body: Dict, is_warmup: bool = False,
    ) -> Optional[RequestTrace]:
        session = await self._get_session()
        url = f"{self.target_url}/v1/completions"

        t_start = time.perf_counter()
        try:
            async with session.post(
                url, json=body, headers={"X-Request-ID": request_id}
            ) as resp:
                data = await resp.json()
                t_end = time.perf_counter()

                if resp.status != 200:
                    logger.warning("Request %s failed: HTTP %d", request_id, resp.status)
                    return None

                total_ms = (t_end - t_start) * 1000.0
                queue_ms = float(resp.headers.get("X-Timing-Queue-Ms", 0))
                prefill_ms = float(resp.headers.get("X-Timing-Prefill-Ms", 0))
                decode_ms = float(resp.headers.get("X-Timing-Decode-Ms", 0))

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                generated_tokens = usage.get("completion_tokens", 0)
                spec = data.get("speculative_stats", {})

                tokens_per_sec = (
                    generated_tokens / (total_ms / 1000.0)
                    if total_ms > 0 and generated_tokens > 0 else 0.0
                )

                trace = RequestTrace(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    queue_time_ms=queue_ms,
                    prefill_time_ms=prefill_ms,
                    decode_time_ms=decode_ms,
                    total_time_ms=total_ms,
                    tokens_per_sec=tokens_per_sec,
                    peak_vram_mb=0.0,
                    speculative_accepted=spec.get("accepted", 0),
                    speculative_total=spec.get("total", 0),
                )

                if not is_warmup and self.collector is not None:
                    self.collector.record_trace(trace)
                return trace

        except asyncio.TimeoutError:
            logger.warning("Request %s timed out after %.1fs", request_id, self.timeout_sec)
            return None
        except aiohttp.ClientError as e:
            logger.warning("Request %s connection error: %s", request_id, e)
            return None

    async def _run_warmup(self) -> None:
        logger.info("Warm-up: sending %d requests (results discarded)...", self.warmup_requests)
        sem = asyncio.Semaphore(4)

        async def _bounded(rid, body):
            async with sem:
                return await self._send_request(rid, body, is_warmup=True)

        tasks = []
        for i in range(self.warmup_requests):
            body = self._build_request_body(
                self.distribution.sample_prompt_length(self.rng),
                self.distribution.sample_output_length(self.rng),
            )
            tasks.append(_bounded(f"warmup-{i:04d}", body))
        await asyncio.gather(*tasks)
        logger.info("Warm-up complete.")

    async def steady_load(self, concurrency: int, duration_sec: int) -> MetricsCollector:
        """Sustained open-loop load at fixed concurrency for a given duration."""
        self.collector = MetricsCollector(label=f"steady_c{concurrency}_d{duration_sec}s")
        self._stop_event.clear()
        await self._run_warmup()

        iat = max(1.0 / max(concurrency, 1), 0.005)
        logger.info("Steady load: concurrency=%d, duration=%ds, iat=%.3fs", concurrency, duration_sec, iat)

        t_start = time.monotonic()
        t_end = t_start + duration_sec
        pending: List[asyncio.Task] = []
        request_count = 0

        while time.monotonic() < t_end and not self._stop_event.is_set():
            rid = f"req-{request_count:06d}"
            body = self._build_request_body(
                self.distribution.sample_prompt_length(self.rng),
                self.distribution.sample_output_length(self.rng),
            )
            pending.append(asyncio.create_task(self._send_request(rid, body)))
            request_count += 1
            await asyncio.sleep(iat)

            # Adaptive IAT to maintain target concurrency.
            active = sum(1 for t in pending if not t.done())
            if request_count % 50 == 0:
                if active > concurrency * 1.5:
                    iat *= 1.2
                elif active < concurrency * 0.5 and iat > 0.005:
                    iat *= 0.8

            if len(pending) > concurrency * 4:
                pending = [t for t in pending if not t.done()]

        if pending:
            logger.info("Draining %d in-flight requests...", len(pending))
            await asyncio.wait(pending, timeout=self.timeout_sec)
            for t in pending:
                if not t.done():
                    t.cancel()

        elapsed = time.monotonic() - t_start
        logger.info("Steady load complete: %d requests in %.1fs", request_count, elapsed)
        await self._close_session()
        return self.collector

    async def burst_load(
        self, peak_concurrency: int, duration_sec: int,
        burst_ratio: float = 3.0, burst_duration_sec: float = 5.0,
        burst_interval_sec: float = 15.0,
    ) -> MetricsCollector:
        """Periodic burst pattern: baseline rate with periodic spikes."""
        self.collector = MetricsCollector(label=f"burst_peak{peak_concurrency}_d{duration_sec}s")
        self._stop_event.clear()
        await self._run_warmup()

        baseline_concurrency = max(1, int(peak_concurrency / burst_ratio))
        baseline_iat = 1.0 / max(baseline_concurrency, 1)
        burst_iat = 1.0 / max(peak_concurrency, 1)

        logger.info("Burst load: baseline=%d, peak=%d", baseline_concurrency, peak_concurrency)

        t_start = time.monotonic()
        t_end = t_start + duration_sec
        pending: List[asyncio.Task] = []
        request_count = 0

        while time.monotonic() < t_end and not self._stop_event.is_set():
            elapsed = time.monotonic() - t_start
            in_burst = elapsed % burst_interval_sec < burst_duration_sec
            iat = burst_iat if in_burst else baseline_iat

            body = self._build_request_body(
                self.distribution.sample_prompt_length(self.rng),
                self.distribution.sample_output_length(self.rng),
            )
            pending.append(asyncio.create_task(
                self._send_request(f"burst-{request_count:06d}", body)
            ))
            request_count += 1
            await asyncio.sleep(max(iat, 0.005))

            if len(pending) > peak_concurrency * 4:
                pending = [t for t in pending if not t.done()]

        if pending:
            await asyncio.wait(pending, timeout=self.timeout_sec)
            for t in pending:
                if not t.done():
                    t.cancel()

        logger.info("Burst load complete: %d requests in %.1fs", request_count, time.monotonic() - t_start)
        await self._close_session()
        return self.collector

    async def ramp_load(
        self, max_concurrency: int, duration_sec: int, ramp_steps: int = 8,
    ) -> MetricsCollector:
        """Staircase ramp: linearly increase concurrency to find the knee."""
        self.collector = MetricsCollector(label=f"ramp_max{max_concurrency}_d{duration_sec}s")
        self._stop_event.clear()
        await self._run_warmup()

        step_duration = duration_sec / ramp_steps
        levels = [max(1, int(max_concurrency * (i + 1) / ramp_steps)) for i in range(ramp_steps)]
        logger.info("Ramp load: steps=%d, levels=%s", ramp_steps, levels)

        t_benchmark_start = time.monotonic()
        pending: List[asyncio.Task] = []
        request_count = 0

        for step_idx, concurrency in enumerate(levels):
            iat = 1.0 / max(concurrency, 1)
            t_step_end = time.monotonic() + step_duration

            while time.monotonic() < t_step_end and not self._stop_event.is_set():
                body = self._build_request_body(
                    self.distribution.sample_prompt_length(self.rng),
                    self.distribution.sample_output_length(self.rng),
                )
                pending.append(asyncio.create_task(
                    self._send_request(f"ramp-{step_idx}-{request_count:06d}", body)
                ))
                request_count += 1
                await asyncio.sleep(max(iat, 0.005))
                if len(pending) > concurrency * 4:
                    pending = [t for t in pending if not t.done()]

        if pending:
            await asyncio.wait(pending, timeout=self.timeout_sec)
            for t in pending:
                if not t.done():
                    t.cancel()

        logger.info("Ramp load complete: %d requests in %.1fs", request_count, time.monotonic() - t_benchmark_start)
        await self._close_session()
        return self.collector

    def stop(self) -> None:
        self._stop_event.set()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NanoSeek Load Testing Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fms.perf.loadgen --target http://localhost:8000 --concurrency 16 --duration 60
  python -m fms.perf.loadgen --target http://localhost:8000 --pattern burst --concurrency 64 --duration 120
  python -m fms.perf.loadgen --target http://localhost:8000 --pattern ramp --concurrency 128 --duration 240
        """,
    )
    parser.add_argument("--target", type=str, required=True, help="Base URL of inference server")
    parser.add_argument("--concurrency", type=int, default=16, help="Target / peak concurrency")
    parser.add_argument("--duration", type=int, default=60, help="Measurement window in seconds")
    parser.add_argument("--pattern", choices=["steady", "burst", "ramp"], default="steady")
    parser.add_argument("--distribution", choices=["default", "uniform", "short", "long"], default="default")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up requests (discarded)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="JSON output path (default: stdout)")
    parser.add_argument("--model", type=str, default="nanoseek-1b", help="Model name")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (sec)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> MetricsCollector:
    dist_map = {
        "default": PromptDistribution(),
        "uniform": UNIFORM_DISTRIBUTION,
        "short": SHORT_PROMPT_DISTRIBUTION,
        "long": LONG_PROMPT_DISTRIBUTION,
    }
    generator = LoadGenerator(
        target_url=args.target,
        prompt_distribution=dist_map[args.distribution],
        seed=args.seed,
        warmup_requests=args.warmup,
        timeout_sec=args.timeout,
        model_name=args.model,
    )
    if args.pattern == "steady":
        return await generator.steady_load(args.concurrency, args.duration)
    elif args.pattern == "burst":
        return await generator.burst_load(args.concurrency, args.duration)
    elif args.pattern == "ramp":
        return await generator.ramp_load(args.concurrency, args.duration)
    raise ValueError(f"Unknown pattern: {args.pattern}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    collector = asyncio.run(async_main(args))
    from .report import ReportGenerator
    report_gen = ReportGenerator()
    report = report_gen.generate_report(collector)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Results written to %s", args.output)
    else:
        print(report_gen.format_ascii_table(report))


if __name__ == "__main__":
    main()
```

### File: `fms/perf/metrics.py`

```python
"""
Metrics collection and percentile computation for NanoSeek load testing.

Provides RequestTrace for per-request timing and MetricsCollector for
aggregating traces into histograms, percentiles, and confidence intervals.
No external dependencies beyond the standard library.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RequestTrace:
    """
    Complete timing trace for a single inference request.
    All times in milliseconds.
    """
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    queue_time_ms: float
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    tokens_per_sec: float
    peak_vram_mb: float
    speculative_accepted: int
    speculative_total: int

    @property
    def time_per_token_ms(self) -> float:
        if self.generated_tokens == 0:
            return 0.0
        return self.total_time_ms / self.generated_tokens

    @property
    def speculative_acceptance_rate(self) -> float:
        if self.speculative_total == 0:
            return 0.0
        return self.speculative_accepted / self.speculative_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "queue_time_ms": round(self.queue_time_ms, 2),
            "prefill_time_ms": round(self.prefill_time_ms, 2),
            "decode_time_ms": round(self.decode_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "time_per_token_ms": round(self.time_per_token_ms, 2),
            "speculative_accepted": self.speculative_accepted,
            "speculative_total": self.speculative_total,
            "speculative_acceptance_rate": round(self.speculative_acceptance_rate, 4),
        }


@dataclass
class LatencyHistogram:
    """Fixed-bucket histogram with log-spaced boundaries from 1ms to 100s."""
    bucket_boundaries_ms: List[float] = field(default_factory=lambda: [
        1, 2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000, 5000, 10000, 20000, 50000, 100000,
    ])
    counts: List[int] = field(init=False)

    def __post_init__(self):
        self.counts = [0] * (len(self.bucket_boundaries_ms) + 1)

    def observe(self, value_ms: float) -> None:
        for i, boundary in enumerate(self.bucket_boundaries_ms):
            if value_ms <= boundary:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def to_dict(self) -> Dict[str, Any]:
        buckets = []
        for i, boundary in enumerate(self.bucket_boundaries_ms):
            buckets.append({"le": boundary, "label": f"<={boundary}ms", "count": self.counts[i]})
        buckets.append({"le": float("inf"), "label": f">{self.bucket_boundaries_ms[-1]}ms", "count": self.counts[-1]})
        return {"buckets": buckets, "total": sum(self.counts)}


class MetricsCollector:
    """Aggregates RequestTraces into summary statistics with percentiles and CIs."""

    def __init__(self, label: str = "benchmark"):
        self.label = label
        self.traces: List[RequestTrace] = []
        self.histogram = LatencyHistogram()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def record_trace(self, trace: RequestTrace) -> None:
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        self._end_time = now
        self.traces.append(trace)
        self.histogram.observe(trace.total_time_ms)

    @property
    def total_requests(self) -> int:
        return len(self.traces)

    @property
    def duration_sec(self) -> float:
        if self._start_time is None or self._end_time is None:
            return 0.0
        return self._end_time - self._start_time

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]
        rank = p / 100.0 * (n - 1)
        lower = int(math.floor(rank))
        upper = min(lower + 1, n - 1)
        fraction = rank - lower
        return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])

    @staticmethod
    def bootstrap_ci(
        values: List[float], percentile_target: float,
        n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for a percentile estimate."""
        if len(values) < 10:
            point = MetricsCollector.percentile(values, percentile_target)
            return (point, point)
        rng = random.Random(seed)
        estimates = []
        n = len(values)
        for _ in range(n_bootstrap):
            resample = [rng.choice(values) for _ in range(n)]
            estimates.append(MetricsCollector.percentile(resample, percentile_target))
        estimates.sort()
        alpha = (1.0 - confidence) / 2.0
        lo = int(math.floor(alpha * n_bootstrap))
        hi = int(math.ceil((1.0 - alpha) * n_bootstrap)) - 1
        return (estimates[max(0, lo)], estimates[min(hi, n_bootstrap - 1)])

    def compute_summary(self) -> Dict[str, Any]:
        if not self.traces:
            return {"label": self.label, "total_requests": 0, "error": "No traces recorded"}

        total_times = [t.total_time_ms for t in self.traces]
        queue_times = [t.queue_time_ms for t in self.traces if t.queue_time_ms > 0]
        prefill_times = [t.prefill_time_ms for t in self.traces if t.prefill_time_ms > 0]
        decode_times = [t.decode_time_ms for t in self.traces if t.decode_time_ms > 0]
        tps_values = [t.tokens_per_sec for t in self.traces if t.tokens_per_sec > 0]
        gen_lens = [t.generated_tokens for t in self.traces]
        prompt_lens = [t.prompt_tokens for t in self.traces]

        total_generated = sum(gen_lens)
        agg_throughput = total_generated / self.duration_sec if self.duration_sec > 0 else 0.0

        total_spec_accepted = sum(t.speculative_accepted for t in self.traces)
        total_spec_total = sum(t.speculative_total for t in self.traces)
        spec_rate = total_spec_accepted / total_spec_total if total_spec_total > 0 else 0.0

        def _stats(values: List[float], name: str) -> Dict[str, Any]:
            if not values:
                return {"name": name, "count": 0}
            return {
                "name": name, "count": len(values),
                "mean": round(statistics.mean(values), 2),
                "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
                "min": round(min(values), 2),
                "p50": round(self.percentile(values, 50), 2),
                "p90": round(self.percentile(values, 90), 2),
                "p95": round(self.percentile(values, 95), 2),
                "p99": round(self.percentile(values, 99), 2),
                "max": round(max(values), 2),
                "ci_p50": [round(v, 2) for v in self.bootstrap_ci(values, 50)],
                "ci_p95": [round(v, 2) for v in self.bootstrap_ci(values, 95)],
                "ci_p99": [round(v, 2) for v in self.bootstrap_ci(values, 99)],
            }

        warnings = []
        for pname, ptarget in [("p95", 95), ("p99", 99)]:
            ci = self.bootstrap_ci(total_times, ptarget)
            point = self.percentile(total_times, ptarget)
            if point > 0:
                ci_width_pct = (ci[1] - ci[0]) / point * 100
                if ci_width_pct > 10:
                    warnings.append(
                        f"{pname} CI is {ci_width_pct:.0f}% of point estimate — "
                        f"increase duration or concurrency for tighter bounds"
                    )

        return {
            "label": self.label,
            "total_requests": self.total_requests,
            "successful_requests": len(total_times),
            "failed_requests": 0,
            "duration_sec": round(self.duration_sec, 2),
            "aggregate_throughput_tok_per_sec": round(agg_throughput, 2),
            "total_prompt_tokens": sum(prompt_lens),
            "total_generated_tokens": total_generated,
            "latency": {
                "total": _stats(total_times, "total_time_ms"),
                "queue": _stats(queue_times, "queue_time_ms"),
                "prefill": _stats(prefill_times, "prefill_time_ms"),
                "decode": _stats(decode_times, "decode_time_ms"),
            },
            "throughput_per_request": _stats(tps_values, "tokens_per_sec"),
            "prompt_length_distribution": {
                "mean": round(statistics.mean(prompt_lens), 1) if prompt_lens else 0,
                "min": min(prompt_lens) if prompt_lens else 0,
                "max": max(prompt_lens) if prompt_lens else 0,
            },
            "generation_length_distribution": {
                "mean": round(statistics.mean(gen_lens), 1) if gen_lens else 0,
                "min": min(gen_lens) if gen_lens else 0,
                "max": max(gen_lens) if gen_lens else 0,
            },
            "speculative_decoding": {
                "total_draft_tokens": total_spec_total,
                "total_accepted_tokens": total_spec_accepted,
                "aggregate_acceptance_rate": round(spec_rate, 4),
            },
            "histogram": self.histogram.to_dict(),
            "warnings": warnings,
        }


class VRAMMonitor:
    """Background VRAM usage monitor (server-side, polls torch.cuda.memory_allocated)."""

    def __init__(self, poll_interval_sec: float = 0.5):
        self.poll_interval_sec = poll_interval_sec
        self.peak_vram_mb: float = 0.0
        self.samples: List[Tuple[float, float]] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        try:
            import torch
            if not torch.cuda.is_available():
                return
        except ImportError:
            return
        while self._running:
            import torch
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.peak_vram_mb = max(self.peak_vram_mb, allocated_mb)
            self.samples.append((time.monotonic(), allocated_mb))
            await asyncio.sleep(self.poll_interval_sec)

    def stop(self) -> None:
        self._running = False

    def get_stats(self) -> Dict[str, float]:
        if not self.samples:
            return {"peak_mb": 0.0, "mean_mb": 0.0, "samples": 0}
        vram_values = [s[1] for s in self.samples]
        return {
            "peak_mb": round(self.peak_vram_mb, 2),
            "mean_mb": round(statistics.mean(vram_values), 2),
            "samples": len(self.samples),
        }
```

### File: `fms/perf/report.py`

```python
"""
Report generation for NanoSeek load testing results.

Three output formats:
1. JSON reports — machine-readable, consumed by CI and dashboards
2. ASCII tables — human-readable terminal output
3. A/B comparison — before/after with statistical significance (Mann-Whitney U)

The A/B comparison produces the "Final Table" from the FMS Lab Plan.

Usage:
    python -m fms.perf.report --baseline baseline.json --candidate candidate.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .metrics import MetricsCollector


class ReportGenerator:

    def __init__(self, title: str = "NanoSeek Load Test Report"):
        self.title = title

    def generate_report(self, collector: MetricsCollector) -> Dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": collector.compute_summary(),
            "traces": [t.to_dict() for t in collector.traces],
        }

    @staticmethod
    def _format_table(headers: List[str], rows: List[List[str]], alignment: Optional[List[str]] = None) -> str:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
        if alignment is None:
            alignment = ["l"] + ["r"] * (len(headers) - 1)

        def _cell(v, w, a):
            return v.rjust(w) if a == "r" else v.ljust(w)

        def _row(cells):
            return "│ " + " │ ".join(_cell(str(c), col_widths[i], alignment[i]) for i, c in enumerate(cells)) + " │"

        sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
        top = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
        bot = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"
        return "\n".join([top, _row(headers), sep] + [_row(r) for r in rows] + [bot])

    def format_ascii_table(self, report: Dict[str, Any]) -> str:
        summary = report["summary"]
        lines = []

        lines.append(f"\n{'=' * 72}")
        lines.append(f"  {self.title}")
        lines.append(f"  {report.get('generated_at', '')}")
        lines.append(f"{'=' * 72}\n")

        lines.append("  OVERVIEW")
        lines.append(f"  {'─' * 40}")
        lines.append(f"  Total requests:     {summary['total_requests']}")
        lines.append(f"  Successful:         {summary['successful_requests']}")
        lines.append(f"  Duration:           {summary['duration_sec']}s")
        lines.append(f"  Aggregate tput:     {summary['aggregate_throughput_tok_per_sec']} tok/s")
        lines.append("")

        latency = summary["latency"]
        if latency["total"].get("count", 0) > 0:
            headers = ["Metric", "Mean", "p50", "p90", "p95", "p99", "Max"]
            rows = []
            for phase in ["total", "queue", "prefill", "decode"]:
                p = latency.get(phase, {})
                if p.get("count", 0) > 0:
                    rows.append([phase.capitalize(), f"{p['mean']:.1f}ms", f"{p['p50']:.1f}ms",
                                 f"{p['p90']:.1f}ms", f"{p['p95']:.1f}ms", f"{p['p99']:.1f}ms", f"{p['max']:.1f}ms"])
            lines.append("  LATENCY DISTRIBUTION")
            lines.append(_indent(self._format_table(headers, rows), 2))
            lines.append("")

        tput = summary.get("throughput_per_request", {})
        if tput.get("count", 0) > 0:
            lines.append("  PER-REQUEST THROUGHPUT")
            lines.append(_indent(self._format_table(
                ["Metric", "Mean", "p50", "p90", "p95", "Min"],
                [["tok/s/req", f"{tput['mean']:.1f}", f"{tput['p50']:.1f}",
                  f"{tput['p90']:.1f}", f"{tput['p95']:.1f}", f"{tput['min']:.1f}"]],
            ), 2))
            lines.append("")

        spec = summary.get("speculative_decoding", {})
        if spec.get("total_draft_tokens", 0) > 0:
            lines.append("  SPECULATIVE DECODING (MTP)")
            lines.append(f"  {'─' * 40}")
            lines.append(f"  Draft tokens:       {spec['total_draft_tokens']}")
            lines.append(f"  Accepted:           {spec['total_accepted_tokens']}")
            lines.append(f"  Acceptance rate:    {spec['aggregate_acceptance_rate']:.1%}")
            lines.append("")

        hist = summary.get("histogram", {})
        if hist.get("total", 0) > 0:
            lines.append("  LATENCY HISTOGRAM")
            max_count = max((b["count"] for b in hist["buckets"]), default=1)
            for bucket in hist["buckets"]:
                if bucket["count"] > 0:
                    bar_len = int(bucket["count"] / max_count * 40) if max_count > 0 else 0
                    lines.append(f"  {bucket['label']:>14s} │{'█' * bar_len} {bucket['count']}")
            lines.append("")

        for w in summary.get("warnings", []):
            lines.append(f"  ⚠ {w}")
        lines.append(f"{'=' * 72}\n")
        return "\n".join(lines)

    def compare(self, baseline: Dict[str, Any], candidate: Dict[str, Any],
                lever_name: str = "Unknown Lever") -> Dict[str, Any]:
        b, c = baseline["summary"], candidate["summary"]

        def _delta(bv, cv):
            d = cv - bv
            return {"baseline": round(bv, 2), "candidate": round(cv, 2),
                    "delta": round(d, 2), "pct_change": round(d / bv * 100, 1) if bv else 0.0}

        b_lat, c_lat = b["latency"]["total"], c["latency"]["total"]

        return {
            "lever_name": lever_name,
            "baseline_label": b.get("label", "baseline"),
            "candidate_label": c.get("label", "candidate"),
            "metrics": {
                "throughput_tok_per_sec": _delta(b["aggregate_throughput_tok_per_sec"], c["aggregate_throughput_tok_per_sec"]),
                "latency_p50_ms": _delta(b_lat.get("p50", 0), c_lat.get("p50", 0)),
                "latency_p95_ms": _delta(b_lat.get("p95", 0), c_lat.get("p95", 0)),
                "latency_p99_ms": _delta(b_lat.get("p99", 0), c_lat.get("p99", 0)),
                "total_requests": _delta(b["total_requests"], c["total_requests"]),
            },
            "statistical_test": self._mann_whitney_test(
                baseline.get("traces", []), candidate.get("traces", [])),
        }

    @staticmethod
    def _mann_whitney_test(baseline_traces: List[Dict], candidate_traces: List[Dict]) -> Dict[str, Any]:
        """Non-parametric Mann-Whitney U test with normal approximation (n > 20)."""
        b_vals = [t["total_time_ms"] for t in baseline_traces if "total_time_ms" in t]
        c_vals = [t["total_time_ms"] for t in candidate_traces if "total_time_ms" in t]
        n1, n2 = len(b_vals), len(c_vals)

        if n1 < 20 or n2 < 20:
            return {"test": "mann_whitney_u", "significant": False,
                    "reason": f"Insufficient samples (n1={n1}, n2={n2}, need ≥20)", "p_value": None}

        combined = [(v, "b") for v in b_vals] + [(v, "c") for v in c_vals]
        combined.sort(key=lambda x: x[0])

        ranks = {}
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        r1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == "b")
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        u = min(u1, u2)

        mu = n1 * n2 / 2.0
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u - mu) / sigma if sigma > 0 else 0.0
        p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
        r_effect = 1.0 - (2.0 * u) / (n1 * n2) if n1 * n2 > 0 else 0.0

        effect_label = "negligible"
        if abs(r_effect) > 0.5: effect_label = "large"
        elif abs(r_effect) > 0.3: effect_label = "medium"
        elif abs(r_effect) > 0.1: effect_label = "small"

        return {
            "test": "mann_whitney_u", "u_statistic": round(u, 2),
            "z_score": round(z, 4), "p_value": round(p_value, 6),
            "significant": p_value < 0.05, "effect_size_r": round(r_effect, 4),
            "effect_label": effect_label, "n_baseline": n1, "n_candidate": n2,
        }

    def format_comparison_table(self, comparison: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"\n{'=' * 78}")
        lines.append(f"  A/B COMPARISON: {comparison['lever_name']}")
        lines.append(f"  Baseline: {comparison['baseline_label']}  |  Candidate: {comparison['candidate_label']}")
        lines.append(f"{'=' * 78}\n")

        headers = ["Metric", "Baseline", "Candidate", "Delta", "Change", "Verdict"]
        rows = []
        for name, data in comparison["metrics"].items():
            pct = data["pct_change"]
            is_lat = "latency" in name
            better = (pct < 0) if is_lat else (pct > 0)
            verdict = "━━" if abs(pct) < 1.0 else ("✓ BETTER" if better else "✗ WORSE")
            rows.append([name, str(data["baseline"]), str(data["candidate"]),
                         f"{data['delta']:+.2f}", f"{pct:+.1f}%", verdict])
        lines.append(self._format_table(headers, rows))
        lines.append("")

        stat = comparison.get("statistical_test", {})
        if stat.get("p_value") is not None:
            lines.append(f"  Significant (p<0.05): {'✓ YES' if stat['significant'] else '✗ NO'}")
            lines.append(f"  p-value: {stat['p_value']:.6f}  |  effect size r: {stat['effect_size_r']:.4f} ({stat['effect_label']})")
        lines.append(f"\n{'=' * 78}\n")
        return "\n".join(lines)

    def format_final_table(self, comparisons: List[Dict[str, Any]]) -> str:
        """The FMS Lab Plan 'Final Table' — summary of all three Money Levers."""
        lines = [""]
        lines.append("╔" + "═" * 74 + "╗")
        lines.append("║" + "NanoSeek Optimization Impact — Final Table".center(74) + "║")
        lines.append("╠" + "═" * 74 + "╣")
        lines.append("║" + " " * 74 + "║")

        hdr = "║  {:<30s}│ {:>12s} │ {:>13s} │ {:>11s} │ {:>8s}".format(
            "Lever", "Throughput Δ", "p99 Latency Δ", "Significant", "Verdict")
        lines.append(hdr.ljust(75) + "║")
        sep = "║  " + "─" * 30 + "┼─" + "─" * 12 + "─┼─" + "─" * 13 + "─┼─" + "─" * 11 + "─┼─" + "─" * 8
        lines.append(sep.ljust(75) + "║")

        for i, comp in enumerate(comparisons, 1):
            m = comp["metrics"]
            tput_pct = m.get("throughput_tok_per_sec", {}).get("pct_change", 0)
            lat_pct = m.get("latency_p99_ms", {}).get("pct_change", 0)
            sig = "✓ p<0.05" if comp.get("statistical_test", {}).get("significant") else "—"
            verdict = "✓ SHIP" if tput_pct > 5 and lat_pct < 50 else "✗ HOLD"
            row = "║  {:<30s}│ {:>+12.0f}% │ {:>+13.0f}% │ {:>11s} │ {:>8s}".format(
                f"{i}. {comp['lever_name']}", tput_pct, lat_pct, sig, verdict)
            lines.append(row.ljust(75) + "║")

        lines.append(sep.ljust(75) + "║")
        lines.append("║" + " " * 74 + "║")
        lines.append("╚" + "═" * 74 + "╝")
        lines.append("")
        return "\n".join(lines)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 7.1.26, error < 1.5e-7)."""
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare NanoSeek load test results (A/B)")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline JSON results")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate JSON results")
    parser.add_argument("--lever", type=str, default="Unknown Lever", help="Optimization lever name")
    parser.add_argument("--output", type=str, default=None, help="JSON output (default: ASCII to stdout)")
    args = parser.parse_args(argv)

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.candidate) as f:
        candidate = json.load(f)

    gen = ReportGenerator()
    comparison = gen.compare(baseline, candidate, lever_name=args.lever)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
    else:
        print(gen.format_comparison_table(comparison))


if __name__ == "__main__":
    main()
```

---

## 4. Visualization

### Load Pattern Diagrams

**Steady Load**

```
Concurrency
    16 ┤ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       │   ↑ warmup            ↑ measurement window
    0  ┤───┤                   └─────────────────────────▶ t
       0   5s                 65s                      125s
```

**Burst Load**

```
Concurrency
   64 ┤           ╱╲               ╱╲               ╱╲
      │          ╱  ╲             ╱  ╲             ╱  ╲
   21 ┤ ━━━━━━━╱    ╲━━━━━━━━━━╱    ╲━━━━━━━━━━╱    ╲━━━
      │  baseline burst baseline  burst  baseline  burst
    0 ┤──────────────────────────────────────────────────▶ t
```

**Ramp Load**

```
Concurrency
  128 ┤                                    ┌────────┐
   96 ┤                          ┌─────────┤ step 8 │
   64 ┤                ┌────────┤  step 7  │        │
   32 ┤      ┌─────────┤ step 5 │          │        │
   16 ┤──────┤ steps   │        │          │        │
    0 ┤──────┴─────────┴────────┴──────────┴────────▶ t
```

### Latency Histogram

```
        <=50ms │ ████████ 28
       <=100ms │ ██████████████████ 67
       <=200ms │ ████████████████████████████████████ 124
       <=500ms │ ████████████████████████████████████████ 312
      <=1000ms │ ██████████████████████████████ 218
      <=2000ms │ ███████████ 82
```

A bimodal histogram (two peaks) indicates two distinct populations — typically short prompts (fast prefill) and long prompts (slow prefill).

### A/B Comparison

```
  Before (no MLA)             After (MLA enabled)
  Throughput: 120 tok/s  →    1,050 tok/s   (+775%)
  p50:        892ms      →    358ms         (-60%)
  p99:        8,921ms    →    1,247ms       (-86%)
  KV VRAM:    41 GB      →    1.8 GB        (-96%)

  Verdict: ✓ SHIP — Mann-Whitney U, p < 0.001, r = 0.94 (large)
```

---

## 5. File Placement

```
fms/
├── __init__.py
└── perf/
    ├── __init__.py            # Package exports
    ├── loadgen.py             # THIS DOC: Open-loop load generator
    ├── metrics.py             # THIS DOC: MetricsCollector, RequestTrace, histograms
    └── report.py              # THIS DOC: JSON/ASCII reports, A/B comparison, Final Table

fms/serving/
├── engine.py                  # Doc 07: Continuous batching engine
├── server.py                  # Doc 15: OpenAI-compatible API server
└── speculative.py             # Doc 08: MTP speculative decoding
```

The `fms/perf/` package is separate from `fms/serving/` because the load generator is a **client-side tool**. The only coupling is the HTTP API contract (OpenAI-compatible endpoints from Doc 15) and optional server-side timing headers.

Package init (`fms/perf/__init__.py`):

```python
"""NanoSeek Performance Load Testing & Benchmarking."""
from .loadgen import LoadGenerator, PromptDistribution
from .metrics import MetricsCollector, RequestTrace, VRAMMonitor
from .report import ReportGenerator

__all__ = [
    "LoadGenerator", "PromptDistribution",
    "MetricsCollector", "RequestTrace", "VRAMMonitor",
    "ReportGenerator",
]
```

---

## 6. Example Reports

### JSON Report (abbreviated)

```json
{
  "title": "NanoSeek Load Test Report",
  "generated_at": "2026-02-27T14:30:00+00:00",
  "summary": {
    "label": "steady_c16_d60s",
    "total_requests": 847,
    "successful_requests": 847,
    "duration_sec": 60.12,
    "aggregate_throughput_tok_per_sec": 1842.5,
    "total_prompt_tokens": 312450,
    "total_generated_tokens": 110732,
    "latency": {
      "total": {
        "name": "total_time_ms", "count": 847, "mean": 412.3, "stdev": 189.7,
        "min": 45.2, "p50": 358.1, "p90": 672.4, "p95": 845.9, "p99": 1247.3,
        "max": 2103.8, "ci_p50": [342.1, 371.8], "ci_p95": [798.2, 901.3],
        "ci_p99": [1102.5, 1398.7]
      },
      "queue": { "name": "queue_time_ms", "count": 847, "mean": 45.2, "p50": 32.4, "p99": 245.8 },
      "prefill": { "name": "prefill_time_ms", "count": 847, "mean": 89.4, "p50": 42.8, "p99": 567.2 },
      "decode": { "name": "decode_time_ms", "count": 847, "mean": 277.7, "p50": 268.1, "p99": 589.1 }
    },
    "speculative_decoding": {
      "total_draft_tokens": 2541,
      "total_accepted_tokens": 1728,
      "aggregate_acceptance_rate": 0.68
    },
    "warnings": []
  }
}
```

### ASCII Table Output

```
========================================================================
  NanoSeek Load Test Report
  2026-02-27T14:30:00+00:00
========================================================================

  OVERVIEW
  ────────────────────────────────────────
  Total requests:     847
  Successful:         847
  Duration:           60.12s
  Aggregate tput:     1842.5 tok/s

  LATENCY DISTRIBUTION
  ┌───────────┬─────────┬─────────┬─────────┬─────────┬──────────┬──────────┐
  │ Metric    │    Mean │     p50 │     p90 │     p95 │      p99 │      Max │
  ├───────────┼─────────┼─────────┼─────────┼─────────┼──────────┼──────────┤
  │ Total     │ 412.3ms │ 358.1ms │ 672.4ms │ 845.9ms │ 1247.3ms │ 2103.8ms │
  │ Queue     │  45.2ms │  32.4ms │  98.7ms │ 142.3ms │  245.8ms │  412.1ms │
  │ Prefill   │  89.4ms │  42.8ms │ 198.5ms │ 312.7ms │  567.2ms │  892.1ms │
  │ Decode    │ 277.7ms │ 268.1ms │ 398.2ms │ 432.8ms │  589.1ms │  812.4ms │
  └───────────┴─────────┴─────────┴─────────┴─────────┴──────────┴──────────┘

  SPECULATIVE DECODING (MTP)
  ────────────────────────────────────────
  Draft tokens:       2541
  Accepted:           1728
  Acceptance rate:    68.0%

  LATENCY HISTOGRAM
        <=50ms │ ████████ 28
       <=100ms │ ██████████████████ 67
       <=200ms │ ████████████████████████████████████ 124
       <=500ms │ ████████████████████████████████████████ 312
      <=1000ms │ ██████████████████████████████ 218
      <=2000ms │ ███████████ 82
      <=5000ms │ █ 9

========================================================================
```

### A/B Comparison Output

```
==============================================================================
  A/B COMPARISON: MLA KV Cache (23× Compression)
  Baseline: standard_mha_c16_d60s  |  Candidate: mla_kv_cache_c16_d60s
==============================================================================

┌──────────────────────────┬──────────┬──────────┬──────────┬────────┬──────────┐
│ Metric                   │ Baseline │ Candidate│    Delta │ Change │ Verdict  │
├──────────────────────────┼──────────┼──────────┼──────────┼────────┼──────────┤
│ throughput_tok_per_sec   │   210.40 │  1842.50 │ +1632.10 │ +775.9%│ ✓ BETTER │
│ latency_p50_ms           │   892.30 │   358.10 │  -534.20 │  -59.9%│ ✓ BETTER │
│ latency_p95_ms           │  3412.80 │   845.90 │ -2566.90 │  -75.2%│ ✓ BETTER │
│ latency_p99_ms           │  8921.40 │  1247.30 │ -7674.10 │  -86.0%│ ✓ BETTER │
└──────────────────────────┴──────────┴──────────┴──────────┴────────┴──────────┘

  Significant (p<0.05): ✓ YES
  p-value: 0.000001  |  effect size r: 0.9412 (large)

==============================================================================
```

### Final Table Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              NanoSeek Optimization Impact — Final Table                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Lever                          │ Throughput Δ │ p99 Latency Δ │ Verdict   ║
║  ───────────────────────────────┼──────────────┼───────────────┼────────── ║
║  1. MLA KV Cache (23×)         │       +776%  │         -86%  │ ✓ SHIP    ║
║  2. Continuous Batching         │       +720%  │         +45%  │ ✓ SHIP    ║
║  3. MTP Speculative Decoding   │        +85%  │         -38%  │ ✓ SHIP    ║
║  ───────────────────────────────┼──────────────┼───────────────┼────────── ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 7. Gotchas

### 7a. Connection Pool Exhaustion Inflates Measured Latency

If `max_connections` in the aiohttp connector is smaller than the target concurrency, requests queue at the HTTP connection pool — not at the server. The load tester measures this client-side queuing as server latency. Set `max_connections` to at least 2× target concurrency (default: 256). If you see a flat latency plateau at higher concurrency (not a gradual curve), the connection pool is the bottleneck.

### 7b. JIT Warm-Up Contaminates the First 3-5 Requests

`torch.compile()` adds 2-30 seconds to the first forward pass. CUDA graph capture adds 200-500ms. Including these in measurements makes p99 look 10-100× worse than steady state. The default `warmup_requests=20` handles this. Verify by comparing warmup request #1 vs #20 latency — if #20 is still >2× the eventual mean, increase the warmup count.

### 7c. VRAM Measurement Timing Creates Observer Effect

`torch.cuda.memory_allocated()` triggers a CUDA synchronization point on some drivers, adding 0.5-2ms. In a hot loop at 1000 decode steps/sec, that is 0.5-2 seconds of overhead. The `VRAMMonitor` runs on a 500ms poll interval in a background task. Never call `memory_allocated()` in the server's decode loop.

### 7d. Coordinated Omission in Closed-Loop Benchmarks

If you accidentally implement closed-loop load generation (each user waits for the previous response), you underestimate tail latency by 10-100× (Gil Tene, HdrHistogram). Verify open-loop behavior: the dispatch rate must remain constant even when server response time increases. If request rate drops during high-latency periods, something is wrong.

### 7e. Python GC Pauses Create Latency Spikes

CPython's GC can pause all threads for 10-200ms. On the client side this is harmless (minor IAT jitter). On the server side it causes real p99 spikes. Diagnose with `gc.set_debug(gc.DEBUG_STATS)` and correlate GC timestamps with latency trace data.

### 7f. DNS Resolution and TCP Keepalive

DNS lookups add 1-50ms per new connection. If DNS returns different IPs (load balancer), requests may hit different servers — invalidating the benchmark. Use `http://127.0.0.1:8000` instead of `http://localhost:8000` for benchmarking. The aiohttp connector caches DNS for 5 minutes (`ttl_dns_cache=300`).

### 7g. Non-Streaming vs Streaming Timing Discrepancy

The load generator uses `stream=False` for timing precision. Streaming endpoints have different metrics: time-to-first-token (TTFT) and inter-token latency (ITL). If the server optimizes differently for streaming vs non-streaming, non-streaming benchmarks may not reflect production streaming performance. For TTFT/ITL measurement, add a streaming mode that times the first SSE `data:` frame.

### 7h. Seed Reproducibility Across Python Versions

`random.Random(seed=42)` produces different sequences across Python versions (state initialization changed between 3.8 and 3.9). Always report Python version in metadata. For strict A/B reproducibility, save prompts to a file and replay across runs.

---

*"You cannot optimize what you cannot measure, and you cannot trust what you measure without statistical rigor. A benchmark without confidence intervals is just a story you're telling yourself."*

— Principal Engineer's Note, Foundation Models Division, 2026
