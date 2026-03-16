"""
Data curation pipeline runner.

Processes ClimbMix-400B parquet shards through:
1. Heuristic text quality filters
2. fastText quality classifier
3. SimHash near-duplicate detection
4. Domain mixture control

Outputs filtered parquets with identical schema for drop-in replacement.

Usage:
    python -m nanoseek.data_curation.run_pipeline \
        --input-dir ~/.cache/nanochat/base_data_climbmix \
        --output-dir ~/.cache/nanochat/filtered_climbmix \
        --quality-threshold 2.0 \
        --num-workers 4 \
        --shards 0-100
"""

import os
import json
import time
import argparse
import logging
from multiprocessing import Pool, Manager
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq

from nanoseek.data_curation.heuristic_filters import filter_document, FilterThresholds
from nanoseek.data_curation.quality_classifier import classify_quality
from nanoseek.data_curation.dedup import compute_simhash, SimHashIndex
from nanoseek.data_curation.mixture import classify_domain, MixtureController, DomainMixture

logger = logging.getLogger(__name__)


def _parse_shard_range(shard_str: str, max_shard: int) -> list[int]:
    """Parse shard range string like '0-100' or '0,5,10' or 'all'."""
    if shard_str == 'all':
        return list(range(max_shard + 1))
    if '-' in shard_str and ',' not in shard_str:
        start, end = shard_str.split('-')
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in shard_str.split(',')]


def process_shard(
    shard_path: str,
    output_dir: str,
    quality_threshold: float,
    hamming_threshold: int,
    dedup_index: SimHashIndex | None,
    mixture_ctrl: MixtureController | None,
    thresholds: FilterThresholds,
    skip_classifier: bool = False,
) -> dict:
    """
    Process a single parquet shard through all pipeline stages.

    Returns per-shard statistics dict.
    """
    shard_name = os.path.basename(shard_path)
    output_path = os.path.join(output_dir, shard_name)

    # Resume: skip if output already exists
    if os.path.exists(output_path):
        return {'shard': shard_name, 'status': 'skipped', 'reason': 'already_exists'}

    stats = {
        'shard': shard_name,
        'status': 'processed',
        'total_docs': 0,
        'kept_docs': 0,
        'rejected_heuristic': 0,
        'rejected_quality': 0,
        'rejected_dedup': 0,
        'rejected_mixture': 0,
        'rejection_reasons': {},
    }

    try:
        pf = pq.ParquetFile(shard_path)
    except Exception as e:
        logger.error(f"Failed to read {shard_path}: {e}")
        return {'shard': shard_name, 'status': 'error', 'error': str(e)}

    kept_texts = []

    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column('text').to_pylist()

        for text in texts:
            stats['total_docs'] += 1

            # Stage 1: Heuristic filters
            keep, metrics = filter_document(text, thresholds)
            if not keep:
                stats['rejected_heuristic'] += 1
                for reason in metrics.get('reject_reasons', []):
                    stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
                continue

            # Stage 2: Quality classifier
            if not skip_classifier:
                score = classify_quality(text)
                if score < quality_threshold:
                    stats['rejected_quality'] += 1
                    continue

            # Stage 3: Deduplication
            if dedup_index is not None:
                fingerprint = compute_simhash(text)
                if dedup_index.is_duplicate(fingerprint):
                    stats['rejected_dedup'] += 1
                    continue
                dedup_index.add(fingerprint)

            # Stage 4: Domain mixture
            if mixture_ctrl is not None:
                domain = classify_domain(text)
                if not mixture_ctrl.should_keep(domain):
                    stats['rejected_mixture'] += 1
                    continue

            kept_texts.append(text)

    stats['kept_docs'] = len(kept_texts)
    stats['keep_rate'] = len(kept_texts) / max(stats['total_docs'], 1)

    # Write output parquet with same schema
    if kept_texts:
        table = pa.table({'text': kept_texts})
        pq.write_table(
            table, output_path,
            row_group_size=1024,
            compression='zstd',
        )
        logger.info(
            f"{shard_name}: kept {len(kept_texts)}/{stats['total_docs']} "
            f"({stats['keep_rate']:.1%})"
        )
    else:
        logger.warning(f"{shard_name}: all documents filtered out!")

    return stats


def run_pipeline(args):
    """Main pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover input shards
    all_parquets = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    if args.shards != 'all':
        shard_indices = set(_parse_shard_range(args.shards, len(all_parquets) - 1))
        selected = []
        for i, f in enumerate(all_parquets):
            if i in shard_indices:
                selected.append(f)
        all_parquets = selected

    logger.info(f"Processing {len(all_parquets)} shards from {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quality threshold: {args.quality_threshold}")
    logger.info(f"Dedup hamming threshold: {args.dedup_hamming}")

    # Initialize shared components
    thresholds = FilterThresholds()
    dedup_index = SimHashIndex(hamming_threshold=args.dedup_hamming)
    mixture_ctrl = MixtureController(
        target=DomainMixture(
            code=args.mix_code, math=args.mix_math,
            science=args.mix_science, web=args.mix_web,
            other=args.mix_other,
        ),
        seed=args.seed,
    ) if not args.no_mixture else None

    # Process shards sequentially (dedup index is shared state)
    # Note: parallelism would require splitting dedup into per-shard then merging,
    # which adds complexity. Sequential is fine for second-pass filtering.
    all_stats = []
    t0 = time.time()

    for i, shard_name in enumerate(all_parquets):
        shard_path = os.path.join(args.input_dir, shard_name)
        stats = process_shard(
            shard_path=shard_path,
            output_dir=args.output_dir,
            quality_threshold=args.quality_threshold,
            hamming_threshold=args.dedup_hamming,
            dedup_index=dedup_index,
            mixture_ctrl=mixture_ctrl,
            thresholds=thresholds,
            skip_classifier=args.no_classifier,
        )
        all_stats.append(stats)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(all_parquets) - i - 1) / rate
            logger.info(
                f"Progress: {i+1}/{len(all_parquets)} shards "
                f"({rate:.1f} shards/s, ETA: {eta/60:.0f}m)"
            )

    # Aggregate statistics
    total_docs = sum(s.get('total_docs', 0) for s in all_stats)
    kept_docs = sum(s.get('kept_docs', 0) for s in all_stats)
    elapsed = time.time() - t0

    summary = {
        'total_shards': len(all_parquets),
        'processed_shards': sum(1 for s in all_stats if s['status'] == 'processed'),
        'skipped_shards': sum(1 for s in all_stats if s['status'] == 'skipped'),
        'error_shards': sum(1 for s in all_stats if s['status'] == 'error'),
        'total_docs': total_docs,
        'kept_docs': kept_docs,
        'overall_keep_rate': kept_docs / max(total_docs, 1),
        'rejected_heuristic': sum(s.get('rejected_heuristic', 0) for s in all_stats),
        'rejected_quality': sum(s.get('rejected_quality', 0) for s in all_stats),
        'rejected_dedup': sum(s.get('rejected_dedup', 0) for s in all_stats),
        'rejected_mixture': sum(s.get('rejected_mixture', 0) for s in all_stats),
        'dedup_index_size': len(dedup_index),
        'elapsed_seconds': elapsed,
        'config': {
            'quality_threshold': args.quality_threshold,
            'dedup_hamming': args.dedup_hamming,
            'skip_classifier': args.no_classifier,
            'skip_mixture': args.no_mixture,
        },
        'per_shard': all_stats,
    }

    if mixture_ctrl:
        summary['mixture_stats'] = mixture_ctrl.get_stats()

    # Write stats
    stats_path = os.path.join(args.output_dir, 'filter_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.0f}s")
    logger.info(f"Total docs: {total_docs:,}")
    logger.info(f"Kept docs:  {kept_docs:,} ({summary['overall_keep_rate']:.1%})")
    logger.info(f"  Rejected by heuristics: {summary['rejected_heuristic']:,}")
    logger.info(f"  Rejected by quality:    {summary['rejected_quality']:,}")
    logger.info(f"  Rejected by dedup:      {summary['rejected_dedup']:,}")
    logger.info(f"  Rejected by mixture:    {summary['rejected_mixture']:,}")
    logger.info(f"Dedup index size: {len(dedup_index):,} unique hashes")
    logger.info(f"Stats written to: {stats_path}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="NanoSeek data curation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-dir', required=True, help='Input parquet directory')
    parser.add_argument('--output-dir', required=True, help='Output parquet directory')
    parser.add_argument('--quality-threshold', type=float, default=2.0,
                        help='Minimum FineWeb-Edu quality score (0-5)')
    parser.add_argument('--dedup-hamming', type=int, default=3,
                        help='SimHash hamming distance threshold for dedup')
    parser.add_argument('--shards', type=str, default='all',
                        help='Shard range to process: "all", "0-100", or "0,5,10"')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for mixture sampling')
    parser.add_argument('--no-classifier', action='store_true',
                        help='Skip fastText quality classifier (faster, less filtering)')
    parser.add_argument('--no-mixture', action='store_true',
                        help='Skip domain mixture control')

    # Domain mixture weights
    parser.add_argument('--mix-code', type=float, default=0.15, help='Target code fraction')
    parser.add_argument('--mix-math', type=float, default=0.10, help='Target math fraction')
    parser.add_argument('--mix-science', type=float, default=0.15, help='Target science fraction')
    parser.add_argument('--mix-web', type=float, default=0.50, help='Target web fraction')
    parser.add_argument('--mix-other', type=float, default=0.10, help='Target other fraction')

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
