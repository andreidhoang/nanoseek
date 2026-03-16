"""
Data curation pipeline for NanoSeek.

Second-pass filtering on ClimbMix-400B (already Karpathy-curated):
- Heuristic text quality filters (FineWeb-style)
- fastText quality classifier (FineWeb-Edu)
- SimHash near-duplicate detection
- Domain mixture control

Usage:
    python -m nanoseek.data_curation.run_pipeline \
        --input-dir ~/.cache/nanochat/base_data_climbmix \
        --output-dir ~/.cache/nanochat/filtered_climbmix \
        --quality-threshold 2.0
"""
