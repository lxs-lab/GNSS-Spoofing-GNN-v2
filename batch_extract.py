"""
batch_extract.py  —  Step 1: extract rich features from all TEXBAT .bin files
==============================================================================
Run this script once before building the dataset or training.

Output: one CSV per .bin file in config.DATA_PROC_DIR, containing columns:
  Time, PRN, CN0_dBHz, Doppler_Hz, CodePhase_chips, CN0_rate, Dop_rate, PeakRatio
"""

import time
from src.extractor import FeatureExtractor
from src import config


def main():
    print('=' * 55)
    print('      GNSS Feature Extraction  (v2 — 6 features)     ')
    print('=' * 55)
    print(f"  Raw data dir : {config.DATA_RAW_DIR}")
    print(f"  Output dir   : {config.DATA_PROC_DIR}")
    print(f"  Files        : {list(config.DATA_FILES.keys())}\n")

    extractor   = FeatureExtractor()
    total_files = len(config.DATA_FILES)
    t0          = time.time()

    for i, (filename, label) in enumerate(config.DATA_FILES.items(), 1):
        print(f"\n[{i}/{total_files}] {filename}  (label={label})")
        extractor.process_single_file(filename)

    elapsed = (time.time() - t0) / 60
    print(f"\n✅  All files processed in {elapsed:.1f} min")
    print(f"📁  Results saved to: {config.DATA_PROC_DIR}")


if __name__ == '__main__':
    main()
