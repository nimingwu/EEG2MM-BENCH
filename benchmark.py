#!/usr/bin/env python3
"""EEG2X-Benchmark v3 —— 唯一入口。用法见 ``python benchmark.py --help``。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eeg2x.cli import main

if __name__ == "__main__":
    sys.exit(main())
