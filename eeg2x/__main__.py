"""允许 ``python -m eeg2x``。与 ``benchmark.py``、控制台脚本 ``eeg2x`` 三者等价。"""
import sys

from eeg2x.cli import main

if __name__ == "__main__":
    sys.exit(main())
