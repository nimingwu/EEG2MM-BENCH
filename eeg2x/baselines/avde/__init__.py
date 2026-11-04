"""AVDE baseline (Dai et al., ICLR 2026, arXiv:2602.22555) for EEG2X-Benchmark.

The package is a near-verbatim copy of the released implementation
(https://github.com/ddicee/avde) with benchmark-side adapters:

``bootstrap.py``
    Puts the copied tree on ``sys.path`` so upstream's flat imports keep
    working, and records where the frozen weights live.
``eeg_data/dataset.py``
    Upstream's dataset contract, reading the benchmark's protocol-v2 HDF5 and
    frozen CLIP feature store instead of AVDE's own pickled arrays.
``inference.py`` / ``pipeline.py``
    Reconstruction and orchestration; everything written stays inside
    EEG2X-Benchmark-v2-test.

Deliberate deviations from upstream are listed in the module docstrings of the
files concerned and recorded in each run's ``run_config.json``.
"""
