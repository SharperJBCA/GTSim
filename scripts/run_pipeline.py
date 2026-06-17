#!/usr/bin/env python
"""Run the full simulation pipeline (no install needed).

Examples (inside the ``lbs`` env):
    # quick 10-night run, signal + noise, NSIDE=64
    micromamba run -n lbs python scripts/run_pipeline.py \
        --config configs/mark1.toml \
        --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
        --nights 10 --outdir outputs/run

    # full survey, signal only
    micromamba run -n lbs python scripts/run_pipeline.py \
        --config configs/mark1.toml \
        --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
        --no-noise --outdir outputs/full

This is a thin wrapper around ``gtsim.cli`` (equivalently: PYTHONPATH=src
python -m gtsim run ...).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gtsim.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
