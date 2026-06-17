# Ground Telescope Simulator

Simulator for a ground-based single-dish radio telescope, reproducing the scan
strategy and time-ordered data (TOD) of the 408 MHz Haslam surveys. It generates
pointing, samples a model sky, adds white + 1/f noise, and bins the result into a
HEALPix map.

## Setup

No install is required: the scripts add `src/` to the path. 

## Quick start

```bash
python scripts/run_pipeline.py \
    --config configs/mark1.toml \
    --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
    --nights 10 --outdir outputs/run --plots
```

Writes `sky_map.fits`, `hits.fits`, `noise_map.fits` (+ PNGs) to `--outdir`.
Run `... python -m gtsim run --help` (or read the script) for all options.

## Layout

```
src/gtsim/      library: config, site, scan, sky, noise, mapmaker, pipeline, cli
configs/        survey TOML configs (mark1.toml = reference; TEMPLATE.toml to copy)
scripts/        run_pipeline.py (full pipeline) + per-stage *_test.py validators
tests/          pytest unit tests  (run: micromamba run -n lbs python -m pytest -q)
ancillary_data/ Haslam map + digitised Fig. 2 TOD
docs/           PLAN.md (design + roadmap), CONFIGURING_SURVEYS.md (handoff guide)
```

## Pipeline stages

`scan` (pointing) > `sky` (signal TOD from the Haslam map) > `noise` (white +
1/f) > `mapmaker` (inverse-variance-binned HEALPix map). Each stage has a
`scripts/*_test.py` that validates it and emits diagnostic plots.

## Adding another survey

The pipeline is config-driven. To add Effelsberg / Parkes / the later JBO epoch,
copy `configs/TEMPLATE.toml` and edit it — see **`docs/CONFIGURING_SURVEYS.md`**
for the field-by-field reference, run/validate instructions, and the known
limitations that need code rather than config.
