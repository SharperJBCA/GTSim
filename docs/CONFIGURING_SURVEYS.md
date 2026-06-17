# Configuring a new Haslam survey

This guide is for adding the other 408 MHz surveys (Effelsberg, Parkes, the later
Jodrell Bank epoch) to the simulator. The pipeline is **config-driven**: for a
survey that scans the same way as the Jodrell Bank Mark 1 reference
(`configs/mark1.toml`), you only need to write a new TOML file — no code changes.

Read this alongside `docs/PLAN.md` (the design) and `configs/mark1.toml` (a
worked example).

---

## 1. What a config controls

Each TOML maps onto the pipeline stages (`scan > sky > noise > map`):

| Stage | Module | Config it reads |
|-------|--------|-----------------|
| Pointing | `scan.py` | `[site]`, `[scan]`, `[window]`, `[coverage]` |
| Sky signal | `sky.py` | (none — the sky map is a CLI argument, e.g. the real 408MHz map) |
| Noise | `noise.py` | `[instrument]` (`net_k_sqrt_s`, `f_knee_hz`, `alpha`), `scan.sample_dt_s` |
| Map | `mapmaker.py` | `--nside` (CLI), uses the noise σ for weights |

The same input sky map (`ancillary_data/haslam408_ds_Remazeilles2014.fits`) is
used for every survey; it is passed on the command line, not in the config.

We may use a different map in the future as real Haslam map has already been filtered along the
meridians, meaning the impact of these simulations on the map will be dampened.

## 2. Config reference

Copy `configs/TEMPLATE.toml` and edit. Fields below are grouped by how much they
vary between surveys.

### Almost always different per survey
- `name` — short identifier.
- `[site] lat_deg, lon_deg, height_m` — the telescope location. Drives sidereal
  time, the dec↔elevation geometry, and Sun/night calculations.
- `[scan] dec_min_deg, dec_max_deg` — the declination range swept.
- `[scan] rate_deg_per_sid_min` — slew rate.
- `[instrument] net_k_sqrt_s` — sensitivity (white-noise level).
- `[window] start, end` — observing dates (`end` is exclusive).
- `[coverage] ra_hours` — RA range kept, `[lo, hi)` in hours.

### Often different
- `[scan] night_start_offset_sid_s` — per-night sweep-phase advance (sets the RA
  sampling comb; 80 sidereal s ⇒ 0.333°/night for Mark 1).
- `[scan] cal_pause_s`, `cal_at` — calibration-diode dwell at the turn(s). Set
  `cal_pause_s = 0` to disable.
- `[window] night_only`, `night_sun_alt_deg` — observe only at night (and how
  dark). `night_only = false` observes the full 24 h (RA cut then selects).
- `[window] exclude_dates` — non-observing dates (e.g. zero-level calibration).
- `[scan] azimuth` — `"auto"` flips S/N at the site latitude; force with
  `"south"`/`"north"` if a survey only used one meridian transit.
- `[instrument] f_knee_hz`, `alpha` — 1/f noise (assumed instrumental).

### Usually the same / informational
- `[scan] mode` — must be `"meridian_elevation"` (only mode implemented).
- `[scan] sample_dt_s` — integration time (1 s).
- `[instrument] freq_mhz`, `system_temp_k` — informational, not used in compute.
- `[instrument] beam_fwhm_deg` — used only if you explicitly smooth a synthetic
  sky via `SkyModel.smoothed()`. The Haslam map is already at ~1° beam, so the
  pipeline does **not** apply extra smoothing.
- `[systematics] ground, atmosphere` — placeholders, **not yet implemented**.

## 3. Recipe: add a survey

1. `cp configs/TEMPLATE.toml configs/effelsberg.toml`.
2. Fill in `name`, `[site]`, dec range, slew rate, NET, dates, RA coverage.
3. Put any uncertain numbers as `TODO` and record the source (paper/figure) in
   `docs/REFERENCES.md`.
4. Sanity-run a few nights with plots (section 4).
5. Validate the geometry (section 5).
6. Run the full survey when happy.

## 4. How to run

```bash
# quick check: 10 nights, signal + noise, with plots
python scripts/run_pipeline.py \
    --config configs/effelsberg.toml \
    --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
    --nights 10 --outdir outputs/effelsberg --plots

# full survey, signal only
python scripts/run_pipeline.py \
    --config configs/effelsberg.toml \
    --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
    --no-noise --outdir outputs/effelsberg_full --plots
```

Outputs in `--outdir`: `sky_map.fits`, `hits.fits`, `noise_map.fits` (+ `.png`
with `--plots`, + `tod.npz` with `--save-tod`). Options: `--nside` (default 64),
`--start/--end` or `--nights`, `--seed`, `--no-noise`, `--save-tod`, `--plots`.

## 5. How to validate a new config

The per-stage scripts default to `configs/mark1.toml`; point them at the new
config by editing the `CONFIG = ...` line near the top, or copy the script. They
print a `PASS/FAIL` and write plots to `outputs/`.

- `scripts/pointing_test.py` — dec coverage, sweep continuity, slew rate, the RA
  comb, night gating, and analytic-vs-astropy agreement. **Run this first** for a
  new site/scan: it confirms the geometry.
- `scripts/sky_model_test.py` — sky TOD + scan footprint on the map; sanity-check
  the footprint lands where the survey should observe.
- `scripts/noise_test.py` — noise PSD vs the model.
- `scripts/mapmaker_test.py` — null test + inverse-variance noise propagation.

Quick manual checks on a new config:
```bash
python -c "
import sys; sys.path.insert(0,'src')
from datetime import timedelta
from gtsim import load_config, ScanStrategy
cfg = load_config('configs/effelsberg.toml')
pt = ScanStrategy(cfg).generate(end=cfg.window.start + timedelta(days=3))
print('samples', pt.n_samples, 'dec', pt.dec_deg.min(), pt.dec_deg.max(),
      'RA[h]', pt.ra_deg.min()/15, pt.ra_deg.max()/15)
"
```

## 6. Known limitations / extension points

These need **code**, not just config — flag them if a survey requires them:

1. **Scan mode.** Only `meridian_elevation` (continuous dec triangle on the
   meridian) is implemented; any other `mode` raises `NotImplementedError`. If a
   survey scanned differently (e.g. constant-declination drift scans, or azimuth
   scans), add a branch in `ScanStrategy` (`_dec_cal_from_lst` + the pointing
   geometry in `site.py`).
2. **Coverage masks.** Only a single rectangular `ra_hours` range is supported.
   The Mark 1 reduced-data region (00–04 h at low dec) and any other
   RA/dec-dependent quality masks are not yet implemented (`coverage` would need
   a mask list + handling in `scan.generate`).
3. **Systematics.** `[systematics] ground/atmosphere` are placeholders; the
   ground/atmosphere slab model (M5) is not built yet.
4. **Beam.** The pipeline samples the (already beam-smoothed) Haslam map directly.
   For a survey with a very different beam you would smooth/convolve explicitly.
5. **Point sources.** The input map is de-sourced (`_ds`); sources such as 3C 86
   are absent (see the 3C 86 TODO in the plan).
6. **Calibration signal.** Cal-dwell samples are *flagged* (`Pointing.cal_mask`)
   and excluded from maps, but the 25 K diode signal is not injected into the TOD.

## 7. Where to record sourced numbers

Create/maintain `docs/REFERENCES.md` with the paper/figure each number comes from
(NET, beam, dec range, slew rate, dates, RA coverage, knee). Anything still a
guess should stay marked `TODO` in the config and the references file until
confirmed.
