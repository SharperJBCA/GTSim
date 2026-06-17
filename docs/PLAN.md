# Ground-Based Single-Dish Telescope Simulator — Design & Plan

## 1. Goal

Build a simulator for a ground-based single-dish radio telescope that can
reproduce the **scan strategy and time-ordered data (TOD)** of historical
408 MHz surveys, with the **Haslam et al. all-sky 408 MHz map** as the primary
science target.

The Haslam map (1° resolution, all-sky) is a mosaic of four surveys taken with
different instruments and observing modes:

| Telescope / Site            | Dish      | Hemisphere coverage         |
|-----------------------------|-----------|-----------------------------|
| Jodrell Bank Mark 1 (Lovell)| 76 m      | Northern sky                |
| Effelsberg (MPIfR)          | 100 m     | Mid/northern declinations   |
| Parkes                      | 64 m      | Southern sky                |
| (Jodrell Bank, later epoch) | 76 m      | fill-in / cross-calibration |

Each contributing survey has its **own** beam, scan rate, integration time,
sensitivity, observing window, and sky coverage. The simulator must therefore
treat a "survey" as a configurable object, not bake in one set of numbers.

### Success criteria
- Given a survey config, produce a realistic TOD: signal + 1/f noise + white
  noise (+ optional ground/atmosphere).
- Reproduce the **meridian elevation-scanning** geometry: the telescope sweeps
  in elevation along the local meridian while the sky drifts through in RA.
- Bin the TOD back into a HEALPix map and recover the input sky (the round-trip
  / null test).
- Configurations for at least the Mark 1 survey, parameterized so Effelsberg
  and Parkes can be added later.

## 2. Reference parameters (known so far)

### Jodrell Bank Mark 1 (Lovell) survey
- **Reference:** Haslam 1970.
- **Sensitivity:** ΔT = 0.2 K in 1 s of integration time
  (white-noise level → NET ≈ 0.2 K·√s). They quote the system noise temperature as 180 K.
- **Scan rate:** 3° per **sidereal** minute, in elevation, along the meridian.
- **Scan cadence:** each scan starts **80 sidereal seconds later** than the
  previous one (this sets the RA spacing between successive meridian scans).
- **Observing window:** 1965-11-22 to 1966-01-27, **night-time only**.
- **RA coverage:** 00ʰ–12ʰ in right ascension.
- **Dec coverate:** -22 to +62 degrees.
- **Known gap:** reduced data between 00ʰ–04ʰ RA at **low declinations**
  (cause TBD — weather and/or instrument; treat as a coverage/quality mask).
- **Zero Level:** On 1965 December 25 and 26 no scans taken as these were used for 
  zenith scans for zero level calibration. 

### Derived / to-confirm quantities
- RA drift per scan: 80 sidereal s × (15°/3600 s) = **0.333° RA** between scans.
- Elevation sweep range and declination coverage: derived from JBO latitude
  (≈ +53.24° N) and the scan elevation limits — **need the actual el range**.
- Beam FWHM: Mark 1 at 408 MHz ≈ 0.85° (consistent with the 1° final map).
  **Confirm per telescope.**
- "Night-time only" → need a Sun-avoidance / local-time gate per date.

> **Open questions to resolve before/while coding** (collect citations in
> `docs/REFERENCES.md`):
> 1. Exact elevation scan limits and whether scans go up, down, or both.
> 2. Beam FWHM and shape per telescope at 408 MHz.
> 3. Definition of "night" used (astronomical twilight? fixed local hours?).
> 4. Sampling/integration time per sample vs. the quoted 1 s sensitivity.
> 5. Effelsberg & Parkes scan parameters (analogous table).
> 6. Calibration scale (the original used the full-beam brightness temperature
>    scale; relevant if we compare absolute levels).

## 3. Architecture

Pipeline of composable stages, each independently testable:

```
SurveyConfig ──▶ ScanStrategy ──▶ Pointing (t → az/el → RA/Dec)
                                        │
                 SkyModel ──────────────┤
                 NoiseModel (1/f + white)┤──▶ TOD generator ──▶ TOD
                 Ground/AtmosModel ──────┘                        │
                                                                  ▼
                                                          Map-maker (bin / destripe)
                                                                  │
                                                                  ▼
                                                         Output map + hit/coverage map
```

### Core data structures
- **TOD**: per-sample arrays — `time` (MJD/sidereal), `az`, `el`, `ra`, `dec`,
  `signal`, plus a `flags`/`mask` array. Stored as an xarray Dataset or a
  lightweight dataclass wrapping numpy arrays; consider HDF5 for persistence.
- **Sky map**: HEALPix (`healpy`) in brightness temperature [K], with a defined
  reference frequency (408 MHz) and beam smoothing.

### Modules (proposed `src/gtsim/` layout)
- `config.py` — `SurveyConfig` dataclass + TOML loader (one TOML per survey).
- `site.py` — observatory location, sidereal-time and az/el↔RA/Dec transforms
  (use `astropy.coordinates` / `astropy.time`; verify against a known case).
- `scan.py` — `ScanStrategy`: generates the sample timeline and pointing from a
  config (meridian elevation sweeps, scan cadence, night/Sun gating, masks).
- `sky.py` — `SkyModel`: load/generate an input 408 MHz sky, beam-convolve,
  sample at pointing. Start from the published Haslam map itself as truth, and
  also support synthetic skies (point sources + diffuse) for null tests.
- `noise.py` — white + 1/f noise generation (see §5).
- `ground.py` — optional ground/atmosphere slab emission model (see §6).
- `tod.py` — assemble signal + noise + systematics into a TOD.
- `mapmaker.py` — naive binning first; destriping later.
- `io.py` — TOD/map read/write.
- `cli.py` — `gtsim run <config.toml>` entry point.

## 4. Scan strategy (the heart of it)

Meridian elevation scanning: the telescope sits on the local meridian (azimuth
due south below the site latitude, due north above it) and sweeps **continuously
back and forth in elevation/declination** at a fixed rate. The sky's RA drifts
past the meridian via Earth rotation, so RA = local sidereal time at every
sample. (Interpretation **b**, confirmed: within a night the dish sweeps
continuously; the *next* night's pattern starts 80 sidereal seconds later,
shifting the RA sampling comb by 0.333°/night so the grid fills over the survey.)

Because the slew rate is per *sidereal* minute and RA = LST, declination is a
triangle wave in unwrapped local sidereal time:

    pattern_arg(t) = (LST_sid_s(t) − n·night_offset) mod (2·up_sweep_sid_s)

with `n` the calendar-night index from survey start.

Implementation steps (see `src/gtsim/scan.py`):
1. For each observing date, find the dark window where the Sun is below
   astronomical twilight (−18°), on a coarse grid (assumed one block/night).
2. Sample that window at the integration time; declination follows the triangle
   wave above (`dec_min`↔`dec_max` at 3°/sidereal-min); RA = LST.
3. Dec sets elevation (`el = 90 − |φ − dec|`) and azimuth (S/N). Coordinates are
   *apparent* (true equator/equinox of date); J2000/Galactic conversion is an
   M2 step.
4. At the upper turn (`dec_max`) the dish dwells ~100 s to fire the gain-
   calibration diode (Haslam Fig. 2): a flat top in the Dec triangle, with the
   dwell samples flagged in `Pointing.cal_mask` (`cal_pause_s`, `cal_at`).
5. Apply the RA-coverage cut (00ʰ–12ʰ). The low-dec/00ʰ–04ʰ quality mask is
   deferred (determined later).
6. Skip the 1965-12-25/26 zero-level calibration nights (phase clock still
   advances).

**Validation** (`scripts/pointing_test.py`): Dec coverage −22..+62, continuity
of the sweep within a night, slew rate 3°/sid-min, the 0.333° night-to-night RA
comb, twilight gating, and analytic-vs-astropy agreement.

## 5. Noise model

- **White noise:** σ per sample from the NET and per-sample integration time:
  σ = NET / √(t_sample). Anchor on ΔT = 0.2 K @ 1 s → NET ≈ 0.2 K·√s for Mark 1. Double check, they had 
  a Dicke receiver with a 180K system temperature but paper is unclear of the bandwidth. 
- **1/f noise:** generate correlated noise with PSD
  P(f) = σ²·(1 + (f_knee/f)^α), parameterized by knee frequency `f_knee` and
  slope `α`. Generate in the Fourier domain per scan/segment. Knee and slope
  are per-survey config (start with plausible values; flag as to-tune).
- 1/f is what motivates the scan strategy and destriping — the fast elevation
  sweep crosses many declinations before the drift moves in RA, so common-mode
  drifts can be separated from sky structure.

## 6. Ground / atmosphere (optional, later phase)

- **Ground slab:** an az/el-dependent additive term (pickup strongest at low
  elevation). Model as a smooth function of elevation, T_ground(el), fixed in
  the horizon frame → appears as a stable offset per elevation, distinct from
  the drifting sky. Good for testing the map-maker's ability to reject
  ground-fixed signal.
- **Atmosphere:** airmass-dependent emission ∝ 1/sin(el) with slow temporal
  fluctuations; largely negligible at 408 MHz but included for completeness.
- Keep these **toggleable** and off by default for the first round-trip test.

## 6b. Running the pipeline

`gtsim.pipeline.run_pipeline` chains scan → sky → noise → map and writes HEALPix
FITS products (`sky_map.fits`, `hits.fits`, `noise_map.fits`; `--save-tod` adds
`tod.npz`; `--plots` adds Mollweide PNGs + a one-night TOD plot). Run it via the
CLI:

```bash
# quick 10-night run, signal + noise
micromamba run -n lbs python scripts/run_pipeline.py \
    --config configs/mark1.toml \
    --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
    --nights 10 --outdir outputs/run

# full survey, signal only
micromamba run -n lbs python scripts/run_pipeline.py \
    --config configs/mark1.toml \
    --sky ancillary_data/haslam408_ds_Remazeilles2014.fits \
    --no-noise --outdir outputs/full
```

Equivalent packaged form: `PYTHONPATH=src micromamba run -n lbs python -m gtsim
run ...`. Options: `--nside`, `--start/--end` (or `--nights`), `--seed`,
`--no-noise`, `--save-tod`, `--plots`.

## 7. Map-making

1. **Phase 1 — naive binning:** accumulate samples into HEALPix pixels,
   weight by inverse white-noise variance, divide by hit count. Produces a map
   plus a hit/coverage map. Sufficient to validate geometry and signal path.
2. **Phase 2 — destriping:** fit per-scan offset/baseline templates to remove
   1/f stripes (à la Madam/Descart). Needed to handle correlated noise and to
   demonstrate recovery of the diffuse sky. We do not need this as we will ultimately
   performing the cleaning of artifacts in the sky frame like the Haslam papers.

## 8. Implementation roadmap

- [x] **M0 — scaffolding:** repo layout (`src/gtsim/`), `lbs` env confirmed,
      deps in `pyproject.toml` (numpy, scipy, astropy, healpy, matplotlib).
- [x] **M1 — pointing & scan:** `config.py` + `site.py` + `scan.py`; produces a
      pointing timeline for the Mark 1 config using interpretation **b**
      (continuous nightly sweeping, 80-sid-s phase offset per night). Validated
      by `scripts/pointing_test.py` (Dec −22..+62, within-night continuity, slew
      3°/sid-min, 0.333° RA comb, −18° twilight gating, analytic-vs-astropy 0.3″)
      and `tests/test_geometry.py`. Pointing is in *apparent* (TETE) coords;
      J2000/Galactic conversion is deferred to M2 sky sampling.
- [x] **M2 — sky signal (sky model):** `sky.py` `SkyModel` loads the Haslam map
      (`ancillary_data/haslam408_ds_Remazeilles2014.fits`, NSIDE=512, Galactic,
      K), with optional Gaussian beam smoothing, and samples along the pointing
      via an apparent→Galactic transform. Validated by `scripts/sky_model_test.py`
      (sensible TOD with plane-crossing humps; scan footprint lands on the sky;
      morphology matches the digitised Fig. 2 trace). Notes: the map is already
      at ~1° beam so it is sampled directly (no extra beam); it is the *desourced*
      (`_ds`) map, so point sources like 3C 86 are absent and need a separate
      catalog if required. Remaining M2 work: synthetic test skies; flux/abs
      calibration for the quantitative Fig. 2 comparison; **TODO: match the model
      trace to the 3C 86 spike in the digitised trace** (inject a point-source
      catalog; the `_ds` map has 3C 86 removed).
- [x] **M3 — round trip:** `mapmaker.py` `MapMaker`/`bin_tod` do inverse-variance
      binning (`m_p = Σ w_i d_i / Σ w_i`, `w_i = 1/σ_i²`) into a Galactic HEALPix
      map, with per-pixel weight/hit maps and `noise_map = 1/√W`. Validated by
      `scripts/mapmaker_test.py` + `tests/test_mapmaker.py`: noiseless null test
      recovers the input to ~7e-15 K, and a white-noise map matches σ/√N_hits
      (`std(map_noise/[σ/√N]) = 0.995`). Cal-dwell samples excluded by default.
- [~] **M4 — noise (model done):** `noise.py` `NoiseModel` generates white +
      1/f TOD with PSD `2*NET^2*(1+(f/f_knee)^alpha)` (Mark 1: NET=0.2 K·s^0.5,
      f_knee=0.5 Hz, alpha=−1, an assumed *instrumental* 1/f). Validated by
      `scripts/noise_test.py` (Fourier-synthesised PSD matches the model: white
      0.08 K²/Hz, knee 0.5 Hz, slope ≈−1) plus a signal+noise demo. Note: at the
      1 s survey cadence the knee sits at Nyquist, so the in-band noise is
      essentially all 1/f. White-noise map propagation verified in M3
      (σ_map = σ/√N_hits). Remaining: characterise the *1/f* contribution to
      recovered-map noise (striping) vs. theory.
- [ ] **M5 — systematics:** ground/atmosphere slab; show map-maker response.
- [ ] **M6 — destriping:** recover diffuse sky in presence of 1/f.
- [ ] **M7 — multi-survey:** Effelsberg & Parkes configs; combine into a mosaic
      and compare to the published Haslam map.

## 9. Configuration sketch (`configs/mark1.toml`)

```toml
name: jbo_mark1_1965
site:
  name: Jodrell Bank
  lat_deg: 53.2367
  lon_deg: -2.3085
  height_m: 78
instrument:
  freq_mhz: 408
  beam_fwhm_deg: 0.85          # TODO confirm
  net_k_sqrt_s: 0.2            # 0.2 K in 1 s
  f_knee_hz: null              # TODO tune
  alpha: 1.0                   # TODO tune
scan:
  mode: meridian_elevation
  rate_deg_per_sid_min: 3.0
  night_start_offset_sid_s: 80    # sweep-phase advance per night -> 0.333 deg RA comb
  dec_min_deg: -22
  dec_max_deg: 62
  azimuth: auto                   # due S below latitude, due N above
  sample_dt_s: 1.0
window:
  start: 1965-11-22
  end:   1966-01-27
  night_only: true
  night_sun_alt_deg: -18          # astronomical twilight
  exclude_dates: [1965-12-25, 1965-12-26]   # zero-level calibration
coverage:
  ra_hours: [0.0, 12.0]
  masks:
    - { ra_hours: [0.0, 4.0], dec_below_deg: <TBD>, reason: "reduced low-dec data" }
systematics:
  ground: false
  atmosphere: false
```

## 10. Testing & validation strategy

- **Unit:** coordinate transforms against `astropy` reference cases; sidereal
  time; noise PSD recovery (generate → estimate PSD → compare to model).
- **Integration:** noiseless round-trip null test (M3); noise-only map matches
  predicted per-pixel σ given hit counts.
- **Science:** compare a simulated single-survey map to the corresponding
  region of the real Haslam map (morphology + level after calibration).

---
*References and source citations live in `docs/REFERENCES.md` (to be created as
parameters are confirmed). All numbers marked TODO/TBD must be sourced before
they are trusted in published results.*
