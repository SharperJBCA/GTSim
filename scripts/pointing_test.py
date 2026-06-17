#!/usr/bin/env python
"""M1 pointing-only validation for the meridian elevation-scan strategy.

Run (inside the ``lbs`` env):
    micromamba run -n lbs python scripts/pointing_test.py

Scan model: continuous back-and-forth meridian sweeping within each night, with
the sweep phase advancing 80 sidereal seconds per night. Checks:
  * Dec coverage matches the configured -22..+62 deg range.
  * The sweep is continuous within a night (no declination jumps).
  * Slew rate ~ 3 deg / sidereal minute.
  * Night-to-night RA comb spacing ~ 0.333 deg (80 sidereal seconds).
  * Night gating: every sample has the Sun below astronomical twilight (-18).
  * Analytic meridian RA/Dec agrees with astropy's full AltAz->apparent transform.
Writes a single-night track plot and a hit map to ``outputs/``.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtsim import ScanStrategy, load_config  # noqa: E402

CONFIG = ROOT / "configs" / "mark1.toml"
OUTDIR = ROOT / "outputs"
SIDEREAL_PER_SOLAR = 1.00273790935
EXPECTED_RA_COMB_DEG = 80.0 * 15.0 / 3600.0  # 80 sidereal s -> deg
REF_DEC = 30.0


def _night_slices(pt):
    """Yield (night_id, index_array) for each night, in time order."""
    for nid in np.unique(pt.night_id):
        yield nid, np.flatnonzero(pt.night_id == nid)


def main() -> int:
    OUTDIR.mkdir(exist_ok=True)
    cfg = load_config(CONFIG)
    strat = ScanStrategy(cfg)

    slice_end = cfg.window.start + timedelta(days=8)
    pt = strat.generate(end=slice_end)

    print(f"survey      : {cfg.name}")
    print(f"slice       : {cfg.window.start} .. {slice_end}")
    print(f"nights/samps: {pt.n_nights} nights, {pt.n_samples} samples")
    if pt.n_samples == 0:
        print("ERROR: no samples generated")
        return 1

    ok = True
    dt = cfg.scan.sample_dt_s

    # --- Dec coverage --------------------------------------------------
    dmin, dmax = pt.dec_deg.min(), pt.dec_deg.max()
    print(f"dec range   : {dmin:+.2f} .. {dmax:+.2f} deg "
          f"(config {cfg.scan.dec_min_deg:+.0f} .. {cfg.scan.dec_max_deg:+.0f})")
    ok &= np.isclose(dmin, cfg.scan.dec_min_deg, atol=0.5)
    ok &= np.isclose(dmax, cfg.scan.dec_max_deg, atol=0.5)

    print(f"RA range    : {pt.ra_deg.min()/15:.2f} .. {pt.ra_deg.max()/15:.2f} h")
    print(f"el range    : {pt.el_deg.min():.2f} .. {pt.el_deg.max():.2f} deg")
    print(f"az values   : {np.unique(np.round(pt.az_deg)).tolist()} (0=N, 180=S)")

    # --- Continuity + slew rate within each night ----------------------
    # Slew rate is measured on ramp samples only (exclude the calibration dwell,
    # where dDec=0 by design).
    max_jump = 0.0
    ramp_ddec = []
    for _, idx in _night_slices(pt):
        order = np.argsort(pt.time_mjd[idx])
        ddec = np.abs(np.diff(pt.dec_deg[idx][order]))
        if not ddec.size:
            continue
        max_jump = max(max_jump, ddec.max())
        cal = pt.cal_mask[idx][order]
        on_ramp = ~(cal[:-1] | cal[1:])
        ramp_ddec.append(ddec[on_ramp])
    ramp_ddec = np.concatenate(ramp_ddec)
    slew = np.median(ramp_ddec) / (dt * SIDEREAL_PER_SOLAR / 60.0)  # deg/sid-min
    print(f"continuity  : max within-night |dDec| = {max_jump:.4f} deg/sample")
    print(f"slew rate   : {slew:.3f} deg/sid-min "
          f"(config {cfg.scan.rate_deg_per_sid_min})")
    ok &= max_jump < 0.1
    ok &= np.isclose(slew, cfg.scan.rate_deg_per_sid_min, atol=0.05)

    # --- Calibration dwell ---------------------------------------------
    cal = pt.cal_mask
    if cfg.scan.cal_pause_s > 0:
        runs = []
        for _, idx in _night_slices(pt):
            c = pt.cal_mask[idx][np.argsort(pt.time_mjd[idx])].astype(np.int8)
            edges = np.diff(np.concatenate([[0], c, [0]]))
            runs.extend((np.flatnonzero(edges == -1)
                         - np.flatnonzero(edges == 1)).tolist())
        runs = np.array(runs)
        cal_dec = pt.dec_deg[cal]
        print(f"cal dwell   : {cal.sum()} samples, mean Dec {cal_dec.mean():+.2f} "
              f"deg, median run {np.median(runs)*dt:.0f} s "
              f"(config {cfg.scan.cal_pause_s:.0f} s @ dec_{cfg.scan.cal_at})")
        ok &= cal.any()
        ok &= np.isclose(cal_dec.mean(), cfg.scan.dec_max_deg, atol=0.5)
        ok &= np.isclose(np.median(runs) * dt, cfg.scan.cal_pause_s, atol=5.0)
    else:
        print("cal dwell   : disabled")

    # --- Night-to-night RA comb at a reference declination -------------
    # One RA per rising crossing of REF_DEC (sign change of dec-REF_DEC). Within
    # a night these crossings are ~14 deg apart (one sweep period); the phase
    # offset shifts them 0.333 deg/night, which shows up as the sub-1-deg gaps.
    cross_ra = []
    for _, idx in _night_slices(pt):
        order = np.argsort(pt.time_mjd[idx])
        dec = pt.dec_deg[idx][order] - REF_DEC
        ra = pt.ra_deg[idx][order]
        up = np.flatnonzero((dec[:-1] < 0) & (dec[1:] >= 0))
        cross_ra.extend(ra[up + 1].tolist())
    cross_ra = np.sort(np.array(cross_ra))
    gaps = np.diff(cross_ra)
    fine = gaps[(gaps > 1e-3) & (gaps < 1.0)]
    comb = np.median(fine) if fine.size else float("nan")
    print(f"RA comb     : median {comb:.4f} deg at Dec={REF_DEC:.0f} "
          f"(expected {EXPECTED_RA_COMB_DEG:.4f}, n={fine.size})")
    ok &= np.isclose(comb, EXPECTED_RA_COMB_DEG, atol=0.02)

    # --- Night gating (random subsample) -------------------------------
    rng = np.random.default_rng(0)
    sub = rng.choice(pt.n_samples, size=min(500, pt.n_samples), replace=False)
    sun = strat.site.sun_alt_deg(Time(pt.time_mjd[sub], format="mjd"))
    print(f"sun alt     : max {sun.max():.2f} deg "
          f"(threshold {cfg.window.night_sun_alt_deg:.0f})")
    ok &= sun.max() < cfg.window.night_sun_alt_deg

    # --- Analytic vs astropy cross-check (apparent/TETE) ---------------
    idx = np.linspace(0, pt.n_samples - 1, min(200, pt.n_samples)).astype(int)
    t = Time(pt.time_mjd[idx], format="mjd")
    ra_ap, dec_ap = strat.site.altaz_to_radec(t, pt.az_deg[idx], pt.el_deg[idx])
    dra = np.abs(((pt.ra_deg[idx] - ra_ap + 180) % 360) - 180)
    ddec = np.abs(pt.dec_deg[idx] - dec_ap)
    print(f"astropy xchk: max dRA {dra.max()*3600:.2f}\", "
          f"max dDec {ddec.max()*3600:.2f}\" (vs apparent/TETE)")
    ok &= dra.max() < 0.01 and ddec.max() < 0.01

    # --- Plots ---------------------------------------------------------
    _plot_single_night(pt, OUTDIR / "single_night_track.png")
    _plot_hitmap(pt, OUTDIR / "hitmap_radec.png")
    print(f"plots       : {OUTDIR}/single_night_track.png, "
          f"{OUTDIR}/hitmap_radec.png")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _plot_single_night(pt, path: Path) -> None:
    nid, idx = next(_night_slices(pt))
    order = np.argsort(pt.time_mjd[idx])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pt.ra_deg[idx][order] / 15.0, pt.dec_deg[idx][order], "-", lw=0.6)
    ax.set(xlabel="RA [h]", ylabel="Dec [deg]",
           title=f"Continuous meridian sweep, one night (id={nid})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_hitmap(pt, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    h = ax.hist2d(pt.ra_deg / 15.0, pt.dec_deg, bins=[240, 168],
                  range=[[0, 12], [-25, 65]], cmin=1)
    fig.colorbar(h[3], ax=ax, label="hits / bin")
    ax.set(xlabel="RA [h]", ylabel="Dec [deg]", title="Pointing hit map (slice)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
