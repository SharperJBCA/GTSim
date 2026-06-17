#!/usr/bin/env python
"""M3 map-maker validation: inverse-variance binning into a HEALPix map.

Run (inside the ``lbs`` env):
    micromamba run -n lbs python scripts/mapmaker_test.py

Checks:
  * Noiseless null test -- sampling the map at nearest pixel and binning back
    recovers the input exactly (to machine precision) on observed pixels.
  * Realistic recovery -- sampling the (interpolated) Haslam map and binning
    reproduces the degraded input map; residuals are small.
  * Inverse-variance noise propagation -- a white-noise-only map has per-pixel
    scatter matching sigma_map = sigma / sqrt(N_hits).
Writes recovered/hit maps and a residual histogram to ``outputs/``.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtsim import MapMaker, NoiseModel, ScanStrategy, SkyModel, load_config  # noqa: E402
from gtsim.mapmaker import bin_tod  # noqa: E402
from gtsim.sky import apparent_eq_to_galactic  # noqa: E402
from astropy.time import Time  # noqa: E402

CONFIG = ROOT / "configs" / "mark1.toml"
HASLAM = ROOT / "ancillary_data" / "haslam408_ds_Remazeilles2014.fits"
OUTDIR = ROOT / "outputs"

NSIDE = 64               # ~0.92 deg pixels (matches the ~1 deg survey beam)
N_NIGHTS = 20            # slice length for coverage


def main() -> int:
    OUTDIR.mkdir(exist_ok=True)
    cfg = load_config(CONFIG)
    sky = SkyModel.from_fits(HASLAM)
    noise = NoiseModel.from_config(cfg)
    mm = MapMaker(NSIDE)
    rng = np.random.default_rng(2)

    strat = ScanStrategy(cfg)
    pt = strat.generate(end=cfg.window.start + timedelta(days=N_NIGHTS))

    # Coordinate transform once; reuse l,b for sampling and pixelisation.
    l, b = apparent_eq_to_galactic(pt.ra_deg, pt.dec_deg,
                                   Time(pt.time_mjd, format="mjd"))
    pix = hp.ang2pix(NSIDE, l, b, lonlat=True)
    print(f"slice       : {N_NIGHTS} nights, {pt.n_samples} samples, nside={NSIDE}")

    # --- Noiseless null test (nearest-pixel sampling) -------------------
    input_map = hp.ud_grade(sky.map, NSIDE)            # truth at output nside
    d_exact = input_map[pix]
    rec = mm.make(pt, d_exact, sigma=noise.white_sigma_k, pix=pix)
    seen = rec.seen
    null_resid = np.abs(rec.signal[seen] - input_map[seen])
    print(f"sky coverage: {seen.sum()} / {seen.size} pixels "
          f"({100*seen.sum()/seen.size:.0f}%)")
    print(f"null test   : max |recovered - input| = {null_resid.max():.2e} K")
    ok = null_resid.max() < 1e-9

    # --- Realistic recovery (interpolated sampling) --------------------
    d_interp = hp.get_interp_val(sky.map, l, b, lonlat=True)
    rec_real = mm.make(pt, d_interp, sigma=noise.white_sigma_k, pix=pix)
    resid = rec_real.signal[seen] - input_map[seen]
    print(f"recovery    : residual median {np.median(resid):+.3f} K, "
          f"std {resid.std():.3f} K (interp vs nearest binning)")

    # --- Inverse-variance white-noise propagation ----------------------
    white = rng.normal(0.0, noise.white_sigma_k, pt.n_samples)
    nmap = mm.make(pt, white, sigma=noise.white_sigma_k,
                   exclude_cal=False, pix=pix)
    z = nmap.signal[nmap.seen] / nmap.noise_map[nmap.seen]   # ~ N(0,1)
    print(f"IVW noise   : std(map_noise / [sigma/sqrt(N)]) = {z.std():.3f} "
          "(expect 1.0)")
    ok &= 0.9 < z.std() < 1.1

    _plot_maps(rec_real, input_map, OUTDIR)
    print(f"plots       : {OUTDIR}/map_recovered.png, {OUTDIR}/map_hits.png, "
          f"{OUTDIR}/map_residual_hist.png")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _plot_maps(rec, input_map, outdir: Path) -> None:
    masked = rec.signal.copy()
    hp.mollview(np.where(rec.seen, masked, hp.UNSEEN), coord="G",
                title=f"Recovered map (nside={rec.nside})", unit="T [K]",
                norm="log", min=10, max=200, cmap="inferno")
    hp.graticule()
    plt.savefig(outdir / "map_recovered.png", dpi=120)
    plt.close()

    hp.mollview(np.where(rec.seen, rec.hits, hp.UNSEEN), coord="G",
                title="Hit count", unit="hits/pixel", cmap="viridis")
    hp.graticule()
    plt.savefig(outdir / "map_hits.png", dpi=120)
    plt.close()

    resid = rec.signal[rec.seen] - input_map[rec.seen]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(resid, -5, 5), bins=120, color="C0")
    ax.set(xlabel="recovered - input [K]", ylabel="pixels",
           title="Recovery residual (interp sampling vs degraded input)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "map_residual_hist.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
