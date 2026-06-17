#!/usr/bin/env python
"""M2 sky-model validation: sample the Haslam 408 MHz map along the pointing.

Run (inside the ``lbs`` env):
    micromamba run -n lbs python scripts/sky_model_test.py

Produces:
  * outputs/sky_tod_one_night.png   -- signal TOD (K) for one night vs RA/Dec,
    showing Galactic-plane crossings.
  * outputs/sky_footprint.png       -- Haslam map (Galactic) with the scan track
    overplotted, to confirm the apparent->Galactic transform lands on the sky.
  * outputs/sky_fig2_comparison.png -- model temperature along an (approximate)
    reconstruction of the Fig. 2 +40..+60 scan pair, overlaid on the digitised
    trace, as a morphology check that we recreate the observed features.
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
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtsim import ScanStrategy, SkyModel, apparent_eq_to_galactic, load_config  # noqa: E402

CONFIG = ROOT / "configs" / "mark1.toml"
HASLAM = ROOT / "ancillary_data" / "haslam408_ds_Remazeilles2014.fits"
DIGITISED = ROOT / "ancillary_data" / "haslam1970" / "haslam1970_tod_regridded_1s.txt"
OUTDIR = ROOT / "outputs"

SIDEREAL_PER_SOLAR = 1.00273790935
DEC_LO, DEC_HI = 38.0, 61.0     # Fig. 2 scan-pair dec range
CAL_PAUSE_S = 100.0             # diode dwell at the apex


def main() -> int:
    OUTDIR.mkdir(exist_ok=True)
    cfg = load_config(CONFIG)
    sky = SkyModel.from_fits(HASLAM)
    print(f"sky map     : nside={sky.nside}, {sky.coordsys}, unit={sky.unit}")
    print(f"map T range : {sky.map.min():.1f} .. {sky.map.max():.1f} K")

    # --- One night of pointing, sampled through the sky -----------------
    strat = ScanStrategy(cfg, site=None)
    pt = strat.generate(end=cfg.window.start + timedelta(days=1))
    signal = sky.sample_pointing(pt)
    print(f"TOD         : {signal.size} samples, "
          f"T {signal.min():.1f} .. {signal.max():.1f} K, "
          f"median {np.median(signal):.1f} K")

    _plot_tod(pt, signal, OUTDIR / "sky_tod_one_night.png")
    _plot_footprint(sky, pt, OUTDIR / "sky_footprint.png")
    _plot_fig2(sky, OUTDIR / "sky_fig2_comparison.png")
    print(f"plots       : {OUTDIR}/sky_tod_one_night.png, "
          f"{OUTDIR}/sky_footprint.png, {OUTDIR}/sky_fig2_comparison.png")
    return 0


def _plot_tod(pt, signal, path: Path) -> None:
    order = np.argsort(pt.time_mjd)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pt.ra_deg[order] / 15.0, signal[order], lw=0.5)
    ax.set(xlabel="RA [h]", ylabel="T [K]", yscale="log",
           title="Sky signal TOD, one night (Galactic-plane crossings = spikes)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_footprint(sky, pt, path: Path) -> None:
    sub = np.linspace(0, pt.n_samples - 1, min(3000, pt.n_samples)).astype(int)
    t = Time(pt.time_mjd[sub], format="mjd")
    l, b = apparent_eq_to_galactic(pt.ra_deg[sub], pt.dec_deg[sub], t)
    hp.mollview(np.log10(sky.map), coord="G", title="Haslam 408 MHz + scan track",
                unit="log10(T/K)", cmap="inferno")
    hp.projscatter(l, b, lonlat=True, s=1, c="cyan")
    hp.graticule()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_fig2(sky, path: Path) -> None:
    """Overlay model T along the Fig. 2 scan pair (Dec 38->61 with a calibration
    dwell at the apex) on the digitised TOD."""
    d = np.genfromtxt(DIGITISED, delimiter=",", comments="#")
    _, ra_h, bright = d.T

    # Locate the apex (the calibration box peak) and give it a 100 s dwell.
    ra_peak = ra_h[np.argmax(bright)]
    dwell_half = 0.5 * CAL_PAUSE_S * SIDEREAL_PER_SOLAR / 3600.0   # hours
    ra_hi, ra_lo = ra_peak + dwell_half, ra_peak - dwell_half
    ra_max, ra_min = ra_h.max(), ra_h.min()

    # Dec: rise 38->61 (RA: ra_max -> ra_hi), dwell at 61, fall 61->38.
    up = ra_h >= ra_hi
    box = (ra_h < ra_hi) & (ra_h > ra_lo)
    down = ra_h <= ra_lo
    dec = np.empty_like(ra_h)
    dec[up] = DEC_LO + (DEC_HI - DEC_LO) * (ra_max - ra_h[up]) / max(ra_max - ra_hi, 1e-6)
    dec[box] = DEC_HI
    dec[down] = DEC_HI - (DEC_HI - DEC_LO) * (ra_lo - ra_h[down]) / max(ra_lo - ra_min, 1e-6)

    epoch = Time(1966.1, format="decimalyear")           # caption: epoch 1966.1
    model = sky.sample_equatorial_apparent(ra_h * 15.0, dec, epoch)

    # TODO: match the model trace to the 3C 86 spike seen in the digitised
    # trace (RA ~ 3.51 h). The Remazeilles map is de-sourced so 3C 86 is absent;
    # to reproduce it we must inject a point-source catalog (3C 86 = 0316+413)
    # and pin down the exact apex RA / ramp rate of this record.

    # Compare only sky samples: drop the calibration box and extreme values.
    lo, hi = np.percentile(bright, [2, 99])
    mask = ~box & (bright > lo) & (bright < hi)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ra_h[~box], model[~box], "C3-", lw=0.8,
            label="Haslam model along track (K)")
    ax.axvspan(ra_lo, ra_hi, color="0.85", label="calibration dwell (61 deg)")
    ax.set(xlabel="RA [h]", ylabel="model T [K]",
           title="Fig. 2 morphology check (Dec 38..61 + 100 s cal dwell)")
    ax.invert_xaxis()
    axr = ax.twinx()
    axr.plot(ra_h[mask], bright[mask], "C0.", ms=2, alpha=0.5,
             label="digitised (arb., cal/source masked)")
    axr.set_ylabel("digitised brightness [arb.]")
    ax.legend(loc="upper left", fontsize=8)
    axr.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
