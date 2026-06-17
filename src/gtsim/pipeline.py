"""End-to-end simulation pipeline: scan -> sky -> noise -> map.

Chains the stages into one call:
  1. ScanStrategy   -> pointing (apparent equatorial, of date)
  2. SkyModel       -> signal TOD [K] (sample the input map along the pointing)
  3. NoiseModel     -> white + 1/f noise TOD [K]  (optional)
  4. MapMaker       -> inverse-variance-binned HEALPix map [K] + hit/noise maps

The apparent->Galactic transform is done once and reused for both sky sampling
and pixelisation. Map products are written as HEALPix FITS files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from .config import load_config
from .mapmaker import BinnedMap, MapMaker
from .noise import NoiseModel
from .scan import Pointing, ScanStrategy
from .sky import SkyModel, apparent_eq_to_galactic


@dataclass
class PipelineProducts:
    pointing: Pointing
    signal: np.ndarray
    noise: np.ndarray
    binned: BinnedMap


def run_pipeline(
    config_path,
    sky_path,
    out_dir,
    start=None,
    end=None,
    nside: int = 64,
    add_noise: bool = True,
    seed: int = 0,
    save_tod: bool = False,
    write_maps: bool = True,
    make_plots: bool = False,
) -> PipelineProducts:
    cfg = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. pointing
    pointing = ScanStrategy(cfg).generate(start=start, end=end)
    if pointing.n_samples == 0:
        raise RuntimeError("no samples generated for the requested window")

    # one coordinate transform, reused below
    l, b = apparent_eq_to_galactic(
        pointing.ra_deg, pointing.dec_deg, Time(pointing.time_mjd, format="mjd"))
    pix = hp.ang2pix(nside, l, b, lonlat=True)

    # 2. signal
    sky = SkyModel.from_fits(sky_path)
    signal = sky.sample_galactic(l, b)

    # 3. noise
    if add_noise:
        nmodel = NoiseModel.from_config(cfg)
        noise = nmodel.generate_for_pointing(pointing, rng=np.random.default_rng(seed))
        sigma = nmodel.white_sigma_k
    else:
        noise = np.zeros_like(signal)
        sigma = None

    tod = signal + noise

    # 4. map-making (inverse-variance weighted)
    binned = MapMaker(nside).make(pointing, tod, sigma=sigma, pix=pix)

    if write_maps:
        _write_map(out_dir / "sky_map.fits", binned.signal, "K")
        _write_map(out_dir / "hits.fits", binned.hits.astype(float), "count")
        _write_map(out_dir / "noise_map.fits", binned.noise_map, "K")
    if make_plots:
        _plot_products(binned, pointing, signal, noise, out_dir)
    if save_tod:
        np.savez_compressed(
            out_dir / "tod.npz",
            time_mjd=pointing.time_mjd, ra_deg=pointing.ra_deg,
            dec_deg=pointing.dec_deg, az_deg=pointing.az_deg,
            el_deg=pointing.el_deg, gal_l=l, gal_b=b, pix=pix,
            cal_mask=pointing.cal_mask, signal=signal, noise=noise, tod=tod)

    return PipelineProducts(pointing, signal, noise, binned)


def _write_map(path: Path, m: np.ndarray, unit: str) -> None:
    hp.write_map(str(path), m, coord="G", column_units=unit, overwrite=True)


def _plot_products(binned: BinnedMap, pointing: Pointing, signal: np.ndarray,
                   noise: np.ndarray, out_dir: Path) -> None:
    """Write Mollweide maps (sky, hits, noise) and a one-night TOD plot."""
    seen = binned.seen
    masked = lambda m: np.where(seen, m, hp.UNSEEN)  # noqa: E731

    hp.mollview(masked(binned.signal), coord="G", norm="log", min=10, max=200,
                cmap="inferno", unit="T [K]", title="Recovered sky map")
    hp.graticule()
    plt.savefig(out_dir / "sky_map.png", dpi=120)
    plt.close()

    hp.mollview(masked(binned.hits.astype(float)), coord="G", cmap="viridis",
                unit="hits/pixel", title="Hit count")
    hp.graticule()
    plt.savefig(out_dir / "hits.png", dpi=120)
    plt.close()

    hp.mollview(masked(binned.noise_map), coord="G", cmap="cividis",
                unit="K", title=r"Per-pixel noise $1/\sqrt{W}$")
    hp.graticule()
    plt.savefig(out_dir / "noise_map.png", dpi=120)
    plt.close()

    # One night of TOD: signal and signal+noise vs RA.
    nid = np.unique(pointing.night_id)[0]
    idx = np.flatnonzero(pointing.night_id == nid)
    idx = idx[np.argsort(pointing.time_mjd[idx])]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pointing.ra_deg[idx] / 15.0, signal[idx] + noise[idx], "C3-",
            lw=0.4, alpha=0.7, label="signal + noise")
    ax.plot(pointing.ra_deg[idx] / 15.0, signal[idx], "C0-", lw=0.6, label="signal")
    ax.set(xlabel="RA [h]", ylabel="T [K]", yscale="log",
           title=f"TOD, one night (id={nid})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "tod_one_night.png", dpi=120)
    plt.close(fig)
