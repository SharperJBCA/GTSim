"""Map-making (stage M3): naive inverse-variance binning into a HEALPix map.

Each pixel estimate is the inverse-variance-weighted mean of the samples that
fall in it,

    m_p = sum_{i in p} w_i d_i / sum_{i in p} w_i ,   w_i = 1 / sigma_i^2 ,

with the accumulated weight W_p = sum w_i giving the per-pixel noise
sigma_map(p) = 1 / sqrt(W_p) (= sigma / sqrt(N_p) for uniform white noise). The
pointing (apparent equatorial, of date) is transformed to Galactic to match the
sky model before pixelisation. Calibration-dwell samples are excluded by default.

There is no destriping here: per the project plan, residual 1/f artefacts are
cleaned later in the sky frame (à la the Haslam papers), not in the TOD.
"""

from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np
from astropy.time import Time

from .sky import apparent_eq_to_galactic


@dataclass
class BinnedMap:
    signal: np.ndarray   # K, hp.UNSEEN where unobserved
    weight: np.ndarray   # sum of weights per pixel (1/K^2)
    hits: np.ndarray     # integer hit count per pixel
    nside: int
    coordsys: str = "galactic"

    @property
    def seen(self) -> np.ndarray:
        return self.hits > 0

    @property
    def noise_map(self) -> np.ndarray:
        """Predicted per-pixel noise [K]: 1/sqrt(weight), UNSEEN where unobserved."""
        out = np.full(self.signal.size, hp.UNSEEN)
        w = self.weight > 0
        out[w] = 1.0 / np.sqrt(self.weight[w])
        return out


def bin_tod(pix: np.ndarray, data: np.ndarray, nside: int,
            weights: np.ndarray | None = None) -> BinnedMap:
    """Inverse-variance bin samples ``data`` at pixels ``pix`` into a map."""
    npix = hp.nside2npix(nside)
    if weights is None:
        weights = np.ones(data.size)
    w_sum = np.bincount(pix, weights=weights, minlength=npix)
    wd_sum = np.bincount(pix, weights=weights * data, minlength=npix)
    hits = np.bincount(pix, minlength=npix).astype(np.int64)
    signal = np.full(npix, hp.UNSEEN)
    seen = w_sum > 0
    signal[seen] = wd_sum[seen] / w_sum[seen]
    return BinnedMap(signal, w_sum, hits, nside)


class MapMaker:
    """Bin a pointing + signal TOD into an inverse-variance-weighted map."""

    def __init__(self, nside: int):
        self.nside = nside

    def pointing_pixels(self, pointing) -> np.ndarray:
        """Galactic HEALPix (RING) pixel index for each pointing sample."""
        t = Time(pointing.time_mjd, format="mjd")
        l, b = apparent_eq_to_galactic(pointing.ra_deg, pointing.dec_deg, t)
        return hp.ang2pix(self.nside, l, b, lonlat=True)

    def make(self, pointing, signal, sigma=None, exclude_cal: bool = True,
             pix: np.ndarray | None = None) -> BinnedMap:
        """Make a map from a pointing and its signal TOD.

        ``sigma`` is the per-sample white-noise std: a scalar (uniform) or an
        array; ``None`` uses unit weights. ``pix`` may be supplied to reuse a
        precomputed pixelisation (avoids a repeat coordinate transform).
        """
        if pix is None:
            pix = self.pointing_pixels(pointing)
        data = np.asarray(signal, dtype=float)

        keep = np.ones(data.size, dtype=bool)
        if exclude_cal and getattr(pointing, "cal_mask", None) is not None \
                and pointing.cal_mask.size:
            keep = ~pointing.cal_mask.astype(bool)

        if sigma is None:
            weights = np.ones(int(keep.sum()))
        else:
            sigma = np.asarray(sigma, dtype=float)
            inv_var = 1.0 / sigma ** 2
            weights = np.full(int(keep.sum()), float(inv_var)) if sigma.ndim == 0 \
                else inv_var[keep]

        return bin_tod(pix[keep], data[keep], self.nside, weights)
