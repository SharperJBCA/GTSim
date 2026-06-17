"""Sky model (stage M2): a HEALPix brightness-temperature sky sampled along the
telescope pointing.

The reference sky is the reprocessed 408 MHz Haslam map (Remazeilles et al.
2014): NSIDE=512, RING ordering, Galactic coordinates, units K. That map is
already convolved with the ~1 deg survey beam, so when it is used as the "truth"
sky it is sampled directly by bilinear interpolation -- no extra beam smoothing.
Optional Gaussian beam smoothing is provided for synthetic skies that start at
delta-function resolution.

Coordinate handling: the M1 pointing is in *apparent* equatorial coordinates
(true equator/equinox of date). Sampling therefore transforms apparent (RA, Dec)
at each sample time to Galactic (l, b) -- this folds in precession from the
1965/66 observing epoch to the J2000-defined Galactic frame -- and interpolates
the map at (l, b).
"""

from __future__ import annotations

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import TETE, Galactic, SkyCoord
from astropy.time import Time


def apparent_eq_to_galactic(ra_deg, dec_deg, time: Time):
    """Apparent equatorial (TETE, of date) -> Galactic (l, b) in degrees."""
    coord = SkyCoord(ra=np.asarray(ra_deg) * u.deg,
                     dec=np.asarray(dec_deg) * u.deg,
                     frame=TETE(obstime=time))
    gal = coord.transform_to(Galactic())
    return gal.l.deg, gal.b.deg


class SkyModel:
    """A HEALPix sky in brightness temperature [K], in Galactic coordinates."""

    def __init__(self, hpx_map: np.ndarray, coordsys: str = "galactic",
                 unit: str = "K"):
        self.map = np.asarray(hpx_map, dtype=float)
        self.nside = hp.npix2nside(self.map.size)
        self.coordsys = coordsys
        self.unit = unit

    @classmethod
    def from_fits(cls, path) -> "SkyModel":
        """Load a HEALPix map (assumed Galactic, K) from a FITS file."""
        return cls(hp.read_map(str(path)))

    def smoothed(self, fwhm_deg: float) -> "SkyModel":
        """Return a copy smoothed with a Gaussian beam of the given FWHM."""
        sm = hp.smoothing(self.map, fwhm=np.radians(fwhm_deg))
        return SkyModel(sm, coordsys=self.coordsys, unit=self.unit)

    def sample_galactic(self, l_deg, b_deg) -> np.ndarray:
        """Bilinearly interpolate the map at Galactic (l, b) [deg]."""
        return hp.get_interp_val(self.map, l_deg, b_deg, lonlat=True)

    def sample_equatorial_apparent(self, ra_deg, dec_deg, time: Time) -> np.ndarray:
        """Sample at apparent equatorial (RA, Dec) [deg] for the given time(s)."""
        l, b = apparent_eq_to_galactic(ra_deg, dec_deg, time)
        return self.sample_galactic(l, b)

    def sample_pointing(self, pointing) -> np.ndarray:
        """Sample the sky along a :class:`gtsim.scan.Pointing` -> signal TOD [K]."""
        time = Time(pointing.time_mjd, format="mjd")
        return self.sample_equatorial_apparent(pointing.ra_deg,
                                               pointing.dec_deg, time)
