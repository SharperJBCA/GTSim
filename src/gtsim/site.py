"""Observatory site: coordinate transforms and timing utilities.

For meridian pointing the geometry is analytic and cheap, so we use closed-form
expressions for the bulk pointing and reserve astropy's full transform for
cross-checks (see :meth:`Site.altaz_to_radec`).

Meridian (upper-transit) relations at geographic latitude ``phi``:
    elevation a = 90 - |phi - dec|
    azimuth   = 180 (due south) if dec < phi, else 0 (due north)
    right ascension = local apparent sidereal time (hour angle = 0)
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import TETE, AltAz, EarthLocation, get_sun
from astropy.time import Time

from .config import SiteConfig

# Ratio of mean solar to sidereal time: 1 sidereal second = this many SI seconds.
SIDEREAL_TO_SOLAR = 1.0 / 1.00273790935


class Site:
    """An observatory location with timing/pointing helpers."""

    def __init__(self, cfg: SiteConfig):
        self.cfg = cfg
        self.location = EarthLocation(
            lat=cfg.lat_deg * u.deg,
            lon=cfg.lon_deg * u.deg,
            height=cfg.height_m * u.m,
        )

    @property
    def lat_deg(self) -> float:
        return self.cfg.lat_deg

    # -- timing -----------------------------------------------------------
    def lst_hours(self, time: Time) -> np.ndarray:
        """Local apparent sidereal time [hours], vectorised over ``time``."""
        lst = time.sidereal_time("apparent", longitude=self.location.lon)
        return np.atleast_1d(lst.hour)

    # -- meridian geometry (analytic) -------------------------------------
    def meridian_altaz(self, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Azimuth, elevation [deg] for a source at upper transit at ``dec``."""
        dec = np.asarray(dec_deg, dtype=float)
        el = 90.0 - np.abs(self.lat_deg - dec)
        az = np.where(dec < self.lat_deg, 180.0, 0.0)
        return az, el

    def meridian_radec(
        self, time: Time, dec_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """RA, Dec [deg] for meridian pointing at ``dec_deg`` and ``time``.

        On the meridian the hour angle is zero, so RA equals the local
        apparent sidereal time; Dec is the commanded declination. These are
        *apparent* coordinates (true equator/equinox of date); converting to
        ICRS/J2000 or Galactic (a precession+nutation step) is deferred to the
        sky-sampling stage. ``time`` and ``dec_deg`` must broadcast together.
        """
        ra = (self.lst_hours(time) * 15.0) % 360.0
        dec = np.broadcast_to(np.asarray(dec_deg, dtype=float), ra.shape).copy()
        return ra, dec

    # -- sun / night ------------------------------------------------------
    def sun_alt_deg(self, time: Time) -> np.ndarray:
        """Sun altitude [deg] at ``time``, vectorised."""
        frame = AltAz(obstime=time, location=self.location)
        alt = get_sun(time).transform_to(frame).alt
        return np.atleast_1d(alt.deg)

    # -- full transform (cross-check) -------------------------------------
    def altaz_to_radec(
        self, time: Time, az_deg: np.ndarray, el_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full astropy AltAz->apparent (TETE) transform.

        Used to validate the analytic meridian path. TETE (true equator/equinox
        of date) is the frame the sidereal-time relation produces, so the two
        agree to sub-arcsecond precision.
        """
        frame = AltAz(
            obstime=time,
            location=self.location,
            az=np.asarray(az_deg) * u.deg,
            alt=np.asarray(el_deg) * u.deg,
        )
        apparent = frame.transform_to(TETE(obstime=time))
        return np.atleast_1d(apparent.ra.deg), np.atleast_1d(apparent.dec.deg)
