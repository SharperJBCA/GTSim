"""Meridian elevation-scan pointing generator (stage M1).

Scan model (Haslam-style "interpretation b"): within each night the telescope
sweeps continuously back and forth along the local meridian, ramping the
commanded declination between ``dec_min_deg`` and ``dec_max_deg`` at
``rate_deg_per_sid_min`` (a triangle wave). The whole sweep pattern is locked to
local sidereal time and advances by ``night_start_offset_sid_s`` sidereal
seconds per calendar night, so the RA grid fills in over the survey.

Because the slew rate is quoted per *sidereal* minute and RA = LST on the
meridian, the declination is a triangle wave in (unwrapped) local sidereal time:

    pattern_arg(t) = (LST_sid_s(t) - n * night_offset) mod period

where ``n`` is the calendar-night index since the survey start. Shifting the
phase by ``night_offset`` each night moves the comb of (RA at which a given Dec
is crossed) by ``night_offset`` sidereal seconds = 0.333 deg/night.

At the turn of each sweep the dish dwells for ``cal_pause_s`` seconds to fire
the gain-calibration diode (Haslam 1970, Fig. 2). This is modelled as a flat top
inserted into the declination triangle: the dwell holds Dec at ``dec_max``
(``cal_at='max'``), ``dec_min`` (``'min'``) or both, lengthening ``period`` by
the dwell(s). Samples taken during a dwell are flagged in ``Pointing.cal_mask``.

Coordinates are *apparent* (true equator/equinox of date); conversion to
ICRS/J2000 or Galactic is deferred to the sky-sampling stage (M2).

Assumptions / simplifications (flagged in docs/PLAN.md):
  * The dark window per night is the contiguous block where the Sun is below
    ``night_sun_alt_deg`` (astronomical twilight, -18 deg); it is found on a
    coarse grid and assumed single (true at mid-latitude in winter).
  * The phase clock advances on every calendar night, including excluded
    (zero-level calibration) nights, which simply contribute no samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

import numpy as np
from astropy import units as u
from astropy.time import Time

from .config import SurveyConfig
from .site import Site

SIDEREAL_PER_SOLAR = 1.00273790935  # sidereal seconds per solar second


@dataclass
class Pointing:
    """Time-ordered pointing for the whole survey (concatenated nights)."""

    time_mjd: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    az_deg: np.ndarray
    el_deg: np.ndarray
    night_id: np.ndarray
    cal_mask: np.ndarray   # True while dwelling for the calibration diode

    @property
    def n_samples(self) -> int:
        return self.time_mjd.size

    @property
    def n_nights(self) -> int:
        return int(np.unique(self.night_id).size) if self.n_samples else 0


class ScanStrategy:
    """Generate :class:`Pointing` from a :class:`SurveyConfig`."""

    SUPPORTED_MODES = ("meridian_elevation",)

    def __init__(self, cfg: SurveyConfig, site: Site | None = None):
        if cfg.scan.mode not in self.SUPPORTED_MODES:
            raise NotImplementedError(
                f"scan mode {cfg.scan.mode!r} is not implemented; supported: "
                f"{self.SUPPORTED_MODES}. Add a new branch in ScanStrategy to "
                "support a different scan pattern.")
        self.cfg = cfg
        self.site = site or Site(cfg.site)

    # -- triangle-wave declination ---------------------------------------
    def _up_sweep_sid_s(self) -> float:
        sc = self.cfg.scan
        span = sc.dec_max_deg - sc.dec_min_deg
        return (span / sc.rate_deg_per_sid_min) * 60.0  # sidereal seconds

    def _dec_cal_from_lst(self, lst_sid_s: np.ndarray, n: int):
        """(Dec [deg], cal_mask) from unwrapped LST [sid s] and night index.

        The declination triangle (dec_min -> dec_max -> dec_min) gains a flat
        calibration dwell at the configured turn(s). ``cal`` is the dwell width
        in sidereal seconds (the diode fires for ``cal_pause_s`` solar seconds,
        during which LST keeps advancing).
        """
        sc = self.cfg.scan
        up = self._up_sweep_sid_s()
        rate = sc.rate_deg_per_sid_min / 60.0           # deg per sidereal second
        cal = sc.cal_pause_s * SIDEREAL_PER_SOLAR        # dwell width in sid s
        cal_at = sc.cal_at if sc.cal_pause_s > 0 else "none"

        if cal_at == "both":
            period = 2.0 * up + 2.0 * cal
        elif cal_at in ("max", "min"):
            period = 2.0 * up + cal
        else:
            period = 2.0 * up

        arg = (lst_sid_s - n * sc.night_start_offset_sid_s) % period
        dec = np.empty_like(arg)
        cal_mask = np.zeros(arg.shape, dtype=bool)

        if cal_at == "max":
            rise = arg < up
            dwell = (arg >= up) & (arg < up + cal)
            fall = arg >= up + cal
            dec[rise] = sc.dec_min_deg + arg[rise] * rate
            dec[dwell] = sc.dec_max_deg
            dec[fall] = sc.dec_max_deg - (arg[fall] - up - cal) * rate
            cal_mask = dwell
        elif cal_at == "min":
            rise = arg < up
            fall = (arg >= up) & (arg < 2.0 * up)
            dwell = arg >= 2.0 * up
            dec[rise] = sc.dec_min_deg + arg[rise] * rate
            dec[fall] = sc.dec_max_deg - (arg[fall] - up) * rate
            dec[dwell] = sc.dec_min_deg
            cal_mask = dwell
        elif cal_at == "both":
            rise = arg < up
            dwell_hi = (arg >= up) & (arg < up + cal)
            fall = (arg >= up + cal) & (arg < 2.0 * up + cal)
            dwell_lo = arg >= 2.0 * up + cal
            dec[rise] = sc.dec_min_deg + arg[rise] * rate
            dec[dwell_hi] = sc.dec_max_deg
            dec[fall] = sc.dec_max_deg - (arg[fall] - up - cal) * rate
            dec[dwell_lo] = sc.dec_min_deg
            cal_mask = dwell_hi | dwell_lo
        else:  # no calibration dwell
            rise = arg <= up
            dec = np.where(rise, sc.dec_min_deg + arg * rate,
                           sc.dec_max_deg - (arg - up) * rate)

        return dec, cal_mask

    # -- observing window per day -----------------------------------------
    def _observing_interval(self, day) -> tuple[Time, Time] | None:
        """The UTC interval observed on ``day``.

        ``night_only`` -> the contiguous Sun-below-threshold (twilight) window;
        otherwise the full 24 h day (the RA-coverage cut then selects what is
        actually kept).
        """
        if self.cfg.window.night_only:
            return self._dark_interval(day)
        midnight = Time(datetime.combine(day, time(0, 0), tzinfo=timezone.utc),
                        scale="utc")
        return midnight, midnight + 86400.0 * u.s

    def _dark_interval(self, day) -> tuple[Time, Time] | None:
        """Contiguous Sun-below-threshold window spanning the night of ``day``.

        Searched on a coarse (4-min) grid from local-ish noon to the next noon.
        """
        noon = datetime.combine(day, time(12, 0), tzinfo=timezone.utc)
        step_s = 240.0
        n = int(24 * 3600 / step_s) + 1
        grid = Time(noon, scale="utc") + np.arange(n) * step_s * u.s
        dark = self.site.sun_alt_deg(grid) < self.cfg.window.night_sun_alt_deg
        if not dark.any():
            return None
        idx = np.flatnonzero(dark)
        return grid[idx[0]], grid[idx[-1]]

    # -- public API -------------------------------------------------------
    def generate(self, start=None, end=None) -> Pointing:
        """Build survey pointing.

        ``start``/``end`` override the config window (e.g. for short test
        slices). The night index that sets the sweep phase is always counted
        from the configured survey start, so slices stay phase-consistent.
        """
        w = self.cfg.window
        start_d = start or w.start
        end_d = end or w.end
        excluded = {d for d in w.exclude_dates}
        dt = self.cfg.scan.sample_dt_s
        ra_lo, ra_hi = self.cfg.coverage.ra_hours

        chunks: list[dict] = []
        day = start_d
        while day < end_d:
            n = (day - w.start).days
            if day in excluded:
                day += timedelta(days=1)
                continue
            interval = self._observing_interval(day)
            if interval is None:
                day += timedelta(days=1)
                continue

            dusk, dawn = interval
            n_samp = int((dawn - dusk).sec / dt)
            if n_samp <= 1:
                day += timedelta(days=1)
                continue
            t = dusk + np.arange(n_samp) * dt * u.s

            lst_h = self.site.lst_hours(t)
            lst_sid_s = lst_h * 3600.0
            # Unwrap LST across the 24h sidereal boundary so the sweep is
            # continuous in time (the sidereal day is not a multiple of the
            # sweep period, so wrapped LST would jump).
            wraps = np.concatenate([[0], np.cumsum(np.diff(lst_sid_s) < -43200.0)])
            lst_unwrapped = lst_sid_s + 86400.0 * wraps

            dec, cal_mask = self._dec_cal_from_lst(lst_unwrapped, n)
            ra = (lst_h * 15.0) % 360.0
            az, el = self.site.meridian_altaz(dec)

            ra_h = ra / 15.0
            keep = (ra_h >= ra_lo) & (ra_h < ra_hi)
            if keep.any():
                chunks.append(
                    dict(
                        time_mjd=t.mjd[keep],
                        ra_deg=ra[keep],
                        dec_deg=dec[keep],
                        az_deg=az[keep],
                        el_deg=el[keep],
                        night_id=np.full(int(keep.sum()), n, dtype=int),
                        cal_mask=cal_mask[keep],
                    )
                )
            day += timedelta(days=1)

        if not chunks:
            empty = np.array([])
            return Pointing(empty, empty, empty, empty, empty,
                            np.array([], dtype=int), np.array([], dtype=bool))

        return Pointing(
            time_mjd=np.concatenate([c["time_mjd"] for c in chunks]),
            ra_deg=np.concatenate([c["ra_deg"] for c in chunks]),
            dec_deg=np.concatenate([c["dec_deg"] for c in chunks]),
            az_deg=np.concatenate([c["az_deg"] for c in chunks]),
            el_deg=np.concatenate([c["el_deg"] for c in chunks]),
            night_id=np.concatenate([c["night_id"] for c in chunks]),
            cal_mask=np.concatenate([c["cal_mask"] for c in chunks]),
        )
