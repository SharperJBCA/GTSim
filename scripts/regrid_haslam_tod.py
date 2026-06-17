#!/usr/bin/env python
"""Regrid the digitised Haslam 1970 Fig. 2 TOD onto a regular 1-second grid.

Run (inside the ``lbs`` env):
    micromamba run -n lbs python scripts/regrid_haslam_tod.py

Source: ``haslam1970_tod_data_digitised.txt`` — a hand-digitised chart-recorder
trace (RA in hours vs brightness in arbitrary units) of a *pair* of meridian
scans (Dec +40->+60->+40) with a 25 K gain-calibration box at the turn.

Because the dish observes on the meridian, RA = local sidereal time, so RA is a
linear proxy for time: 1 sidereal second of RA == 1 sidereal second elapsed.
We therefore map RA -> elapsed time and resample onto a uniform grid at the
instrument's 1-second (solar) integration cadence. Time increases with RA
(forward transit order). Brightness is left in arbitrary units; the calibration
box and point source (3C 86) remain in the trace and should be masked before
measuring sky/noise statistics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIDEREAL_PER_SOLAR = 1.00273790935  # sidereal seconds per solar second
GRID_DT_S = 1.0                     # solar seconds (integration time)

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / "ancillary_data" / "haslam1970"
SRC = DATADIR / "haslam1970_tod_data_digitised.txt"
OUT = DATADIR / "haslam1970_tod_regridded_1s.txt"
PLOT = DATADIR / "haslam1970_tod_regridded_1s.png"


def main() -> int:
    raw = np.genfromtxt(SRC, delimiter=",", comments="#")
    ra_raw, b_raw = raw[:, 0], raw[:, 1]

    # Average exact-duplicate RA values and sort ascending (RA increasing with
    # time). np.unique returns sorted unique RA plus an inverse index we use to
    # average the brightness of any repeated RA.
    ra_u, inv = np.unique(ra_raw, return_inverse=True)
    b_u = np.bincount(inv, weights=b_raw) / np.bincount(inv)

    # RA (sidereal) -> elapsed solar time, t=0 at the earliest transit.
    ra_min = ra_u[0]
    t_src_s = (ra_u - ra_min) * 3600.0 / SIDEREAL_PER_SOLAR  # solar seconds

    # Uniform 1-second grid.
    n = int(np.floor(t_src_s[-1] / GRID_DT_S)) + 1
    t_grid = np.arange(n) * GRID_DT_S
    b_grid = np.interp(t_grid, t_src_s, b_u)
    ra_grid = ra_min + (t_grid * SIDEREAL_PER_SOLAR) / 3600.0  # hours

    header = (
        "Haslam 1970 Fig. 2 TOD, regridded to a uniform 1-second grid.\n"
        "Source: haslam1970_tod_data_digitised.txt (RA hrs, brightness arb.).\n"
        f"Mapping: RA=LST; t = (RA-RA0)*3600/{SIDEREAL_PER_SOLAR} solar s; "
        "time increases with RA.\n"
        "Brightness is arbitrary units (cal box + 3C86 still present).\n"
        "columns: time_s, ra_hours, brightness"
    )
    np.savetxt(OUT, np.column_stack([t_grid, ra_grid, b_grid]),
               fmt=["%.3f", "%.6f", "%.6e"], delimiter=",", header=header)

    print(f"source points : {ra_raw.size} (-> {ra_u.size} unique RA)")
    print(f"RA span       : {ra_min:.4f} .. {ra_u[-1]:.4f} hrs")
    print(f"duration       : {t_src_s[-1]:.1f} solar s ({t_src_s[-1]/60:.2f} min)")
    print(f"grid          : {n} samples @ {GRID_DT_S:.0f} s")
    print(f"median src dt : {np.median(np.diff(t_src_s)):.2f} s "
          f"(max {np.diff(t_src_s).max():.2f} s) -> 1 s grid oversamples")
    print(f"wrote         : {OUT}")

    _plot(ra_raw, b_raw, ra_grid, b_grid, t_grid, PLOT)
    print(f"plot          : {PLOT}")
    return 0


def _plot(ra_raw, b_raw, ra_grid, b_grid, t_grid, path: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6))
    ax0.plot(ra_raw, b_raw, ".", ms=2, alpha=0.4, label="digitised")
    ax0.plot(ra_grid, b_grid, "-", lw=0.7, label="regridded 1 s")
    ax0.invert_xaxis()  # match the figure: RA decreasing to the right
    ax0.set(xlabel="RA [h]", ylabel="brightness [arb.]",
            title="Haslam 1970 Fig. 2 TOD — regridded vs digitised")
    ax0.legend(loc="upper right")
    ax0.grid(alpha=0.3)

    ax1.plot(t_grid, b_grid, "-", lw=0.7)
    ax1.set(xlabel="time [s]", ylabel="brightness [arb.]",
            title="Regridded TOD on uniform 1-second grid")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
