#!/usr/bin/env python
"""Estimate the 1/f noise power spectrum of the regridded Haslam 1970 TOD.

Run (inside the ``lbs`` env):
    micromamba run -n lbs python scripts/measure_1f_noise.py

Uses the 0-800 s segment of ``haslam1970_tod_regridded_1s.txt`` (which sits
before the gain-calibration box). A high-pass filter removes scales larger than
200 s (f < 1/200 = 0.005 Hz) so the slowly-varying Galactic sky signal does not
masquerade as steep 1/f. The power spectrum is then fit with

    P(f) = w * (1 + (f_knee / f)^alpha)

in the band above the high-pass cutoff and below ~0.2 Hz (the trustworthy limit
set by the ~2.5 s native digitisation spacing; higher frequencies are
interpolation, not data).

Note: this short segment is sky-dominated, so the measured power-law index
characterises the residual Galactic signal plus instrumental 1/f, not pure
detector 1/f. No white-noise plateau is reached in the trusted band, so only
the spectral slope is reported; isolating the detector knee needs the sky
removed (scan differencing or a sky model).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / "ancillary_data" / "haslam1970"
SRC = DATADIR / "haslam1970_tod_regridded_1s.txt"
PLOT = DATADIR / "haslam1970_tod_1f_psd.png"

FS = 1.0                  # sampling rate [Hz] (1 s grid)
T_MAX = 800.0             # segment end [s]
HP_SCALE_S = 200.0        # remove scales larger than this
F_HP = 1.0 / HP_SCALE_S   # high-pass cutoff [Hz]
F_FIT_HI = 0.2            # upper trustworthy frequency [Hz]


F_REF = 0.01              # reference frequency for the amplitude [Hz]


def power_law(f, log_a, alpha):
    """Red-noise power law P = A * f^-alpha (fit in log space)."""
    return log_a - alpha * np.log(f)


def logbin(f, p, nbins=24):
    """Geometric-mean binning of a periodogram (f>0)."""
    good = f > 0
    f, p = f[good], p[good]
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), nbins + 1)
    idx = np.digitize(f, edges)
    fb, pb = [], []
    for k in range(1, nbins + 1):
        sel = idx == k
        if sel.sum() >= 1:
            fb.append(np.exp(np.mean(np.log(f[sel]))))
            pb.append(np.mean(p[sel]))
    return np.array(fb), np.array(pb)


def main() -> int:
    d = np.genfromtxt(SRC, delimiter=",", comments="#")
    t, _, b = d.T
    m = (t >= 0) & (t < T_MAX)
    t, b = t[m], b[m]

    # Linear detrend then high-pass (zero-phase Butterworth) to drop >200 s.
    b_dt = signal.detrend(b, type="linear")
    sos = signal.butter(4, F_HP, btype="highpass", fs=FS, output="sos")
    b_hp = signal.sosfiltfilt(sos, b_dt)

    # Periodogram of the high-passed segment.
    f, pxx = signal.periodogram(b_hp, fs=FS, window="hann", detrend=False)
    fb, pb = logbin(f, pxx)

    # Fit a power law P = A f^-alpha in the valid band (log space, robust).
    band = (f >= F_HP) & (f <= F_FIT_HI)
    popt, _ = curve_fit(power_law, f[band], np.log(pxx[band]), p0=(0.0, 1.5))
    log_a, alpha = popt
    p_ref = np.exp(log_a) * F_REF ** (-alpha)

    print(f"segment       : 0-{T_MAX:.0f} s, {t.size} samples @ {FS:.0f} Hz")
    print(f"high-pass     : f > {F_HP:.4f} Hz (scales < {HP_SCALE_S:.0f} s)")
    print(f"fit band      : {F_HP:.4f} .. {F_FIT_HI:.2f} Hz")
    print(f"slope         : alpha = {alpha:.2f}  (P ~ f^-alpha)")
    print(f"amplitude     : P({F_REF:.2f} Hz) = {p_ref:.3e} arb^2/Hz")
    print("note          : no white plateau in band -> slope only; "
          "segment is sky-dominated")
    print(f"plot          : {PLOT}")

    _plot(f, pxx, fb, pb, (log_a, alpha), PLOT)
    return 0


def _plot(f, pxx, fb, pb, fit, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(f[1:], pxx[1:], color="0.8", lw=0.6, label="periodogram")
    ax.loglog(fb, pb, "o", ms=4, color="C0", label="log-binned")
    log_a, alpha = fit
    ff = np.logspace(np.log10(F_HP), np.log10(F_FIT_HI), 200)
    ax.loglog(ff, np.exp(log_a) * ff ** (-alpha), "C3-",
              label=fr"fit: $P\propto f^{{-\alpha}}$, $\alpha$={alpha:.2f}")
    ax.axvline(F_HP, color="0.5", ls="-.", lw=0.8, label="HP cutoff (200 s)")
    ax.axvspan(F_FIT_HI, FS / 2, color="0.9", label="interp. (untrusted)")
    ax.set(xlabel="frequency [Hz]", ylabel="PSD [arb$^2$/Hz]",
           title="Haslam 1970 TOD (0-800 s) — 1/f power spectrum")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
