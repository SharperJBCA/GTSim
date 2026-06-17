#!/usr/bin/env python
"""M4 noise-model validation: white + 1/f power spectrum, and a signal+noise demo.

Run (inside the ``lbs`` env):
    micromamba run -n lbs python scripts/noise_test.py

The model PSD is P(f) = 2*NET^2 * (1 + (f/f_knee)^alpha). With the survey's 1 s
sampling the knee (0.5 Hz) sits at Nyquist, so to *verify* the model shape we
synthesise at a finer cadence where the white plateau, knee and 1/f slope are
all visible. We then add noise to the Haslam signal TOD at the real 1 s cadence.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtsim import NoiseModel, ScanStrategy, SkyModel, load_config  # noqa: E402

CONFIG = ROOT / "configs" / "mark1.toml"
HASLAM = ROOT / "ancillary_data" / "haslam408_ds_Remazeilles2014.fits"
OUTDIR = ROOT / "outputs"

DT_DEMO = 0.05            # fine cadence for PSD validation [s] (fs = 20 Hz)
N_DEMO = 2 ** 18         # ~3.6 h of samples


def main() -> int:
    OUTDIR.mkdir(exist_ok=True)
    cfg = load_config(CONFIG)
    noise = NoiseModel.from_config(cfg)
    rng = np.random.default_rng(1)

    print(f"NET         : {noise.net_k_sqrt_s} K.s^0.5 -> sigma(1 s) = "
          f"{noise.white_sigma_k:.3f} K")
    print(f"white PSD   : {noise.white_psd:.4f} K^2/Hz (= 2*NET^2)")
    print(f"1/f         : f_knee = {noise.f_knee_hz} Hz, alpha = {noise.alpha}")

    # --- PSD validation at a fine cadence ------------------------------
    x = noise.generate(N_DEMO, dt=DT_DEMO, rng=rng)
    f, pxx = sps.welch(x, fs=1.0 / DT_DEMO, nperseg=8192)
    fpos = f > 0

    white_meas = np.median(pxx[f > 2 * noise.f_knee_hz])     # above the knee
    sub = (f >= 0.005) & (f <= 0.05)                         # well below the knee
    slope = np.polyfit(np.log(f[sub]), np.log(pxx[sub]), 1)[0]
    # knee: where measured PSD crosses 2*white (model crossover value)
    target = 2.0 * noise.white_psd
    fk_meas = f[fpos][np.argmin(np.abs(pxx[fpos] - target))]

    print(f"measured    : white = {white_meas:.4f} K^2/Hz "
          f"(model {noise.white_psd:.4f})")
    print(f"measured    : slope = {slope:.2f} (model {noise.alpha}; "
          "Welch leakage biases steep spectra slightly flat)")
    print(f"measured    : knee ~ {fk_meas:.2f} Hz (model {noise.f_knee_hz})")
    ok = (abs(white_meas / noise.white_psd - 1) < 0.2
          and abs(slope - noise.alpha) < 0.2
          and abs(fk_meas / noise.f_knee_hz - 1) < 0.5)

    _plot_psd(f[fpos], pxx[fpos], noise, OUTDIR / "noise_psd.png")

    # --- Signal + noise demo at the survey 1 s cadence -----------------
    sky = SkyModel.from_fits(HASLAM)
    strat = ScanStrategy(cfg)
    pt = strat.generate(end=cfg.window.start + timedelta(days=1))
    sig = sky.sample_pointing(pt)
    nse = noise.generate_for_pointing(pt, rng=rng)
    _plot_signal_noise(pt, sig, nse, OUTDIR / "noise_tod_demo.png")

    print(f"plots       : {OUTDIR}/noise_psd.png, {OUTDIR}/noise_tod_demo.png")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _plot_psd(f, pxx, noise, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(f, pxx, color="0.7", lw=0.6, label="periodogram (Welch)")
    ax.loglog(f, noise.psd_model(f), "C3-", lw=1.4,
              label=fr"model $2\,NET^2(1+(f/f_k)^{{\alpha}})$")
    ax.axhline(noise.white_psd, color="C0", ls=":", lw=0.9,
               label=f"white = {noise.white_psd:.3f} K$^2$/Hz")
    ax.axvline(noise.f_knee_hz, color="C2", ls="--", lw=0.9,
               label=f"$f_k$ = {noise.f_knee_hz} Hz")
    ax.set(xlabel="frequency [Hz]", ylabel="PSD [K$^2$/Hz]",
           title="Noise model PSD: white + 1/f (validated at 20 Hz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_signal_noise(pt, sig, nse, path: Path) -> None:
    nid = np.unique(pt.night_id)[0]
    idx = np.flatnonzero(pt.night_id == nid)
    idx = idx[np.argsort(pt.time_mjd[idx])][:900]
    t = (pt.time_mjd[idx] - pt.time_mjd[idx][0]) * 86400.0
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, sig[idx], "C0-", lw=0.7, label="signal (K)")
    ax.plot(t, sig[idx] + nse[idx], "C3-", lw=0.5, alpha=0.7,
            label="signal + noise (K)")
    ax.set(xlabel="time [s]", ylabel="T [K]",
           title="Haslam signal + white/1f noise, one night (1 s cadence)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
