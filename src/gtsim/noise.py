"""Noise model (stage M4): white + 1/f time-ordered noise.

The one-sided power spectral density is

    P(f) = P_white * (1 + (f / f_knee)^alpha)

with the white level fixed by the noise-equivalent temperature (NET):

    P_white = 2 * NET^2   [K^2/Hz]   (independent of the sampling cadence)

and the per-sample white standard deviation sigma = NET / sqrt(dt). ``alpha`` is
the 1/f spectral index (P ~ f^alpha); alpha = -1 is classic 1/f. For the Mark 1
survey NET = 0.2 K.s^0.5, f_knee = 0.5 Hz, alpha = -1 (an assumed *instrumental*
1/f, since the digitised trace is signal-dominated and cannot constrain it).

Realisations are synthesised in the Fourier domain: each rfft bin is drawn from a
complex Gaussian with E[|X_k|^2] = P(f_k) * N * fs / 2, which gives a real time
series whose expected periodogram is P(f).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SurveyConfig


@dataclass
class NoiseModel:
    net_k_sqrt_s: float
    f_knee_hz: float
    alpha: float
    sample_dt_s: float = 1.0

    @classmethod
    def from_config(cls, cfg: SurveyConfig) -> "NoiseModel":
        ins = cfg.instrument
        return cls(net_k_sqrt_s=ins.net_k_sqrt_s, f_knee_hz=ins.f_knee_hz,
                   alpha=ins.alpha, sample_dt_s=cfg.scan.sample_dt_s)

    @property
    def white_sigma_k(self) -> float:
        """Per-sample white-noise std [K] at the model sampling cadence."""
        return self.net_k_sqrt_s / np.sqrt(self.sample_dt_s)

    @property
    def white_psd(self) -> float:
        """One-sided white-noise PSD level [K^2/Hz] (= 2*NET^2)."""
        return 2.0 * self.net_k_sqrt_s ** 2

    def psd_model(self, f) -> np.ndarray:
        """One-sided noise PSD [K^2/Hz] at frequencies ``f`` (inf at f=0)."""
        f = np.asarray(f, dtype=float)
        out = np.full(f.shape, np.inf)
        nz = f > 0
        out[nz] = self.white_psd * (1.0 + (f[nz] / self.f_knee_hz) ** self.alpha)
        return out

    def generate(self, n_samples: int, dt: float | None = None, rng=None) -> np.ndarray:
        """Generate a noise realisation [K] of length ``n_samples``."""
        dt = self.sample_dt_s if dt is None else dt
        rng = np.random.default_rng() if rng is None else rng
        fs = 1.0 / dt
        f = np.fft.rfftfreq(n_samples, d=dt)

        psd = np.empty(f.shape)
        nz = f > 0
        psd[nz] = self.white_psd * (1.0 + (f[nz] / self.f_knee_hz) ** self.alpha)
        psd[~nz] = 0.0                      # no DC (zero-mean)

        amp = np.sqrt(psd * n_samples * fs / 2.0)
        re = rng.standard_normal(f.size)
        im = rng.standard_normal(f.size)
        spec = amp * (re + 1j * im) / np.sqrt(2.0)
        spec[0] = 0.0
        if n_samples % 2 == 0:
            spec[-1] = amp[-1] * re[-1]     # Nyquist bin is real
        return np.fft.irfft(spec, n=n_samples)

    def generate_for_pointing(self, pointing, rng=None) -> np.ndarray:
        """Noise TOD [K] aligned to a pointing, drawn independently per night.

        1/f correlations are continuous within a night but not across the day
        gaps between observing sessions, so each night gets its own realisation.
        """
        rng = np.random.default_rng() if rng is None else rng
        out = np.empty(pointing.n_samples)
        for nid in np.unique(pointing.night_id):
            idx = np.flatnonzero(pointing.night_id == nid)
            order = idx[np.argsort(pointing.time_mjd[idx])]
            out[order] = self.generate(idx.size, rng=rng)
        return out
