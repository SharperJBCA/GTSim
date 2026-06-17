"""Survey configuration: dataclasses + TOML loader.

One TOML file describes one survey (site, instrument, scan strategy, observing
window, coverage, systematics). Optional fields default to ``None`` when absent
from the file so configs can grow without breaking older ones.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class SiteConfig:
    name: str
    lat_deg: float
    lon_deg: float
    height_m: float = 0.0


@dataclass
class InstrumentConfig:
    freq_mhz: float
    beam_fwhm_deg: float | None = None
    net_k_sqrt_s: float | None = None
    system_temp_k: float | None = None
    f_knee_hz: float | None = None
    alpha: float | None = None


@dataclass
class ScanConfig:
    mode: str
    rate_deg_per_sid_min: float
    night_start_offset_sid_s: float  # sweep-phase advance per calendar night
    dec_min_deg: float
    dec_max_deg: float
    azimuth: str = "auto"        # "auto" | "south" | "north"
    sample_dt_s: float = 1.0
    cal_pause_s: float = 0.0     # calibration-diode dwell at the turn(s); 0 = off
    cal_at: str = "max"          # "max" | "min" | "both"


@dataclass
class WindowConfig:
    start: date
    end: date
    night_only: bool = True
    night_sun_alt_deg: float = 0.0
    exclude_dates: list[date] = field(default_factory=list)


@dataclass
class CoverageConfig:
    ra_hours: tuple[float, float] = (0.0, 24.0)


@dataclass
class SystematicsConfig:
    ground: bool = False
    atmosphere: bool = False


@dataclass
class SurveyConfig:
    name: str
    site: SiteConfig
    instrument: InstrumentConfig
    scan: ScanConfig
    window: WindowConfig
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    systematics: SystematicsConfig = field(default_factory=SystematicsConfig)


def _as_date(value: object) -> date:
    """Accept either a TOML date (parsed to ``datetime.date``) or ISO string."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"expected a date or ISO date string, got {value!r}")


def load_config(path: str | Path) -> SurveyConfig:
    """Load a survey configuration from a TOML file."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    window_raw = dict(raw["window"])
    exclude = [_as_date(d) for d in window_raw.pop("exclude_dates", [])]
    window = WindowConfig(
        start=_as_date(window_raw.pop("start")),
        end=_as_date(window_raw.pop("end")),
        exclude_dates=exclude,
        **window_raw,
    )

    coverage_raw = dict(raw.get("coverage", {}))
    if "ra_hours" in coverage_raw:
        coverage_raw["ra_hours"] = tuple(coverage_raw["ra_hours"])
    coverage = CoverageConfig(**coverage_raw)

    return SurveyConfig(
        name=raw["name"],
        site=SiteConfig(**raw["site"]),
        instrument=InstrumentConfig(**raw["instrument"]),
        scan=ScanConfig(**raw["scan"]),
        window=window,
        coverage=coverage,
        systematics=SystematicsConfig(**raw.get("systematics", {})),
    )
