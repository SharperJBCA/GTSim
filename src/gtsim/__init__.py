"""Ground-based single-dish telescope simulator.

Stage M1 provides configuration loading, site/coordinate utilities, and the
meridian elevation-scan pointing generator. See ``docs/PLAN.md``.
"""

from .config import SurveyConfig, load_config
from .site import Site
from .scan import ScanStrategy, Pointing
from .sky import SkyModel, apparent_eq_to_galactic
from .noise import NoiseModel
from .mapmaker import MapMaker, BinnedMap, bin_tod

__all__ = [
    "SurveyConfig",
    "load_config",
    "Site",
    "ScanStrategy",
    "Pointing",
    "SkyModel",
    "apparent_eq_to_galactic",
    "NoiseModel",
    "MapMaker",
    "BinnedMap",
    "bin_tod",
]
