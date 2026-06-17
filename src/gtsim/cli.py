"""Command-line interface: ``gtsim run`` drives the full simulation pipeline."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gtsim", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run scan -> sky -> noise -> map pipeline")
    r.add_argument("--config", required=True, help="survey TOML config")
    r.add_argument("--sky", required=True, help="input HEALPix sky FITS (Galactic, K)")
    r.add_argument("--outdir", default="outputs/run", help="output directory")
    r.add_argument("--nside", type=int, default=64, help="output map NSIDE")
    r.add_argument("--start", help="override window start (YYYY-MM-DD)")
    r.add_argument("--end", help="override window end (YYYY-MM-DD)")
    r.add_argument("--nights", type=int,
                   help="run only the first N nights from the start date")
    r.add_argument("--no-noise", action="store_true", help="signal only (no noise)")
    r.add_argument("--seed", type=int, default=0, help="noise RNG seed")
    r.add_argument("--save-tod", action="store_true", help="also save the TOD (.npz)")
    r.add_argument("--plots", action="store_true", help="also write PNG plots")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    if args.nights is not None:
        from .config import load_config
        base = start or load_config(args.config).window.start
        start, end = base, base + timedelta(days=args.nights)

    prod = run_pipeline(
        config_path=args.config, sky_path=args.sky, out_dir=args.outdir,
        start=start, end=end, nside=args.nside, add_noise=not args.no_noise,
        seed=args.seed, save_tod=args.save_tod, make_plots=args.plots,
    )
    b = prod.binned
    seen = int(b.seen.sum())
    print(f"samples     : {prod.pointing.n_samples}")
    print(f"coverage    : {seen}/{b.seen.size} pixels "
          f"({100 * seen / b.seen.size:.0f}%) at nside={b.nside}")
    print(f"noise       : {'on' if prod.noise.any() else 'off'}")
    extras = " + PNG plots" if args.plots else ""
    print(f"outputs     : {Path(args.outdir).resolve()} "
          f"(sky_map.fits, hits.fits, noise_map.fits{extras})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
