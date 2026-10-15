"""Command line interface: ``fishgen gui``, ``fishgen generate`` and ``fishgen preview``."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from .animation import generate
from .export import FORMATS, export_all
from .genome import KINDS
from .species import RANDOM, SPECIES_NAMES, genome_for, kind_of


def parse_size(text: str) -> tuple[int, int]:
    """'32' -> (32, 32); '48x32' -> (48, 32)."""
    try:
        if "x" in text.lower():
            w, h = text.lower().split("x")
            size = int(w), int(h)
        else:
            size = int(text), int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must look like 32 or 48x32, got {text!r}")
    if min(size) < 8 or max(size) > 512:
        raise argparse.ArgumentTypeError("size must be between 8 and 512 pixels")
    return size


def parse_formats(text: str) -> list[str]:
    if text == "all":
        return list(FORMATS)
    fmts = [f.strip() for f in text.split(",") if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown format(s) {bad}; choose from {', '.join(FORMATS)} or 'all'")
    return fmts


def _add_render_args(p: argparse.ArgumentParser):
    p.add_argument("--seed", type=int, default=None, help="seed (random if omitted)")
    p.add_argument("--kind", default=None, choices=KINDS,
                   help="kind of animal; a random one of this kind unless --species picks a preset")
    p.add_argument("--species", default=None, choices=SPECIES_NAMES, metavar="NAME",
                   help=f"preset (implies its kind): {', '.join(SPECIES_NAMES)}")
    p.add_argument("--size", type=parse_size, default=(32, 32), metavar="N|WxH", help="sprite size (default 32)")
    p.add_argument("--swim-frames", type=int, default=8, help="frames per swim cycle (default 8)")
    p.add_argument("--turn-frames", type=int, default=10, help="frames per turn-around (default 10)")
    p.add_argument("--fps", type=float, default=10.0, help="playback speed (default 10)")
    p.add_argument("--pitch", type=float, default=0.25, help="camera tilt in radians (default 0.25)")


def cmd_generate(args) -> int:
    first = args.seed if args.seed is not None else random.randrange(1_000_000)
    seeds = [first + i for i in range(args.count)]
    out_root = Path(args.out)
    for seed in seeds:
        t0 = time.perf_counter()
        genome = genome_for(seed, args.species, args.kind)
        sprite = generate(seed, *args.size, swim_frames=args.swim_frames, turn_frames=args.turn_frames,
                          fps=args.fps, pitch=args.pitch, genome=genome)
        out_dir = out_root / f"{genome.kind}_{seed}" if args.count > 1 or not args.flat else out_root
        files = export_all(sprite, out_dir, args.formats, gif_scale=args.scale, frame_scale=args.frame_scale,
                           background=args.bg, padding=args.padding, godot_dir=args.godot_dir)
        print(f"{genome.kind} {seed}: {len(files)} files -> {out_dir}  ({time.perf_counter() - t0:.2f}s)")
    return 0


def cmd_preview(args) -> int:
    try:
        from .preview import run
    except ImportError as e:  # pragma: no cover - depends on optional dep
        print(f"preview needs pygame-ce: pip install 'fishgen[preview]'  ({e})", file=sys.stderr)
        return 1
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    run(seed=seed, size=args.size, count=args.fish, zoom=args.zoom, swim_frames=args.swim_frames,
        turn_frames=args.turn_frames, fps=args.fps, pitch=args.pitch, out=Path(args.out),
        species=args.species, kind=args.kind)
    return 0


def cmd_gui(args) -> int:
    from .gui import run
    from .species import RANDOM

    species = args.species or RANDOM
    kind = args.kind or (kind_of(species) if species != RANDOM else "fish")
    return run(seed=args.seed, species=species, out=Path(args.out), kind=kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fishgen", description="Procedural pixel-art fish generator.")
    sub = parser.add_subparsers(dest="command")

    gu = sub.add_parser("gui", help="open the fish designer window (the default with no command)")
    gu.add_argument("--seed", type=int, default=None, help="starting seed (random if omitted)")
    gu.add_argument("--kind", default=None, choices=KINDS, help="starting kind of animal")
    gu.add_argument("--species", default=None, choices=SPECIES_NAMES, metavar="NAME", help="starting preset")
    gu.add_argument("--out", default="out", help="default export folder (default ./out)")
    gu.set_defaults(func=cmd_gui)

    g = sub.add_parser("generate", help="generate fish and export them")
    _add_render_args(g)
    g.add_argument("--count", type=int, default=1, help="how many fish (consecutive seeds)")
    g.add_argument("--formats", type=parse_formats, default=list(FORMATS),
                   help=f"comma list of {','.join(FORMATS)} or 'all' (default all)")
    g.add_argument("--out", default="out", help="output directory (default ./out)")
    g.add_argument("--flat", action="store_true", help="with a single fish, write straight into --out")
    g.add_argument("--scale", type=int, default=4, help="GIF upscale factor (default 4)")
    g.add_argument("--frame-scale", type=int, default=1, help="upscale factor for loose PNG frames (default 1)")
    g.add_argument("--bg", default=None, metavar="#RRGGBB", help="opaque GIF background instead of transparency")
    g.add_argument("--padding", type=int, default=0, help="pixels between sprite sheet cells (default 0)")
    g.add_argument("--godot-dir", default="res://", help="res:// folder the Godot .tres points at (default res://)")
    g.set_defaults(func=cmd_generate)

    pv = sub.add_parser("preview", help="open a live aquarium window (needs pygame-ce)")
    _add_render_args(pv)
    pv.add_argument("--fish", type=int, default=1, help="number of fish in the tank (default 1)")
    pv.add_argument("--zoom", type=int, default=4, help="pixel zoom (default 4)")
    pv.add_argument("--out", default="out", help="where the E key exports to (default ./out)")
    pv.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["gui"])
    if args.species not in (None, RANDOM) and args.kind not in (None, kind_of(args.species)):
        parser.error(f"--species {args.species} is a {kind_of(args.species)}, not a {args.kind}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
