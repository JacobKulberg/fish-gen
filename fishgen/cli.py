"""Command line interface: ``fishgen gui``, ``fishgen generate`` and ``fishgen preview``."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from .animation import DEFAULT_PX_PER_M, MAX_PX, MIN_PX, STYLES, generate
from .export import FORMATS, export_all
from .genome import KINDS
from .species import RANDOM, SPECIES_NAMES, genome_for, kind_of, make_genome, morphs_of

DEFAULT_SIZE = 32  # sprites are N x N; drawing animals to scale is opt-in (--to-scale)


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
    if min(size) < 8 or max(size) > 2048:
        raise argparse.ArgumentTypeError("size must be between 8 and 2048 pixels")
    return size


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def parse_species(text: str) -> str:
    """Species names ignore case, spaces, hyphens and underscores: 'neon-tetra' -> 'Neon tetra'.
    A unique prefix is enough too ('hammer' -> 'Hammerhead shark')."""
    want = _norm(text)
    exact = [n for n in SPECIES_NAMES if _norm(n) == want]
    if exact:
        return exact[0]
    near = [n for n in SPECIES_NAMES if _norm(n).startswith(want) or want and want in _norm(n)]
    if len(near) == 1:
        return near[0]
    hint = f"did you mean {', '.join(near)}?" if near else f"choose from {', '.join(SPECIES_NAMES)}"
    raise argparse.ArgumentTypeError(f"unknown species {text!r}; {hint}")


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
    p.add_argument("--species", default=None, type=parse_species, metavar="NAME",
                   help=f"preset (implies its kind; case and spaces don't matter): {', '.join(SPECIES_NAMES)}")
    p.add_argument("--morph", default=None, metavar="NAME",
                   help="colour variety of the species, e.g. --species Koi --morph Showa (default: from the seed)")
    p.add_argument("--size", type=parse_size, default=(DEFAULT_SIZE, DEFAULT_SIZE), metavar="N|WxH",
                   help=f"sprite canvas, e.g. 16, 32, 64, 96, 128 or 48x32; the animal is fitted into it "
                        f"(default {DEFAULT_SIZE})")
    p.add_argument("--to-scale", action="store_true",
                   help="instead of a fixed canvas, draw animals to scale with each other (see --px-per-m)")
    p.add_argument("--px-per-m", type=float, default=DEFAULT_PX_PER_M, metavar="PX",
                   help=f"to scale: pixels per metre of animal (default {DEFAULT_PX_PER_M:g})")
    p.add_argument("--min-px", type=int, default=MIN_PX,
                   help=f"to scale: the smallest animals are drawn this long (default {MIN_PX})")
    p.add_argument("--max-px", type=int, default=MAX_PX,
                   help=f"to scale: the biggest animals are shrunk to fit a sprite this big (default {MAX_PX})")
    p.add_argument("--style", default="pixel", choices=STYLES,
                   help="pixel: pixel art with a limited palette and an outline (default); "
                        "smooth: anti-aliased full-colour renders")
    p.add_argument("--colors", type=int, default=None, metavar="N",
                   help="pixel art: palette size (default: 10 at 16px up to 24 at 128px)")
    p.add_argument("--no-outline", dest="outline", action="store_false", help="pixel art: no dark outline")
    p.add_argument("--swim-frames", type=int, default=8, help="frames per swim cycle (default 8)")
    p.add_argument("--turn-frames", type=int, default=10, help="frames per turn-around (default 10)")
    p.add_argument("--fps", type=float, default=10.0, help="playback speed (default 10)")
    p.add_argument("--pitch", type=float, default=0.25, help="camera tilt in radians (default 0.25)")


def _add_export_args(p: argparse.ArgumentParser, formats: tuple[str, ...] = FORMATS):
    p.add_argument("--count", type=int, default=1, help="how many animals (consecutive seeds)")
    p.add_argument("--formats", type=parse_formats, default=list(formats),
                   help=f"comma list of {','.join(FORMATS)} or 'all' (default {','.join(formats)})")
    p.add_argument("--out", default="out", help="output directory (default ./out)")
    p.add_argument("--flat", action="store_true", help="with a single animal, write straight into --out")
    p.add_argument("--scale", type=int, default=4, help="GIF upscale factor (default 4)")
    p.add_argument("--frame-scale", type=int, default=1, help="upscale factor for loose PNG frames (default 1)")
    p.add_argument("--bg", default=None, metavar="#RRGGBB", help="opaque GIF background instead of transparency")
    p.add_argument("--padding", type=int, default=0, help="pixels between sprite sheet cells (default 0)")
    p.add_argument("--godot-dir", default="res://", help="res:// folder the Godot .tres points at (default res://)")


def _genome(args, seed: int):
    if args.morph:
        if args.species in (None, RANDOM):
            raise SystemExit("--morph needs --species")
        if args.morph not in morphs_of(args.species):
            known = ", ".join(morphs_of(args.species)) or "no varieties"
            raise SystemExit(f"{args.species} has {known}, not {args.morph!r}")
        return make_genome(args.species, seed, morph=args.morph)
    return genome_for(seed, args.species, args.kind)


def render_opts(args) -> dict:
    fixed = None if args.to_scale else args.size
    if args.colors is not None and not 2 <= args.colors <= 256:
        raise SystemExit("--colors must be between 2 and 256")
    opts = dict(swim_frames=args.swim_frames, turn_frames=args.turn_frames, fps=args.fps, pitch=args.pitch,
                scale_by_size=fixed is None, px_per_m=args.px_per_m, min_px=args.min_px, max_px=args.max_px,
                style=args.style, colors=args.colors, outline=args.outline)
    if fixed:
        opts["width"], opts["height"] = fixed
    return opts


def _stem(args, genome, seed: int) -> str:
    """out/neon_tetra_32px_7 for a species, out/fish_32px_7 for a random animal."""
    base = _norm_stem(args.species) if args.species not in (None, RANDOM) else genome.kind
    size = "scale" if args.to_scale else (f"{args.size[0]}px" if args.size[0] == args.size[1]
                                          else f"{args.size[0]}x{args.size[1]}")
    return f"{base}_{size}_{seed}"


def _norm_stem(name: str) -> str:
    return "_".join("".join(ch if ch.isalnum() else " " for ch in name.lower()).split())


def cmd_generate(args) -> int:
    first = args.seed if args.seed is not None else random.randrange(1_000_000)
    seeds = [first + i for i in range(args.count)]
    out_root = Path(args.out)
    for seed in seeds:
        t0 = time.perf_counter()
        genome = _genome(args, seed)
        sprite = generate(seed, genome=genome, **render_opts(args))
        stem = _stem(args, genome, seed)
        out_dir = out_root / stem if args.count > 1 or not args.flat else out_root
        files = export_all(sprite, out_dir, args.formats, gif_scale=args.scale, frame_scale=args.frame_scale,
                           background=args.bg, padding=args.padding, godot_dir=args.godot_dir, stem=stem)
        w, h = sprite.size
        what = f"{genome.morph + ' ' if genome.morph else ''}{genome.kind}, {genome.length_cm:.0f} cm, {w}x{h}px"
        name = args.species if args.species not in (None, RANDOM) else genome.kind
        print(f"{name} {seed} ({what}): {len(files)} files -> {out_dir}  ({time.perf_counter() - t0:.2f}s)")
    return 0


def cmd_sprite(args) -> int:
    """``fishgen sprite "neon tetra" 32``: the quick way to one N x N sprite of one species."""
    args.species = args.name
    args.size = args.pixels or args.size
    return cmd_generate(args)


def cmd_preview(args) -> int:
    try:
        from .preview import run
    except ImportError as e:  # pragma: no cover - depends on optional dep
        print(f"preview needs pygame-ce: pip install 'fishgen[preview]'  ({e})", file=sys.stderr)
        return 1
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    run(seed=seed, count=args.fish, zoom=args.zoom, out=Path(args.out), species=args.species, kind=args.kind,
        opts=render_opts(args))
    return 0


def cmd_gui(args) -> int:
    from .gui import run
    from .species import RANDOM

    species = args.species or RANDOM
    kind = args.kind or (kind_of(species) if species != RANDOM else "fish")
    return run(seed=args.seed, species=species, out=Path(args.out), kind=kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fishgen", description="Procedural pixel-art sea-creature sprite generator.")
    sub = parser.add_subparsers(dest="command")

    gu = sub.add_parser("gui", help="open the fish designer window (the default with no command)")
    gu.add_argument("--seed", type=int, default=None, help="starting seed (random if omitted)")
    gu.add_argument("--kind", default=None, choices=KINDS, help="starting kind of animal")
    gu.add_argument("--species", default=None, type=parse_species, metavar="NAME", help="starting preset")
    gu.add_argument("--out", default="out", help="default export folder (default ./out)")
    gu.set_defaults(func=cmd_gui)

    g = sub.add_parser("generate", help="generate fish and export them")
    _add_render_args(g)
    _add_export_args(g)
    g.set_defaults(func=cmd_generate)

    sp = sub.add_parser("sprite", help="quick: one NxN sprite of a species, e.g. fishgen sprite 'neon tetra' 32")
    sp.add_argument("name", type=parse_species, metavar="SPECIES",
                    help="species preset; case, spaces and hyphens don't matter (neon-tetra, \"Great white\")")
    sp.add_argument("pixels", type=parse_size, nargs="?", default=None, metavar="SIZE",
                    help=f"canvas: 16, 32, 64, 96, 128 or WxH (default {DEFAULT_SIZE})")
    _add_render_args(sp)
    _add_export_args(sp, formats=("sheet", "gif"))
    sp.set_defaults(func=cmd_sprite)

    pv = sub.add_parser("preview", help="open a live aquarium window (needs pygame-ce)")
    _add_render_args(pv)
    pv.add_argument("--fish", type=int, default=1, help="number of fish in the tank (default 1)")
    pv.add_argument("--zoom", type=float, default=4.0, help="zoom (default 4)")
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
