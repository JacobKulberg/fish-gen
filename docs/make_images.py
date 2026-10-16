"""Regenerate the README images in docs/images/.

    python docs/make_images.py            # all of them
    python docs/make_images.py species    # just some: hero showcase turn scale rays random species sheet

Everything is rendered at smoothness 1 (8 swim / 10 turn frames at 10 fps) to keep the GIFs small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fishgen.animation import generate
from fishgen.species import SPECIES_NAMES, RANDOM, make_genome

OUT = Path(__file__).parent / "images"
TOP, BOTTOM = np.array([36, 84, 122]), np.array([12, 32, 56])
FONT = ImageFont.load_default(size=11)
OPTS = dict(swim_frames=8, turn_frames=10, smoothness=1)


def backdrop(w: int, h: int) -> Image.Image:
    """The water: a vertical gradient, lighter towards the surface."""
    t = np.linspace(0.0, 1.0, h)[:, None]
    rows = (TOP * (1 - t) + BOTTOM * t).round().astype(np.uint8)
    return Image.fromarray(np.repeat(rows[:, None, :], w, axis=1), "RGB").convert("RGBA")


def big(frame: np.ndarray, scale: int) -> Image.Image:
    im = Image.fromarray(frame)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def save_gif(frames: list[Image.Image], path: Path, ms: int = 100):
    frames = [f.convert("RGB") for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(path)


# The colour variety each picture shows, where seed 1 would pick a rarer one.
MORPHS = {"Clownfish": "Ocellaris", "Koi": "Kohaku"}


def sprite(name: str, w: int, h: int, seed: int = 1, **kw):
    genome = make_genome(name, seed, morph=MORPHS.get(name))
    return generate(seed, w, h, genome=genome, **{**OPTS, **kw})


def grid_gif(sprites, cols: int, cell: tuple[int, int], scale: int, frames_of, path: Path, count: int):
    """An animated grid: each sprite's own frames (cycled) centred in its cell."""
    cw, ch = cell
    rows = (len(sprites) + cols - 1) // cols
    seqs = [frames_of(s) for s in sprites]
    out = []
    for i in range(count):
        im = backdrop(cols * cw, rows * ch)
        for k, seq in enumerate(seqs):
            f = big(seq[i % len(seq)], scale)
            im.alpha_composite(f, ((k % cols) * cw + (cw - f.width) // 2, (k // cols) * ch + (ch - f.height) // 2))
        out.append(im)
    save_gif(out, path)


def hero():
    names = ["Clownfish", "Great white shark", "Manta ray", "Orca", "Koi", "Blue tang", "Octopus", "Tuna",
             "Spotted eagle ray", "Bottlenose dolphin", "Pufferfish", "Hammerhead shark"]
    sprites = [sprite(n, 64, 44) for n in names]
    grid_gif(sprites, 4, (192, 132), 3, lambda s: s.showcase(), OUT / "hero.gif", 52)


def showcase():
    s = sprite("Koi", 80, 60)
    save_gif([_on(backdrop(320, 240), big(f, 4)) for f in s.showcase()], OUT / "showcase.gif")


def _on(bg: Image.Image, fg: Image.Image) -> Image.Image:
    bg.alpha_composite(fg, ((bg.width - fg.width) // 2, (bg.height - fg.height) // 2))
    return bg


def turn():
    a = sprite("Great white shark", 80, 60).animations
    seq = a["swim_right"].frames + a["turn_right_to_left"].frames + a["swim_left"].frames
    save_gif([_on(backdrop(320, 240), big(f, 4)) for f in seq], OUT / "turn.gif")


def rays():
    sprites = [sprite(n, 64, 48) for n in ("Manta ray", "Spotted eagle ray", "Cownose ray")]
    grid_gif(sprites, 3, (256, 192), 4, lambda s: s.animations["swim_right"].frames, OUT / "rays.gif", 8)


def random_fish():
    sprites = [generate(seed, 40, 28, **OPTS) for seed in range(1, 25)]
    grid_gif(sprites, 6, (120, 84), 3, lambda s: s.animations["swim_right"].frames, OUT / "random_fish.gif", 8)


def species():
    names = [n for n in SPECIES_NAMES if n != RANDOM]
    cols, cw, ch = 8, 144, 114
    im = backdrop(cols * cw, ((len(names) + cols - 1) // cols) * ch)
    draw = ImageDraw.Draw(im)
    for k, n in enumerate(names):
        x, y = (k % cols) * cw, (k // cols) * ch
        f = big(sprite(n, 48, 32, swim_frames=2, turn_frames=2).animations["swim_right"].frames[0], 3)
        im.alpha_composite(f, (x + (cw - f.width) // 2, y + 16))
        draw.text((x + 6, y + 3), n, font=FONT, fill=(225, 235, 245))
    im.convert("RGB").save(OUT / "species.png")
    print(OUT / "species.png")


def scale():
    names = ["Humpback whale", "Orca", "Great white shark", "Manta ray", "Bottlenose dolphin", "Tuna", "Koi",
             "Clownfish", "Neon tetra"]
    z, gap, base = 3, 30, 225  # shown at 3x, sprites standing on one baseline
    sprites = [(n, generate(1, genome=make_genome(n, 1, morph=MORPHS.get(n)), scale_by_size=True, swim_frames=2, turn_frames=2,
                            smoothness=1)) for n in names]
    width = 10 + sum(s.size[0] * z + gap for _, s in sprites)
    im = backdrop(width, 316)
    draw = ImageDraw.Draw(im)
    draw.text((10, 8), "Drawn to scale with the default settings: 16 px per metre, sprites from 16 to 128 px "
                       "(shown at 3x)", font=FONT, fill=(225, 235, 245))
    x = 10
    last_label_end = -1
    for n, s in sprites:
        f = big(s.animations["swim_right"].frames[0], z)
        box = Image.fromarray(s.animations["swim_right"].frames[0]).getbbox() or (0, 0, *s.size)
        im.alpha_composite(f, (x, base - box[3] * z))
        g = s.genome
        size = f"{g.length_cm / 100:.2g} m" if g.length_cm >= 100 else f"{g.length_cm:.0f} cm"
        label = [n, f"{size}, {s.size[0]}x{s.size[1]}px"]
        ly = 240 if x > last_label_end else 270  # drop a label that would run into the last one
        for i, line in enumerate(label):
            draw.text((x, ly + 14 * i), line, font=FONT, fill=(225, 235, 245))
        if ly == 240:
            last_label_end = x + max(draw.textlength(t, font=FONT) for t in label) + 6
        x += f.width + gap
    im.convert("RGB").save(OUT / "scale.png")
    print(OUT / "scale.png")


def sheet():
    s = sprite("Clownfish", 32, 24)
    rows = list(s.animations.values())
    z, w, h = 3, 32 * 3 + 6, 24 * 3 + 6
    im = backdrop(max(len(a.frames) for a in rows) * w + 6, len(rows) * h + 6)
    for r, a in enumerate(rows):
        for c, f in enumerate(a.frames):
            im.alpha_composite(big(f, z), (6 + c * w, 6 + r * h))
    im.convert("RGB").save(OUT / "sheet.png")
    print(OUT / "sheet.png")


JOBS = {"hero": hero, "showcase": showcase, "turn": turn, "rays": rays, "random": random_fish,
        "species": species, "scale": scale, "sheet": sheet}

if __name__ == "__main__":
    for job in sys.argv[1:] or JOBS:
        JOBS[job]()
