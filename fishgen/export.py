"""Exporters: animated GIFs, sprite sheet + JSON, Godot 4 SpriteFrames, loose PNG frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .animation import SHOWCASE_ORDER, FishSprite
from .geometry import FishModel

FORMATS = ("gif", "sheet", "godot", "frames")


def _upscale(img: Image.Image, scale: int) -> Image.Image:
    if scale == 1:
        return img
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def _parse_hex(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected a #rrggbb colour, got {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def save_gif(sprite: FishSprite, frames: list[np.ndarray], path: Path, fps: float,
             scale: int = 4, background: str | None = None) -> Path:
    """Write RGBA frames as a GIF (transparent unless a background is given).

    Pixel-art frames already share one small palette and go in exactly. Smooth frames are
    reduced to one adaptive 256-colour palette for the whole animation (no colour flicker
    between frames), and GIF's on/off transparency flattens their translucent fins: over the
    background if one is given, otherwise over a mid-water blue and then cut off.
    """
    bg = np.array(_parse_hex(background) if background else (30, 70, 105), dtype=np.float64)
    flat = []
    for f in frames:
        a = f[..., 3:4].astype(np.float64) / 255.0
        rgb = f[..., :3] * a + bg * (1 - a)
        flat.append(np.round(rgb).astype(np.uint8))
    h, w = frames[0].shape[:2]
    # One palette for the whole animation, from a mosaic of every frame's visible pixels.
    opaque = [im[f[..., 3] >= 128] for im, f in zip(flat, frames)]
    pool = np.concatenate([o for o in opaque if len(o)] or [np.zeros((1, 3), np.uint8)])
    if len(pool) > 200_000:
        pool = pool[np.random.default_rng(0).choice(len(pool), 200_000, replace=False)]
    side = int(np.ceil(np.sqrt(len(pool))))
    mosaic = np.zeros((side * side, 3), np.uint8)
    mosaic[:len(pool)] = pool
    mosaic[len(pool):] = pool[0]
    pal_img = Image.fromarray(mosaic.reshape(side, side, 3)).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = []
    for im, f in zip(flat, frames):
        q = np.array(Image.fromarray(im).quantize(palette=pal_img, dither=Image.Dither.NONE))
        q = q + 1  # index 0 is kept for transparency
        if background:
            q[f[..., 3] < 8] = 1 + np.array(Image.fromarray(np.full((1, 1, 3), bg, np.uint8))
                                          .quantize(palette=pal_img, dither=Image.Dither.NONE))[0, 0]
        else:
            q[f[..., 3] < 128] = 0
        p = Image.fromarray(q.astype(np.uint8), "P")
        pal = [0, 0, 0] + pal_img.getpalette()[:255 * 3]
        if background:
            pal[0:3] = list(bg.astype(int))
        p.putpalette(pal)
        images.append(_upscale(p, scale))
    kwargs = dict(save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
                  disposal=2, optimize=False)
    if not background:
        kwargs["transparency"] = 0
    images[0].save(path, **kwargs)
    return path


def export_gifs(sprite: FishSprite, out_dir: Path, scale: int = 4, background: str | None = None) -> list[Path]:
    gif_dir = out_dir / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, anim in sprite.animations.items():
        paths.append(save_gif(sprite, anim.frames, gif_dir / f"{name}.gif", anim.fps, scale, background))
    fps = sprite.animations["swim_right"].fps
    paths.append(save_gif(sprite, sprite.showcase(), gif_dir / "showcase.gif", fps, scale, background))
    return paths


def build_sheet(sprite: FishSprite, padding: int = 0):
    """Lay out one row per animation. Returns (RGBA image, {name: [(x, y, w, h), ...]})."""
    w, h = sprite.size
    anims = [sprite.animations[n] for n in SHOWCASE_ORDER if n in sprite.animations]
    cols = max(len(a.frames) for a in anims)
    sheet_w = cols * (w + padding) + padding
    sheet_h = len(anims) * (h + padding) + padding
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    rects: dict[str, list[tuple[int, int, int, int]]] = {}
    for row, anim in enumerate(anims):
        y = padding + row * (h + padding)
        rects[anim.name] = []
        for col, frame in enumerate(anim.frames):
            x = padding + col * (w + padding)
            sheet[y:y + h, x:x + w] = frame
            rects[anim.name].append((x, y, w, h))
    return Image.fromarray(sheet), rects


def export_sheet(sprite: FishSprite, out_dir: Path, stem: str, padding: int = 0) -> tuple[Path, Path, dict]:
    """Sprite sheet PNG + engine-agnostic JSON describing every animation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img, rects = build_sheet(sprite, padding)
    png = out_dir / f"{stem}.png"
    img.save(png)
    w, h = sprite.size
    meta = {
        "generator": "fishgen",
        "version": __version__,
        "seed": sprite.genome.seed,
        "image": png.name,
        "image_size": [img.width, img.height],
        "frame_size": [w, h],
        "pivot": [w / 2, h / 2],
        "length_cm": round(sprite.genome.length_cm, 1),
        "px_per_m": round(sprite.renderer.scale * FishModel(sprite.genome, detail=24).real_extent()
                          / sprite.genome.length_cm * 100.0, 2),
        "padding": padding,
        "style": sprite.style,
        "palette": sprite.renderer.palette.to_dict(),
    }
    if sprite.palette is not None:  # pixel art: every colour the frames use
        meta["colors"] = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in sprite.palette]
    animations = {}
    for name, anim in sprite.animations.items():
        animations[name] = {
            "fps": anim.fps,
            "loop": anim.loop,
            # Play this when a non-looping animation finishes (or to chain loops).
            "next": anim.next,
            # yaw: the heading of this frame in radians (0 faces right); move the animal by
            # cos(yaw) * speed so it slows into a turn and speeds out of it.
            "frames": [
                {"x": x, "y": y, "w": fw, "h": fh, "duration_ms": anim.frame_ms,
                 "yaw": round(float(pose.yaw), 4)}
                for (x, y, fw, fh), pose in zip(rects[name], anim.poses)
            ],
        }
    data = {"meta": meta, "animations": animations, "genome": sprite.genome.to_dict()}
    js = out_dir / f"{stem}.json"
    js.write_text(json.dumps(data, indent=2))
    return png, js, rects


def export_godot(sprite: FishSprite, out_dir: Path, stem: str, rects: dict, res_dir: str = "res://") -> Path:
    """Godot 4 SpriteFrames resource referencing the sprite sheet PNG.

    Copy both ``<stem>.png`` and ``<stem>.tres`` into ``res_dir`` in your project and
    assign the .tres to an AnimatedSprite2D.
    """
    res_dir = res_dir if res_dir.endswith("/") else res_dir + "/"
    lines_sub: list[str] = []
    anim_blocks: list[str] = []
    sub_id = 0
    for name, anim in sprite.animations.items():
        frame_entries = []
        for (x, y, w, h) in rects[name]:
            sub_id += 1
            lines_sub.append(
                f'[sub_resource type="AtlasTexture" id="AtlasTexture_{sub_id}"]\n'
                f'atlas = ExtResource("1_sheet")\n'
                f"region = Rect2({x}, {y}, {w}, {h})\n"
            )
            frame_entries.append(f'{{\n"duration": 1.0,\n"texture": SubResource("AtlasTexture_{sub_id}")\n}}')
        anim_blocks.append(
            "{\n"
            f'"frames": [{", ".join(frame_entries)}],\n'
            f'"loop": {"true" if anim.loop else "false"},\n'
            f'"name": &"{name}",\n'
            f'"speed": {float(anim.fps)}\n'
            "}"
        )
    header = f'[gd_resource type="SpriteFrames" load_steps={sub_id + 2} format=3]\n'
    ext = f'[ext_resource type="Texture2D" path="{res_dir}{stem}.png" id="1_sheet"]\n'
    body = "[resource]\nanimations = [" + ", ".join(anim_blocks) + "]\n"
    path = out_dir / f"{stem}.tres"
    path.write_text("\n".join([header, ext, *lines_sub, body]))
    return path


def export_frames(sprite: FishSprite, out_dir: Path, scale: int = 1) -> list[Path]:
    paths = []
    for name, anim in sprite.animations.items():
        d = out_dir / "frames" / name
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(anim.frames):
            p = d / f"{name}_{i:02d}.png"
            _upscale(Image.fromarray(f), scale).save(p)
            paths.append(p)
    return paths


def export_all(sprite: FishSprite, out_dir: Path, formats: list[str], gif_scale: int = 4,
               frame_scale: int = 1, background: str | None = None, padding: int = 0,
               godot_dir: str = "res://", stem: str | None = None) -> list[Path]:
    """Export a fish in the requested formats into ``out_dir``. Returns written files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{sprite.genome.kind}_{sprite.genome.seed}"
    written: list[Path] = []
    rects = None
    if "sheet" in formats or "godot" in formats:
        png, js, rects = export_sheet(sprite, out_dir, stem, padding)
        written += [png, js]
    if "godot" in formats:
        written.append(export_godot(sprite, out_dir, stem, rects, godot_dir))
    if "gif" in formats:
        written += export_gifs(sprite, out_dir, gif_scale, background)
    if "frames" in formats:
        written += export_frames(sprite, out_dir, frame_scale)
    return written
