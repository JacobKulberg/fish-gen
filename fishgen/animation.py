"""Animation definitions: swim cycles and the turn-arounds that link them.

The swim cycle and the turns share one continuous tail-beat phase, so any
animation can hand off to the next one without a pop:

    swim_right -> turn_right_to_left -> swim_left -> turn_left_to_right -> swim_right

Pufferfish also get ``puff_right`` / ``puff_left``: it swells into a spiky ball, holds,
and deflates again, then carries on swimming the same way.

The last frame of each animation flows into the first frame of the one it
links to (see ``Animation.next``).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .genome import FishGenome
from .geometry import FishModel, Pose, turn_heading
from .pixel import pixelate
from .render import FishRenderer


@dataclass
class Animation:
    name: str
    poses: list[Pose]
    fps: float
    loop: bool
    next: str | None  # animation that should play after this one
    frames: list[np.ndarray] = field(default_factory=list)  # RGBA frames (h, w, 4) uint8

    @property
    def frame_ms(self) -> int:
        return round(1000 / self.fps)


def swim_poses(frames: int, yaw: float, pitch: float) -> list[Pose]:
    return [Pose(phase=i / frames, yaw=yaw, pitch=pitch) for i in range(frames)]


def turn_poses(frames: int, swim_frames: int, start_yaw: float, end_yaw: float, pitch: float,
               lag: float = 0.9) -> list[Pose]:
    """Turn through the viewer the way a fish does: the head swings round first
    and the body follows it, folding into a U that rolls down to the tail.

    The tail-beat phase keeps advancing at the swim rate and lands on a whole
    number of cycles, so the turn starts and ends in step with the swim loop.
    """
    cycles = max(1, round(frames / swim_frames))
    poses = []
    for j in range(frames):
        t = (j + 0.5) / frames
        mid = math.sin(math.pi * t)
        poses.append(Pose(
            phase=(j * cycles / frames) % 1.0,
            yaw=float(turn_heading(t, 0.0, start_yaw, end_yaw, lag)),
            turn=t,
            turn_from=start_yaw,
            turn_to=end_yaw,
            turn_lag=lag,
            # The tail beat eases off while the body folds.
            amplitude=1.0 - 0.6 * mid,
            pitch=pitch,
        ))
    return poses


def puff_poses(swim_frames: int, yaw: float, pitch: float, cycles: int = 3) -> list[Pose]:
    """Puff up, hold the spiky ball, deflate. Lasts whole swim cycles so it hands back to the swim seamlessly."""
    frames = swim_frames * cycles
    poses = []
    for j in range(frames):
        t = j / frames
        # Swell fast, hold, shrink back slowly (ends deflated, ready to swim on).
        up, down = min(1.0, t / 0.22), min(1.0, max(0.0, (1.0 - t) / 0.3))
        k = min(up, down)
        k = k * k * (3 - 2 * k)
        poses.append(Pose(phase=(j / swim_frames) % 1.0, yaw=yaw, pitch=pitch, inflate=k, amplitude=1.0 - 0.85 * k))
    return poses


SHOWCASE_ORDER = ("swim_right", "puff_right", "turn_right_to_left", "swim_left", "puff_left", "turn_left_to_right")


def build_animations(swim_frames: int = 8, turn_frames: int = 10, fps: float = 10.0,
                     pitch: float = 0.25, lag: float = 0.9, puff: bool = False) -> dict[str, Animation]:
    right, left = 0.0, math.pi
    anims = {
        "swim_right": Animation("swim_right", swim_poses(swim_frames, right, pitch), fps, True, "swim_right"),
        "turn_right_to_left": Animation(
            "turn_right_to_left", turn_poses(turn_frames, swim_frames, right, left, pitch, lag), fps, False, "swim_left"),
        "swim_left": Animation("swim_left", swim_poses(swim_frames, left, pitch), fps, True, "swim_left"),
        "turn_left_to_right": Animation(
            "turn_left_to_right", turn_poses(turn_frames, swim_frames, left, right, pitch, lag), fps, False, "swim_right"),
    }
    if puff:
        anims["puff_right"] = Animation("puff_right", puff_poses(swim_frames, right, pitch), fps, False, "swim_right")
        anims["puff_left"] = Animation("puff_left", puff_poses(swim_frames, left, pitch), fps, False, "swim_left")
    return anims


def smooth_timing(swim_frames: int, turn_frames: int, fps: float, smoothness: float) -> tuple[int, int, float]:
    """Frame counts and fps ``smoothness`` times as fluid at the same speed: 3 turns 8 swim
    frames, 10 turn frames at 10 fps into 24 and 30 at 30 fps. The fps follows the rounded swim
    frame count exactly, so a tail beat always lasts as long as it did."""
    if smoothness <= 0:
        raise ValueError(f"smoothness must be above 0, got {smoothness}")
    swim = max(2, round(swim_frames * smoothness))
    turn = max(2, round(turn_frames * smoothness))
    return swim, turn, fps * swim / swim_frames


DEFAULT_SIZE = 64  # sprites are N x N; drawing animals to scale is opt-in
DEFAULT_PX_PER_M = 16.0  # a 1 m animal is 16 px long, a 4.7 m great white 75 px
MIN_PX = 16  # the smallest animals are drawn at least this long, so there is something to see
MAX_PX = 128  # and the biggest (whales) get a sprite no bigger than this
STYLES = ("pixel", "smooth")


def px_per_unit(genome: FishGenome, px_per_m: float = DEFAULT_PX_PER_M, min_px: int = MIN_PX,
                max_px: int = MAX_PX) -> float:
    """Pixels per body length that draw ``genome`` at its real size: ``px_per_m`` pixels to a
    metre of ``genome.length_cm``, clamped so it measures ``min_px``..``max_px`` pixels."""
    extent = FishModel(genome, detail=24).real_extent()
    want = px_per_m * genome.length_cm / 100.0
    return float(np.clip(want, min_px, max_px)) / max(extent, 1e-6)


@dataclass
class FishSprite:
    """A fully rendered fish: genome, palette and all animation frames."""

    genome: FishGenome
    renderer: FishRenderer
    animations: dict[str, Animation]
    style: str = "pixel"
    palette: np.ndarray | None = None  # (n, 3) uint8: every colour the pixel-art frames use

    @property
    def size(self) -> tuple[int, int]:
        return self.renderer.width, self.renderer.height

    def showcase(self, swim_loops: int = 2) -> list[np.ndarray]:
        """A seamless loop demonstrating everything: swim, turn, swim back, turn."""
        a = self.animations
        seq: list[np.ndarray] = []
        for name in [n for n in SHOWCASE_ORDER if n in a]:
            loops = swim_loops if a[name].loop else 1
            seq += a[name].frames * loops
        return seq


def generate(seed: int, width: int = DEFAULT_SIZE, height: int = DEFAULT_SIZE, swim_frames: int = 8,
             turn_frames: int = 10, fps: float = 10.0, smoothness: float = 1.0, pitch: float = 0.25, genome: FishGenome | None = None,
             scale_by_size: bool = False, px_per_m: float = DEFAULT_PX_PER_M, min_px: int = MIN_PX,
             max_px: int = MAX_PX, style: str = "pixel", colors: int | None = None, outline: bool = True,
             threads: int | None = None) -> FishSprite:
    """Render every animation of an animal.

    By default the animal is fitted into a ``width`` x ``height`` canvas (64 x 64).

    ``smoothness`` multiplies the frame counts and fps together (see ``smooth_timing``): 3 is
    three times as fluid, swimming and turning at the same speed.

    With ``scale_by_size`` it is drawn at its real size instead, ``px_per_m`` pixels to the
    metre, on a canvas just big enough for every frame, so animals put side by side are to
    scale. Small animals are still drawn ``min_px`` long, and the biggest are shrunk to fit a
    ``max_px`` canvas.

    ``style`` "pixel" (the default) turns the renders into pixel art: hard edges, one palette
    of ``colors`` colours (None = chosen for the size) shared by every frame, and an outline.
    "smooth" keeps the anti-aliased full-colour renders.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")
    genome = genome or FishGenome.random(seed)
    swim_frames, turn_frames, fps = smooth_timing(swim_frames, turn_frames, fps, smoothness)
    # Some animals ask for a steeper camera or a stiffer turn.
    anims = build_animations(swim_frames, turn_frames, fps, pitch + genome.camera_tilt, genome.turn_lag,
                             puff=genome.puff > 0)
    all_poses = [p for a in anims.values() for p in a.poses]
    # Pixel art keeps a clear ring round the animal for its outline.
    margin = 2.0 if style == "pixel" and outline else 1.0
    if scale_by_size:
        renderer = FishRenderer(genome, fit_poses=all_poses, margin=margin, max_size=max_px,
                                px_per_unit=px_per_unit(genome, px_per_m, min_px, max_px))
    else:
        renderer = FishRenderer(genome, width, height, fit_poses=all_poses, margin=margin)
    # Frames are independent, and the heavy lifting (C code and numpy) runs outside the GIL.
    workers = threads or min(8, os.cpu_count() or 1)
    if workers > 1 and len(all_poses) > 1:
        with ThreadPoolExecutor(workers) as ex:
            done = list(ex.map(renderer.render_with_eyes, all_poses))
    else:
        done = [renderer.render_with_eyes(p) for p in all_poses]
    frames = [f for f, _ in done]
    palette = None
    if style == "pixel":
        frames, palette = pixelate(frames, colors, outline, eyes=[e for _, e in done])
    i = 0
    for anim in anims.values():
        anim.frames = frames[i:i + len(anim.poses)]
        i += len(anim.poses)
    return FishSprite(genome, renderer, anims, style, palette)
