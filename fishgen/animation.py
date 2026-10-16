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
from dataclasses import dataclass, field

import numpy as np

from .genome import FishGenome
from .geometry import Pose, turn_heading
from .render import FishRenderer


@dataclass
class Animation:
    name: str
    poses: list[Pose]
    fps: float
    loop: bool
    next: str | None  # animation that should play after this one
    indexed: list[np.ndarray] = field(default_factory=list)  # palette-slot frames (h, w) uint8
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


def canvas_size(genome: FishGenome, width: int, height: int) -> tuple[int, int]:
    """Sprite canvas for ``genome``: ``width`` x ``height`` is the canvas of an animal of size 1,
    so bigger animals (sharks, whales) get bigger sprites and small fish smaller ones."""
    k = genome.size
    return (max(12, min(512, round(width * k))), max(12, min(512, round(height * k))))


@dataclass
class FishSprite:
    """A fully rendered fish: genome, palette and all animation frames."""

    genome: FishGenome
    renderer: FishRenderer
    animations: dict[str, Animation]

    @property
    def size(self) -> tuple[int, int]:
        return self.renderer.width, self.renderer.height

    def showcase(self, swim_loops: int = 2, indexed: bool = False) -> list[np.ndarray]:
        """A seamless loop demonstrating everything: swim, turn, swim back, turn."""
        a = self.animations
        seq: list[np.ndarray] = []
        for name in [n for n in SHOWCASE_ORDER if n in a]:
            loops = swim_loops if a[name].loop else 1
            seq += (a[name].indexed if indexed else a[name].frames) * loops
        return seq


def generate(seed: int, width: int = 32, height: int = 32, swim_frames: int = 8, turn_frames: int = 10,
             fps: float = 10.0, pitch: float = 0.25, genome: FishGenome | None = None,
             scale_by_size: bool = True) -> FishSprite:
    """Render every animation of a fish. ``width`` x ``height`` is the sprite size; with
    ``scale_by_size`` that is the size for an animal of ``genome.size`` 1 and the canvas
    grows or shrinks with the animal (see ``canvas_size``)."""
    genome = genome or FishGenome.random(seed)
    if scale_by_size:
        width, height = canvas_size(genome, width, height)
    # Some animals ask for a steeper camera (rays are seen from above) or a stiffer turn.
    anims = build_animations(swim_frames, turn_frames, fps, pitch + genome.camera_tilt, genome.turn_lag,
                             puff=genome.puff > 0)
    all_poses = [p for a in anims.values() for p in a.poses]
    renderer = FishRenderer(genome, width, height, fit_poses=all_poses)
    for anim in anims.values():
        anim.indexed = [renderer.render_indexed(p) for p in anim.poses]
        anim.frames = [renderer.to_rgba(f) for f in anim.indexed]
    return FishSprite(genome, renderer, anims)
