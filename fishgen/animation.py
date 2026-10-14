"""Animation definitions: swim cycles and the turn-arounds that link them.

The swim cycle and the turns share one continuous tail-beat phase, so any
animation can hand off to the next one without a pop:

    swim_right -> turn_right_to_left -> swim_left -> turn_left_to_right -> swim_right

The last frame of each animation flows into the first frame of the one it
links to (see ``Animation.next``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .genome import FishGenome
from .geometry import Pose
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
               bend: float = 0.9) -> list[Pose]:
    """Rotate through the viewer (the fish shows its face mid-turn).

    The tail-beat phase keeps advancing at the swim rate and lands on a whole
    number of cycles, so the turn starts and ends in step with the swim loop.
    """
    cycles = max(1, round(frames / swim_frames))
    direction = 1.0 if end_yaw > start_yaw else -1.0
    poses = []
    for j in range(frames):
        t = (j + 0.5) / frames
        # Slightly slower through the middle so the face-on frames get screen time.
        e = t + 0.25 * math.sin(2 * math.pi * t) / (2 * math.pi)
        mid = math.sin(math.pi * t)
        poses.append(Pose(
            phase=(j * cycles / frames) % 1.0,
            yaw=start_yaw + (end_yaw - start_yaw) * e,
            # Head leads, body lags behind: a C-curve that peaks mid-turn.
            bend=direction * bend * mid,
            amplitude=1.0 - 0.55 * mid,
            pitch=pitch,
        ))
    return poses


SHOWCASE_ORDER = ("swim_right", "turn_right_to_left", "swim_left", "turn_left_to_right")


def build_animations(swim_frames: int = 8, turn_frames: int = 10, fps: float = 10.0,
                     pitch: float = 0.25) -> dict[str, Animation]:
    right, left = 0.0, math.pi
    return {
        "swim_right": Animation("swim_right", swim_poses(swim_frames, right, pitch), fps, True, "swim_right"),
        "turn_right_to_left": Animation(
            "turn_right_to_left", turn_poses(turn_frames, swim_frames, right, left, pitch), fps, False, "swim_left"),
        "swim_left": Animation("swim_left", swim_poses(swim_frames, left, pitch), fps, True, "swim_left"),
        "turn_left_to_right": Animation(
            "turn_left_to_right", turn_poses(turn_frames, swim_frames, left, right, pitch), fps, False, "swim_right"),
    }


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
        for name in SHOWCASE_ORDER:
            loops = swim_loops if a[name].loop else 1
            seq += (a[name].indexed if indexed else a[name].frames) * loops
        return seq


def generate(seed: int, width: int = 32, height: int = 32, swim_frames: int = 8, turn_frames: int = 10,
             fps: float = 10.0, pitch: float = 0.25, genome: FishGenome | None = None) -> FishSprite:
    genome = genome or FishGenome.random(seed)
    anims = build_animations(swim_frames, turn_frames, fps, pitch)
    all_poses = [p for a in anims.values() for p in a.poses]
    renderer = FishRenderer(genome, width, height, fit_poses=all_poses)
    for anim in anims.values():
        anim.indexed = [renderer.render_indexed(p) for p in anim.poses]
        anim.frames = [renderer.to_rgba(f) for f in anim.indexed]
    return FishSprite(genome, renderer, anims)
