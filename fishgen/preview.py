"""Live aquarium preview (pygame-ce).

Fish swim back and forth and play their turn-around when they reach a wall.
Turns are only triggered at the end of a swim cycle, which is exactly the
hand-off point the animations are designed for - the same rule a game should use.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pygame

from .animation import SHOWCASE_ORDER, FishSprite, generate
from .export import FORMATS, export_all

HELP = "SPACE new fish   </> seed -/+   T turn   G sheet   P pause   +/- zoom   E export   ESC quit"
WATER_TOP = (34, 86, 128)
WATER_BOTTOM = (10, 28, 52)


def _surface(frame, zoom: int) -> pygame.Surface:
    h, w = frame.shape[:2]
    surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGBA").convert_alpha()
    return pygame.transform.scale_by(surf, zoom) if zoom != 1 else surf


class TankFish:
    def __init__(self, sprite: FishSprite, zoom: int, bounds: pygame.Rect, rng: random.Random):
        self.sprite = sprite
        self.rng = rng
        self.set_zoom(zoom)
        w, h = self.frame_size
        self.x = rng.uniform(bounds.left + w, max(bounds.left + w + 1, bounds.right - w))
        self.base_y = rng.uniform(bounds.top + h, max(bounds.top + h + 1, bounds.bottom - h))
        self.bob = rng.uniform(0, math.tau)
        self.speed = rng.uniform(18, 34)  # sprite pixels per second
        self.anim = rng.choice(["swim_right", "swim_left"])
        self.frame = rng.randrange(len(self.sprite.animations[self.anim].frames))
        self.clock = 0.0
        self.turn_queued = False

    def set_zoom(self, zoom: int):
        self.zoom = zoom
        self.surfaces = {n: [_surface(f, zoom) for f in a.frames] for n, a in self.sprite.animations.items()}

    @property
    def frame_size(self) -> tuple[int, int]:
        w, h = self.sprite.size
        return w * self.zoom, h * self.zoom

    def update(self, dt: float, bounds: pygame.Rect):
        anim = self.sprite.animations[self.anim]
        self.clock += dt
        step = 1.0 / anim.fps
        while self.clock >= step:
            self.clock -= step
            self.frame += 1
            if self.frame >= len(anim.frames):
                self.frame = 0
                if not anim.loop:
                    self.anim = anim.next
                elif self._wants_turn(bounds):
                    self.anim = "turn_right_to_left" if self.anim == "swim_right" else "turn_left_to_right"
                    self.turn_queued = False
                anim = self.sprite.animations[self.anim]
        # Velocity follows where the fish is actually facing, so it slows into the
        # turn, stops while facing the viewer and accelerates out the other way.
        yaw = anim.poses[self.frame].yaw
        self.x += math.cos(yaw) * self.speed * self.zoom * dt
        self.bob += dt * 0.8

    def _wants_turn(self, bounds: pygame.Rect) -> bool:
        if self.turn_queued:
            return True
        w, _ = self.frame_size
        lookahead = self.speed * self.zoom * 0.6
        if self.anim == "swim_right" and self.x + w / 2 + lookahead > bounds.right:
            return True
        if self.anim == "swim_left" and self.x - w / 2 - lookahead < bounds.left:
            return True
        return self.rng.random() < 0.03

    def draw(self, screen: pygame.Surface):
        surf = self.surfaces[self.anim][self.frame]
        y = self.base_y + math.sin(self.bob) * 2 * self.zoom
        screen.blit(surf, surf.get_rect(center=(round(self.x), round(y))))


class Preview:
    def __init__(self, seed: int, size, count: int, zoom: int, swim_frames: int, turn_frames: int,
                 fps: float, pitch: float, out: Path):
        pygame.init()
        pygame.display.set_caption("fishgen preview")
        self.screen = pygame.display.set_mode((1100, 650), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.rng = random.Random(seed)
        self.seed = seed
        self.size = size
        self.count = count
        self.zoom = zoom
        self.opts = dict(swim_frames=swim_frames, turn_frames=turn_frames, fps=fps, pitch=pitch)
        self.out = out
        self.paused = False
        self.show_sheet = False
        self.message = ""
        self.message_t = 0.0
        self.bubbles = [self._bubble(random.uniform(0, 650)) for _ in range(40)]
        self.spawn()

    # ------------------------------------------------------------------
    def bounds(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(16, 40, w - 32, h - 80)

    def spawn(self):
        self.fish = []
        for i in range(self.count):
            sprite = generate(self.seed + i, *self.size, **self.opts)
            self.fish.append(TankFish(sprite, self.zoom, self.bounds(), self.rng))
        seeds = f"{self.seed}" if self.count == 1 else f"{self.seed}..{self.seed + self.count - 1}"
        pygame.display.set_caption(f"fishgen preview - seed {seeds}")

    def say(self, text: str):
        self.message, self.message_t = text, 3.0

    def _bubble(self, y: float):
        w, _ = self.screen.get_size()
        return [random.uniform(0, w), y, random.uniform(10, 30), random.choice([1, 1, 2])]

    # ------------------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key in (pygame.K_SPACE, pygame.K_n):
            self.seed = random.randrange(1_000_000)
            self.spawn()
        elif key == pygame.K_RIGHT:
            self.seed += 1
            self.spawn()
        elif key == pygame.K_LEFT:
            self.seed -= 1
            self.spawn()
        elif key == pygame.K_t:
            for f in self.fish:
                f.turn_queued = True
        elif key == pygame.K_g:
            self.show_sheet = not self.show_sheet
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.set_zoom(self.zoom + 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.set_zoom(max(1, self.zoom - 1))
        elif key == pygame.K_e:
            for f in self.fish:
                d = self.out / f"fish_{f.sprite.genome.seed}"
                export_all(f.sprite, d, list(FORMATS))
            self.say(f"exported {len(self.fish)} fish to {self.out.resolve()}")
        return True

    def set_zoom(self, zoom: int):
        self.zoom = zoom
        for f in self.fish:
            f.set_zoom(zoom)

    # ------------------------------------------------------------------
    def draw_background(self):
        w, h = self.screen.get_size()
        band = 8
        for y in range(0, h, band):
            t = y / max(1, h - 1)
            c = [round(a + (b - a) * t) for a, b in zip(WATER_TOP, WATER_BOTTOM)]
            self.screen.fill(c, (0, y, w, band))
        for b in self.bubbles:
            s = b[3] * 3
            pygame.draw.rect(self.screen, (120, 180, 215), (round(b[0]), round(b[1]), s, s), 1)

    def draw_sheet(self):
        """Overlay every animation of the first fish, all frames animating side by side."""
        f = self.fish[0]
        w, h = f.sprite.size
        sw, sh = self.screen.get_size()
        cols = max(len(a.frames) for a in f.sprite.animations.values()) + 1
        rows = len(f.sprite.animations)
        fit = min((sw - 60) // cols - 4, (sh - 140) // rows - 32)
        z = max(1, min(self.zoom, fit // max(w, h)))
        t = pygame.time.get_ticks() / 1000
        panel = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 170))
        self.screen.blit(panel, (0, 0))
        y = 50
        for name in SHOWCASE_ORDER:
            anim = f.sprite.animations[name]
            label = self.font.render(f"{name}  ({len(anim.frames)}f @ {anim.fps:g}fps -> {anim.next})", True, (230, 230, 230))
            self.screen.blit(label, (20, y))
            y += 20
            for i, frame in enumerate(anim.frames):
                surf = _surface(frame, z)
                x = 20 + i * (w * z + 4)
                pygame.draw.rect(self.screen, (40, 60, 80), (x, y, w * z, h * z))
                self.screen.blit(surf, (x, y))
            # Live loop of this animation at the end of the row.
            idx = int(t * anim.fps) % len(anim.frames)
            x = 20 + len(anim.frames) * (w * z + 4) + 16
            pygame.draw.rect(self.screen, (70, 90, 60), (x, y, w * z, h * z))
            self.screen.blit(_surface(anim.frames[idx], z), (x, y))
            y += h * z + 12

    def draw_hud(self, fps: float):
        seeds = ", ".join(str(f.sprite.genome.seed) for f in self.fish[:6])
        if len(self.fish) > 6:
            seeds += ", ..."
        top = self.font.render(f"seed {seeds}   size {self.size[0]}x{self.size[1]}   zoom {self.zoom}x   {fps:4.0f} fps",
                               True, (220, 235, 245))
        self.screen.blit(top, (12, 10))
        h = self.screen.get_height()
        self.screen.blit(self.font.render(HELP, True, (170, 200, 220)), (12, h - 26))
        if self.message_t > 0:
            msg = self.font.render(self.message, True, (255, 230, 140))
            self.screen.blit(msg, (12, h - 48))

    # ------------------------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = min(clock.tick(60) / 1000, 0.1)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    running = self.handle_key(ev.key)
            if not self.paused:
                bounds = self.bounds()
                for f in self.fish:
                    f.update(dt, bounds)
                for b in self.bubbles:
                    b[1] -= b[2] * dt
                    b[0] += math.sin(b[1] * 0.05) * 0.2
                    if b[1] < -10:
                        b[:] = self._bubble(self.screen.get_height() + 5)
            self.message_t -= dt

            self.draw_background()
            for f in sorted(self.fish, key=lambda f: f.base_y):
                f.draw(self.screen)
            if self.show_sheet:
                self.draw_sheet()
            self.draw_hud(clock.get_fps())
            pygame.display.flip()
        pygame.quit()


def run(seed: int, size=(32, 32), count: int = 1, zoom: int = 4, swim_frames: int = 8, turn_frames: int = 10,
        fps: float = 10.0, pitch: float = 0.25, out: Path = Path("out")):
    Preview(seed, size, count, zoom, swim_frames, turn_frames, fps, pitch, out).run()
