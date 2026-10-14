"""3D fish model: a lofted body tube plus flat fins, deformed along a bendable spine.

Rest ("local") space uses three coordinates:
    a  distance along the spine, 0 = nose, 1 = base of the tail (body length = 1)
    y  up
    z  sideways (out of the fish's right flank, towards the viewer when facing right)

A Pose bends the spine (swimming wave) and points it along a heading (yaw).
Turns are follow-the-leader: the head swings round first and each point along
the spine follows the same path a moment later, so the body folds into a U
that travels down to the tail instead of the fish spinning like a rigid model. The posed mesh is then projected orthographically
with a slight downward camera pitch, so lateral motion reads in a side view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .genome import FinGenes, FishGenome

# Part ids (stored per triangle; 0 means "empty pixel").
BODY = 1
DORSAL = 2
ANAL = 3
CAUDAL = 4
PELVIC_NEAR = 5
PELVIC_FAR = 6
PECTORAL_NEAR = 7
PECTORAL_FAR = 8
FIN_PARTS = (DORSAL, ANAL, CAUDAL, PELVIC_NEAR, PELVIC_FAR, PECTORAL_NEAR, PECTORAL_FAR)
FAR_PARTS = (PELVIC_FAR, PECTORAL_FAR)

# Per-vertex attribute layout (after the posed normal is prepended by the renderer):
#   body: (a, sin t, cos t)   fins: (m along fin, r from root to edge, 0)
N_ATTRS = 3


@dataclass
class Pose:
    phase: float = 0.0  # swim cycle, 0..1 (wraps)
    yaw: float = 0.0  # head heading: 0 = facing right, pi = facing left, pi/2 = facing the viewer
    # Turning (follow-the-leader): progress 0..1 from turn_from to turn_to, or None when
    # not turning. ``yaw`` should then hold the head's heading (see turn_heading).
    turn: float | None = None
    turn_from: float = 0.0
    turn_to: float = 0.0
    turn_lag: float = 0.9  # how far the tail trails the head, as a fraction of the turn
    amplitude: float = 1.0  # multiplier on the swim wave
    pitch: float = 0.25  # camera tilt, radians (looking slightly down on the fish)


def turn_heading(progress: float, along, start: float, end: float, lag: float):
    """Heading of the spine point ``along`` (0 = nose, 1 = tail tip) at turn ``progress``.

    Each point runs the same eased heading change as the head, delayed by
    ``lag * along``, so the fold rolls from the head down to the tail.
    """
    s = np.clip(progress * (1.0 + lag) - lag * np.asarray(along, dtype=float), 0.0, 1.0)
    ease = s * s * (3 - 2 * s)  # smoothstep
    return start + (end - start) * ease


@dataclass
class Mesh:
    """Rest-space mesh. Vertices are (a, y, z)."""

    verts: np.ndarray  # (n, 3)
    attrs: np.ndarray  # (n, N_ATTRS)
    tris: np.ndarray  # (t, 3) int
    tri_part: np.ndarray  # (t,) int
    # Precomputed adjacency for normals: grid patches as (start_index, rows, cols, flip)
    patches: list[tuple[int, int, int, bool]]


@dataclass
class Eye:
    a: float
    y: float
    z: float  # positive = near (right) side
    radius: float


def _resample(points: list[tuple[float, float]], n: int) -> np.ndarray:
    """Resample a polyline to n points evenly spaced by arc length."""
    p = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-9:
        return np.repeat(p[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(targets, cum, p[:, 0]), np.interp(targets, cum, p[:, 1])], axis=1)


class FishModel:
    """Builds the rest mesh from a genome and poses it."""

    def __init__(self, genome: FishGenome, detail: int = 32):
        self.g = genome
        self.rings = max(24, int(detail * 1.5))
        self.ring_verts = 16
        self.fin_cols = max(8, detail // 3)
        self.fin_rows = 4
        self.tail_end = 1.0 + genome.tail_length * 1.35 + 0.05
        self._build()

    # ------------------------------------------------------------------ profile
    def height_profile(self, a: np.ndarray | float) -> np.ndarray:
        g = self.g
        a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
        front = a < g.peak
        u_f = np.clip(a / g.peak, 0, 1)
        f_front = (1.0 - (1.0 - u_f) ** 2) ** g.nose
        u_b = np.clip((a - g.peak) / (1.0 - g.peak), 0, 1)
        f_back = g.peduncle + (1.0 - g.peduncle) * np.cos(u_b * math.pi / 2) ** g.taper
        return np.maximum(np.where(front, f_front, f_back), 0.015)

    def top(self, a):
        return self.g.height * self.height_profile(a) * self.g.back_arch

    def bottom(self, a):
        return self.g.height * self.height_profile(a) * (1.0 - self.g.back_arch)

    def half_width(self, a):
        g = self.g
        a = np.asarray(a, dtype=float)
        squeeze = 1.0 - 0.55 * np.clip((a - g.peak) / (1.0 - g.peak), 0, 1)
        return 0.5 * g.height * g.thickness * self.height_profile(a) * squeeze

    # ------------------------------------------------------------------ build
    def _build(self):
        verts: list[np.ndarray] = []
        attrs: list[np.ndarray] = []
        tris: list[np.ndarray] = []
        parts: list[np.ndarray] = []
        patches: list[tuple[int, int, int, bool]] = []
        count = 0

        def add_grid(v: np.ndarray, at: np.ndarray, part: int, wrap: bool = False, flip: bool = False):
            nonlocal count
            rows, cols = v.shape[:2]
            idx = np.arange(rows * cols).reshape(rows, cols) + count
            if wrap:
                idx_next = np.roll(idx, -1, axis=1)
                q00, q01 = idx[:-1, :], idx_next[:-1, :]
                q10, q11 = idx[1:, :], idx_next[1:, :]
            else:
                q00, q01 = idx[:-1, :-1], idx[:-1, 1:]
                q10, q11 = idx[1:, :-1], idx[1:, 1:]
            t = np.concatenate([
                np.stack([q00, q10, q11], -1).reshape(-1, 3),
                np.stack([q00, q11, q01], -1).reshape(-1, 3),
            ])
            verts.append(v.reshape(-1, 3))
            attrs.append(at.reshape(-1, N_ATTRS))
            tris.append(t)
            parts.append(np.full(len(t), part))
            patches.append((count, rows, cols, flip))
            count += rows * cols

        g = self.g

        # Body tube: rings along a, vertices around angle t (t=pi/2 is the back).
        a = np.linspace(0.0, 1.0, self.rings) ** 1.15  # denser near the nose
        t = np.linspace(0.0, 2 * math.pi, self.ring_verts, endpoint=False)
        A, T = np.meshgrid(a, t, indexing="ij")
        sin_t, cos_t = np.sin(T), np.cos(T)
        Y = np.where(sin_t > 0, self.top(A), self.bottom(A)) * sin_t
        Z = self.half_width(A) * cos_t
        add_grid(np.stack([A, Y, Z], -1), np.stack([A, sin_t, cos_t], -1), BODY, wrap=True)

        # Dorsal / anal fins sit on the top / bottom line of the body.
        def median_fin(fg: FinGenes, part: int, sign: float):
            if not fg.present:
                return
            m = np.linspace(0.0, 1.0, self.fin_cols)
            prof = _fin_profile(fg.shape, m)
            ra = fg.start + (fg.end - fg.start) * m
            base = self.top(ra) if sign > 0 else self.bottom(ra)
            root_y = sign * base * 0.85
            h = fg.height * prof
            edge_a = ra + fg.sweep * h * 0.9
            edge_y = sign * (base + h)
            r = np.linspace(0.0, 1.0, self.fin_rows)
            R, M = np.meshgrid(r, m, indexing="ij")
            va = ra[None, :] + (edge_a - ra)[None, :] * R
            vy = root_y[None, :] + (edge_y - root_y)[None, :] * R
            add_grid(np.stack([va, vy, np.zeros_like(va)], -1), np.stack([M, R, np.zeros_like(R)], -1), part)

        median_fin(g.dorsal, DORSAL, 1.0)
        median_fin(g.anal, ANAL, -1.0)

        # Caudal fin: from the peduncle (root) out to a shape-specific edge.
        ped_top, ped_bot = float(self.top(1.0)), float(self.bottom(1.0))
        n = self.fin_cols + 4
        edge = _resample(_tail_outline(g, max(g.height, 0.3)), n)
        root = np.stack([np.full(n, 0.97), np.linspace(ped_top, -ped_bot, n) * 0.9], 1)
        r = np.linspace(0.0, 1.0, self.fin_rows + 1)
        R = r[:, None]
        va = root[None, :, 0] + (edge[None, :, 0] - root[None, :, 0]) * R
        vy = root[None, :, 1] + (edge[None, :, 1] - root[None, :, 1]) * R
        M = np.broadcast_to(np.linspace(0, 1, n)[None, :], va.shape)
        add_grid(np.stack([va, vy, np.zeros_like(va)], -1), np.stack([M, np.broadcast_to(R, va.shape), np.zeros_like(va)], -1), CAUDAL)

        # Paired fins hang off the flanks. Their shapes are stored in "fin space"
        # (u back along the body, v down) and placed per pose because they flap.
        self.paired: list[tuple[int, np.ndarray, np.ndarray, int]] = []

        def paired_fin(part: int, side: float, root_a: float, root_y: float, length: float, droop: float, splay: float):
            m = np.linspace(0.0, 1.0, self.fin_cols // 2 + 2)
            prof = np.sin(np.pi * m) ** 0.8
            # Root: short segment along the body; edge: rounded lobe swept back.
            ra = root_a + 0.04 * m
            edge_u = 0.04 * m + length * (0.35 + 0.65 * prof) * (0.6 + 0.4 * m)
            edge_v = droop * length * prof
            r = np.linspace(0.0, 1.0, self.fin_rows)
            R, M = np.meshgrid(r, m, indexing="ij")
            fin_u = (0.04 * M) + (edge_u[None, :] - 0.04 * M) * R
            fin_v = edge_v[None, :] * R
            rz = side * self.half_width(ra)[None, :] * 0.9 * np.ones_like(R)
            out = side * (splay + 0.6 * fin_v) * R * length
            local = np.stack([root_a + fin_u, root_y - fin_v * 0.5, rz], -1)
            self.paired.append((part, local.reshape(-1, 3), out.reshape(-1), count))
            add_grid(local + np.stack([0 * R, 0 * R, out * 0.9], -1), np.stack([M, R, np.zeros_like(R)], -1), part)

        pect_a = min(g.eye_pos + 0.16, g.peak)
        pect_y = -float(self.bottom(pect_a)) * 0.35
        pl = g.pectoral_size
        paired_fin(PECTORAL_NEAR, 1.0, pect_a, pect_y, pl, 0.45, 0.35)
        paired_fin(PECTORAL_FAR, -1.0, pect_a, pect_y, pl, 0.45, 0.35)
        if g.pelvic.present:
            pa = g.pelvic.start
            py = -float(self.bottom(pa)) * 0.9
            ph = g.pelvic.height
            paired_fin(PELVIC_NEAR, 1.0, pa, py, ph * 1.4, 1.0, 0.15)
            paired_fin(PELVIC_FAR, -1.0, pa, py, ph * 1.4, 1.0, 0.15)

        self.mesh = Mesh(
            verts=np.concatenate(verts),
            attrs=np.concatenate(attrs),
            tris=np.concatenate(tris),
            tri_part=np.concatenate(parts),
            patches=patches,
        )

        ea = g.eye_pos
        ey = float(self.top(ea)) * g.eye_height
        ez = float(self.half_width(ea)) * math.sqrt(max(0.0, 1.0 - g.eye_height ** 2)) * 1.05
        er = g.eye_size * g.height * 0.5
        self.eyes = [Eye(ea, ey, ez, er), Eye(ea, ey, -ez, er)]

    # ------------------------------------------------------------------ posing
    def spine(self, pose: Pose):
        """Sample the bent spine. Returns (a_samples, positions (n,3), headings (n,))."""
        g = self.g
        a = np.linspace(0.0, self.tail_end, 160)
        amp = g.swim_amplitude * pose.amplitude
        cyc = 2 * math.pi * pose.phase
        env = np.clip((a - 0.15) / 0.85, 0.0, None) ** 1.4
        wave = amp * env * np.sin(cyc - 2 * math.pi * g.wave_count * a)
        # The caudal fin flexes on top of the spine wave, lagging slightly behind it.
        tail = np.clip((a - 1.0) / (self.tail_end - 1.0), 0.0, 1.0)
        wave += amp * 0.9 * tail * np.sin(cyc - 2 * math.pi * g.wave_count - 0.9)
        # The head yaws a little against the tail beat (as real fish do).
        head_wobble = 0.12 * amp * math.sin(cyc + math.pi * 0.85)
        if pose.turn is None:
            base = pose.yaw
        else:
            base = turn_heading(pose.turn, a / self.tail_end, pose.turn_from, pose.turn_to, pose.turn_lag)
        heading = base + head_wobble + wave
        da = np.diff(a)
        mid = 0.5 * (heading[1:] + heading[:-1])
        step = np.stack([np.cos(mid), np.zeros_like(mid), np.sin(mid)], -1) * da[:, None]
        pos = np.concatenate([np.zeros((1, 3)), -np.cumsum(step, axis=0)])
        return a, pos, heading

    def pose_points(self, local: np.ndarray, pose: Pose, spine=None) -> np.ndarray:
        """Map rest-space points (..., 3) to posed world space."""
        a_s, pos, heading = spine if spine is not None else self.spine(pose)
        a = local[..., 0]
        px = np.interp(a, a_s, pos[:, 0])
        pz = np.interp(a, a_s, pos[:, 2])
        h = np.interp(a, a_s, heading)
        # Points beyond the nose (a < 0) continue straight ahead.
        ahead = np.minimum(a, 0.0)
        px = px - ahead * np.cos(h)
        pz = pz - ahead * np.sin(h)
        lat_x, lat_z = -np.sin(h), np.cos(h)
        x = px + local[..., 2] * lat_x
        z = pz + local[..., 2] * lat_z
        return np.stack([x, local[..., 1], z], -1)

    def posed(self, pose: Pose):
        """Return posed vertices in camera space (x right, y up, z towards viewer),
        camera-space normals, and the camera-space eye centres + eye normals."""
        g = self.g
        spine = self.spine(pose)
        verts = self.mesh.verts.copy()

        # Flap the paired fins: rotate their outward offset around the root line.
        flap = 0.5 + 0.35 * math.sin(2 * math.pi * (pose.phase * 2 + 0.25))
        for part, local, out, start in self.paired:
            l = local.copy()
            k = flap if part in (PECTORAL_NEAR, PECTORAL_FAR) else 0.6
            l[:, 2] += out * (0.3 + k)
            verts[start:start + len(l)] = l

        world = self.pose_points(verts, pose, spine)

        # Recentre on the body's midpoint so the sprite stays anchored.
        centre_local = np.array([[0.45, 0.0, 0.0]])
        centre = self.pose_points(centre_local, pose, spine)[0]
        world = world - centre

        normals = self._normals(world)
        # Body normals must point outwards: compare with the direction from the spine.
        nb = self.rings * self.ring_verts
        axis = self.pose_points(np.stack([verts[:nb, 0], 0 * verts[:nb, 0], 0 * verts[:nb, 0]], -1), pose, spine) - centre
        outward = world[:nb] - axis
        flip = np.sign(np.sum(normals[:nb] * outward, axis=1, keepdims=True))
        normals[:nb] *= np.where(flip == 0, 1.0, flip)

        eye_local = np.array([[e.a, e.y, e.z] for e in self.eyes])
        eye_world = self.pose_points(eye_local, pose, spine) - centre
        eye_axis = self.pose_points(eye_local * np.array([1, 1, 1.5]), pose, spine) - centre - eye_world
        eye_axis /= np.linalg.norm(eye_axis, axis=1, keepdims=True) + 1e-9

        return (
            _camera(world, pose.pitch),
            _camera(normals, pose.pitch),
            _camera(eye_world, pose.pitch),
            _camera(eye_axis, pose.pitch),
        )

    def _normals(self, world: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(world)
        for i, (start, rows, cols, _flip) in enumerate(self.mesh.patches):
            p = world[start:start + rows * cols].reshape(rows, cols, 3)
            if i == 0:  # body tube: wraps around in the column direction
                da = np.gradient(p, axis=0)
                dt = np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)
                n = np.cross(dt, da)
            else:
                da = np.gradient(p, axis=0) if rows > 1 else np.zeros_like(p)
                dm = np.gradient(p, axis=1) if cols > 1 else np.zeros_like(p)
                n = np.cross(da, dm)
            normals[start:start + rows * cols] = n.reshape(-1, 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        return normals


def _camera(p: np.ndarray, pitch: float) -> np.ndarray:
    """World -> camera: tilt so we look slightly down onto the fish."""
    c, s = math.cos(pitch), math.sin(pitch)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([x, y * c - z * s, z * c + y * s], -1)


def _fin_profile(shape: str, m: np.ndarray) -> np.ndarray:
    if shape == "triangle":
        peak = 0.3
        return np.where(m < peak, m / peak, (1 - m) / (1 - peak)) ** 0.9
    if shape == "rounded":
        return np.sin(np.pi * m) ** 0.6
    if shape == "sail":
        return np.sin(np.pi * np.clip(m, 0, 1) ** 0.55) ** 0.7 * (0.6 + 0.6 * m)
    # long: low, even fin that runs the whole length
    return np.clip(np.sin(np.pi * m) * 3.0, 0, 1) * 0.6


def _tail_outline(g: FishGenome, h: float) -> list[tuple[float, float]]:
    """Outline of the caudal fin from its top root to its bottom root in (a, y)."""
    L = g.tail_length
    S = g.tail_spread * h * 0.5
    x0 = 0.97
    shape = g.tail_shape
    if shape == "forked":
        return [(x0, S * 0.25), (x0 + L, S), (x0 + L * 0.95, S * 0.75), (x0 + L * 0.45, 0.0),
                (x0 + L * 0.95, -S * 0.75), (x0 + L, -S), (x0, -S * 0.25)]
    if shape == "lunate":
        pts = [(x0, S * 0.25)]
        for k in np.linspace(0, 1, 9):
            ang = math.pi / 2 - k * math.pi
            pts.append((x0 + L * (0.35 + 0.65 * abs(math.sin(ang)) ** 1.5), S * math.sin(ang)))
        pts.append((x0, -S * 0.25))
        return pts
    if shape == "truncate":
        return [(x0, S * 0.3), (x0 + L * 0.9, S * 0.9), (x0 + L, S * 0.6), (x0 + L, -S * 0.6),
                (x0 + L * 0.9, -S * 0.9), (x0, -S * 0.3)]
    if shape == "pointed":
        return [(x0, S * 0.45), (x0 + L * 0.7, S * 0.35), (x0 + L * 1.35, 0.0), (x0 + L * 0.7, -S * 0.35), (x0, -S * 0.45)]
    # rounded / fan: a lobe
    spread, power, reach = (1.15, 0.6, 1.2) if shape == "fan" else (0.85, 0.8, 1.0)
    # Start and end at the peduncle so the lobe flares out of the body
    # (rather than sitting on a thin stalk).
    pts = [(x0, S * 0.3)]
    for k in np.linspace(0.08, 0.92, 13):
        ang = math.pi / 2 - k * math.pi
        c = max(0.0, math.cos(ang)) ** power
        pts.append((x0 + L * reach * (0.3 + 0.7 * c), S * spread * math.sin(ang)))
    pts.append((x0, -S * 0.3))
    return pts
