"""Turns posed 3D animals into realistic full-colour frames.

Each frame is rasterised at a supersampled resolution and averaged down, so edges and
markings are anti-aliased. Shading is done in linear light:

* albedo from the colour genes: countershading, pattern, scales, skin noise, scars,
  gill covers or slits, mouth, lateral line, fin rays, suckers...
* a sun from above with a soft shadow map, a hemisphere of water light (bright
  above, dark below), screen-space ambient occlusion where parts crowd each other,
  wet specular sheen, silvery mirror flanks that reflect the bright surface above,
  a cool rim of water light, and light shining through thin fins
* fins are translucent membranes, composited over whatever is behind them
* eyes with iris, pupil, socket shadow and a catch-light; urchin and pufferfish
  spines as shaded tapering cylinders
"""

from __future__ import annotations

import math
import threading
from typing import Iterable

import numpy as np

from . import native
from . import palette as P
from .geometry import (ANAL, ARM, BODY, CAUDAL, DORSAL, FAR_PARTS, FIN_PARTS, LOBE, PECTORAL_FAR, PECTORAL_NEAR,
                       PELVIC_FAR, PELVIC_NEAR, WING_PARTS, FishModel, Pose, _camera)
from .genome import FishGenome
from .raster import EMPTY_DEPTH, rasterize

SUN_WORLD = np.array([0.18, 1.0, 0.42]) / np.linalg.norm([0.18, 1.0, 0.42])
SUN = np.array([1.0, 0.96, 0.88]) * 0.95
SKY = np.array([0.3, 0.42, 0.5])  # water light from above
DEEP = np.array([0.035, 0.055, 0.075])  # from below
ENV_TOP = np.array([0.95, 1.0, 1.0])  # what a mirror-bright flank reflects: the surface...
ENV_BOTTOM = np.array([0.03, 0.08, 0.13])  # ...and the deep
RIM = np.array([0.35, 0.55, 0.7])
PAIRED = (PECTORAL_NEAR, PECTORAL_FAR, PELVIC_NEAR, PELVIC_FAR)


def _lin(rgb) -> np.ndarray:
    """sRGB 0..255 -> linear 0..1."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c: np.ndarray) -> np.ndarray:
    if native.lib is not None:
        out = native.f64(c).copy()
        native.lib.fg_srgb(native.ptr(out), out.size)
        return out
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _hex_lin(color: str | None, default) -> np.ndarray:
    return _lin(P.hex_rgb(color)) if color else np.asarray(default, dtype=np.float64)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _mix(a, b, t):
    t = np.asarray(t, dtype=np.float64)
    if t.ndim and np.ndim(a) and np.shape(a)[-1] == 3 and t.shape[-1:] != (3,):
        t = t[..., None]
    return a + (b - a) * t


def _native_call(fn, x, y, *args):
    """Run a C noise function elementwise over broadcast x, y."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    x, y = native.f64(x), native.f64(y)
    out = np.empty(x.shape)
    fn(native.ptr(x), native.ptr(y), x.size, *args, native.ptr(out))
    return out


def _hash(i, j, seed):
    if native.lib is not None:
        return _native_call(native.lib.fg_hash, i, j, float(seed))
    h = np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453
    return h - np.floor(h)


def vnoise(x, y, seed: float = 0.0):
    """Smooth value noise in -1..1."""
    if native.lib is not None:
        return _native_call(native.lib.fg_fbm, x, y, float(seed), 1)
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    u, v = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    a = _hash(xi, yi, seed)
    b = _hash(xi + 1, yi, seed)
    c = _hash(xi, yi + 1, seed)
    d = _hash(xi + 1, yi + 1, seed)
    return 2.0 * (a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v) - 1.0


def fbm(x, y, seed: float = 0.0, octaves: int = 4):
    if native.lib is not None:
        return _native_call(native.lib.fg_fbm, x, y, float(seed), int(octaves))
    total, amp, norm = 0.0, 1.0, 0.0
    for k in range(octaves):
        total = total + amp * vnoise(x, y, seed + 13.0 * k)
        norm += amp
        x, y, amp = x * 2.03 + 1.7, y * 2.03 - 0.9, amp * 0.5
    return total / norm


def _supersample(size: int) -> int:
    if size <= 128:
        return 4
    if size <= 320:
        return 3
    if size <= 900:
        return 2
    return 1


class FishRenderer:
    """Renders one animal at a fixed scale so every frame lines up.

    Give ``width`` x ``height`` to fit the animal into that canvas, or ``px_per_unit``
    (pixels per body length) to draw it at that scale on a canvas sized to fit it (and at
    most ``max_size`` pixels across).
    """

    def __init__(self, genome: FishGenome, width: int | None = 32, height: int | None = 32,
                 fit_poses: Iterable[Pose] = (), px_per_unit: float | None = None,
                 supersample: int | None = None, margin: float = 1.0, max_size: int | None = None):
        self.genome = genome
        self.margin = margin  # empty pixels kept round the animal (room for a pixel-art outline)
        poses = list(fit_poses) or [Pose(phase=p / 8) for p in range(8)]
        if px_per_unit is None:
            self.width, self.height = int(width), int(height)
            self.model = FishModel(genome, detail=max(self.width, self.height))
            self.scale = self._fit_scale(poses)
        else:
            ext_x, ext_y = self._extents(FishModel(genome, detail=32), poses)
            if max_size:  # shrink the biggest animals until their canvas fits max_size
                px_per_unit = min(px_per_unit, (max_size / 2 - margin) / max(ext_x, ext_y))
            self.width = max(4, 2 * math.ceil(ext_x * px_per_unit + margin))
            self.height = max(4, 2 * math.ceil(ext_y * px_per_unit + margin))
            self.model = FishModel(genome, detail=max(self.width, self.height))
            self.scale = min(px_per_unit, self._fit_scale(poses))
        self.ss = supersample or _supersample(max(self.width, self.height))
        self._tls = threading.local()  # per-frame state, so frames can render on several threads
        self.palette = P.build_palette(genome.colors)
        self._materials()

    @property
    def _up(self) -> np.ndarray:  # world up, in camera space, for the frame being rendered
        return self._tls.up

    @property
    def _sun(self) -> np.ndarray:  # direction to the sun, in camera space
        return self._tls.sun

    @staticmethod
    def _extents(model: FishModel, poses: list[Pose]) -> tuple[float, float]:
        ext_x = ext_y = 1e-6
        thick = _spine_radius(model.g)
        for pose in poses:
            cam, _, _, _, lines = model.posed(pose)
            pts = np.concatenate([cam, lines.reshape(-1, 3)])
            ext_x = max(ext_x, np.abs(pts[:, 0]).max() + (thick if len(lines) else 0.0))
            ext_y = max(ext_y, np.abs(pts[:, 1]).max() + (thick if len(lines) else 0.0))
        return ext_x, ext_y

    def _fit_scale(self, poses: list[Pose]) -> float:
        """Largest scale at which every pose fits inside the canvas, keeping the animal centred."""
        ext_x, ext_y = self._extents(self.model, poses)
        m = self.margin
        return max(1e-6, min((self.width / 2 - m) / ext_x, (self.height / 2 - m) / ext_y))

    # ------------------------------------------------------------------ colours
    def _materials(self):
        g = self.genome
        c = g.colors
        pal = self.palette.colors
        self.c_base = _lin(pal[P.BASE])
        self.c_belly = _lin(pal[P.BELLY])
        self.c_fin = _lin(pal[P.FIN])
        self.c_fin_dark = _lin(pal[P.FIN_DARK])
        self.c_accent = _lin(pal[P.ACCENT])
        self.c_accent2 = _lin(pal[P.ACCENT2])
        self.c_fin_edge = _hex_lin(c.fin_edge_color, self.c_fin_dark * 0.5)
        self.c_accent_edge = _hex_lin(c.accent_edge_color, self.c_accent * 0.4)
        kind = g.kind
        # Specular sharpness and how far light wraps round the body (soft, fleshy animals more).
        self.shininess = {"fish": 60.0, "shark": 28.0, "whale": 45.0, "ray": 30.0, "squid": 70.0,
                          "octopus": 40.0, "urchin": 20.0}[kind]
        self.wrap = {"fish": 0.25, "shark": 0.15, "whale": 0.2, "ray": 0.2, "squid": 0.45, "octopus": 0.4,
                     "urchin": 0.2}[kind]
        if g.iris_color:
            self.c_iris = _hex_lin(g.iris_color, 0)
        elif kind in ("squid", "octopus"):
            self.c_iris = _lin((205, 170, 90))
        elif g.eye_style == "dark":
            self.c_iris = _lin((38, 36, 32))
        else:
            self.c_iris = _lin((196, 170, 112))

    # ------------------------------------------------------------------ main
    def render(self, pose: Pose) -> np.ndarray:
        """Render a pose to an (h, w, 4) uint8 RGBA array."""
        return self.render_with_eyes(pose)[0]

    def render_with_eyes(self, pose: Pose) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
        """Render a pose; also returns each visible eye as (x, y, radius) in output pixels."""
        ss = self.ss
        W, H = self.width * ss, self.height * ss
        s = self.scale * ss
        g = self.genome
        model = self.model
        cam, normals, eyes, eye_axes, lines = model.posed(pose)
        cx, cy = W / 2, H / 2
        screen = np.stack([cx + cam[:, 0] * s, cy - cam[:, 1] * s, cam[:, 2] * s], -1)
        attrs = np.concatenate([normals, model.mesh.attrs], axis=1)
        mesh = model.mesh
        fin_tri = np.isin(mesh.tri_part, FIN_PARTS)

        # Pass 1: everything solid (body, wings, arms). Pass 2: the fins, laid over it.
        part, depth, attr = rasterize(screen, attrs, mesh.tris[~fin_tri], mesh.tri_part[~fin_tri], W, H)
        if fin_tri.any():
            fpart, fdepth, fattr = rasterize(screen, attrs, mesh.tris[fin_tri], mesh.tri_part[fin_tri], W, H)
            fpart[fdepth <= depth] = 0
        else:
            fpart = np.zeros_like(part)
            fdepth, fattr = depth, attr
        if ss == 1:
            part, fpart = _cleanup(part), _cleanup(fpart)

        self._tls.up = _camera(np.array([0.0, 1.0, 0.0]), pose.pitch)
        self._tls.sun = _camera(SUN_WORLD, pose.pitch)
        shadow = self._shadow_map(screen, cam, pose, s)

        rgb = np.zeros((H, W, 3))
        alpha = np.zeros((H, W))
        solid = part > 0
        if solid.any():
            ao = self._ssao(depth, solid, s)
            rgb[solid] = self._shade_solid(part, depth, attr, solid, ao, shadow, s, pose, W, H)
            alpha[solid] = 1.0
        fins = fpart > 0
        if fins.any():
            frgb, fa = self._shade_fins(fpart, fdepth, fattr, fins, shadow, s, W, H)
            rgb[fins] = frgb * fa[:, None] + rgb[fins] * (1 - fa[:, None])
            alpha[fins] = fa + alpha[fins] * (1 - fa)

        depth_all = np.where(fins, fdepth, depth)
        if len(lines):
            self._draw_spines(rgb, alpha, depth_all, lines, s, W, H)
        seen = self._draw_eyes(rgb, alpha, depth_all, eyes, eye_axes, pose, s, W, H)

        # Tone map (soft shoulder so highlights don't clip flat) and back to sRGB.
        knee = 0.8
        over = np.maximum(rgb - knee, 0.0)
        rgb = np.where(rgb > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), rgb)
        out = _srgb(rgb)
        if ss > 1:  # average each ss x ss block of samples (premultiplied by alpha)
            pre = (out * alpha[..., None]).reshape(self.height, ss, self.width, ss, 3).mean(axis=(1, 3))
            a = alpha.reshape(self.height, ss, self.width, ss).mean(axis=(1, 3))
            out = np.where(a[..., None] > 1e-6, pre / np.maximum(a[..., None], 1e-6), 0.0)
            alpha = a
        rgba = np.concatenate([out, alpha[..., None]], axis=-1)
        return np.round(np.clip(rgba, 0, 1) * 255).astype(np.uint8), [(x / ss, y / ss, r / ss) for x, y, r in seen]

    # ------------------------------------------------------------------ lighting helpers
    def _shadow_map(self, screen, cam, pose, s):
        """Depth of the animal seen from the sun, for soft shadows. Returns a lookup function."""
        L = self._sun
        u = np.cross([0.0, 0.0, 1.0], L)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(L, u)
        rot = np.stack([u, v, L])  # camera -> light space (z towards the sun)
        lp = cam @ rot.T * s
        lo = lp[:, :2].min(axis=0) - 2
        hi = lp[:, :2].max(axis=0) + 2
        res = 1.0 if self.ss <= 2 else 0.5  # shadow-map texels per supersample pixel
        Wl, Hl = int(math.ceil((hi[0] - lo[0]) * res)) + 1, int(math.ceil((hi[1] - lo[1]) * res)) + 1
        sp = np.stack([(lp[:, 0] - lo[0]) * res, (lp[:, 1] - lo[1]) * res, lp[:, 2] * res], -1)
        mesh = self.model.mesh
        # Only solid parts cast shadows: thin fins standing edge-on to the sun would shadow themselves.
        keep = ~np.isin(mesh.tri_part, FIN_PARTS)
        _, ldepth, _ = rasterize(sp, np.zeros((len(sp), 1)), mesh.tris[keep], mesh.tri_part[keep], Wl, Hl)
        soft = max(1.0, 0.012 * s * res)  # penumbra in texels: underwater light is diffuse

        def lookup(px, py, pz, bias):
            """Lit fraction (0..1) of camera-space points given in supersample pixels."""
            q = np.stack([px, py, pz], -1) @ rot.T
            qx, qy, qz = (q[:, 0] - lo[0]) * res, (q[:, 1] - lo[1]) * res, q[:, 2] * res
            lit = np.zeros(len(qx))
            taps = [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)]
            for ox, oy in taps:
                ix = np.clip(np.floor(qx + ox * soft).astype(int), 0, Wl - 1)
                iy = np.clip(np.floor(qy + oy * soft).astype(int), 0, Hl - 1)
                occ = ldepth[iy, ix]
                lit += (occ <= qz + bias * res + soft * 0.8) | (occ == EMPTY_DEPTH)
            return lit / len(taps)

        return lookup

    def _ssao(self, depth, mask, s):
        """Screen-space ambient occlusion: darken where nearer surfaces crowd round a pixel."""
        H, W = depth.shape
        if native.lib is not None:
            occ = np.empty((H, W))
            native.lib.fg_ssao(native.ptr(native.f64(depth)), H, W, float(s), native.ptr(occ))
            return occ
        occ = np.zeros((H, W))
        n = 0
        for rad in (max(2.0, 0.02 * s), max(4.0, 0.05 * s)):
            for k in range(8):
                ang = k * math.pi / 4 + rad
                dx, dy = int(round(math.cos(ang) * rad)), int(round(math.sin(ang) * rad))
                nd = _shift(depth, dy, dx, fill=EMPTY_DEPTH)
                diff = nd - depth
                occ += np.clip((diff - 0.3 * rad) / (1.5 * rad), 0.0, 1.0) * (diff < 6 * rad)
                n += 1
        return occ / n

    def _light(self, albedo, n, sun_lit, ao, gloss, silver, wrap, rim=1.0, back=None, transmit=0.0):
        """Shade linear albedo (k,3) with normals (k,3) facing the camera."""
        L, up = self._sun, self._up
        ndl = n @ L
        diff = np.clip((ndl + wrap) / (1 + wrap), 0.0, 1.0) * sun_lit
        hemi = 0.5 + 0.5 * (n @ up)
        amb = (DEEP + (SKY - DEEP) * hemi[:, None]) * (1.0 - 0.75 * ao)[:, None]
        col = albedo * (SUN * diff[:, None] + amb)
        # Wet sheen.
        h = L + np.array([0.0, 0.0, 1.0])
        h = h / np.linalg.norm(h)
        spec = np.clip(n @ h, 0.0, 1.0) ** self.shininess * sun_lit * gloss
        col = col + SUN * spec[:, None] * 0.9
        # Mirror-silver flanks reflect the bright surface above and the dark deep below.
        if np.any(silver > 0):
            nv = np.clip(n[:, 2], 0.0, 1.0)
            refl_up = 2 * nv * (n @ up) - up[2]
            env = ENV_BOTTOM + (ENV_TOP - ENV_BOTTOM) * smoothstep(-0.6, 0.6, refl_up)[:, None] ** 1.5
            env = env * (0.6 + 0.4 * hemi)[:, None] * (1.0 - 0.5 * ao)[:, None] * 0.85
            tint = 0.75 + 0.25 * albedo / (albedo.max(axis=1, keepdims=True) + 1e-6)
            col = col + (env * tint - col) * silver[:, None]
        # A cool rim of scattered water light round the silhouette.
        fres = (1.0 - np.clip(n[:, 2], 0.0, 1.0)) ** 4
        col = col + RIM * (fres * 0.22 * rim)[:, None] * (0.4 + 0.6 * hemi)[:, None]
        if back is not None:  # light coming through a thin membrane from behind
            col = col + albedo * SUN * (np.clip(-ndl, 0.0, 1.0) * transmit * sun_lit)[:, None] * back
        return col

    def _pixel_xyz(self, mask, depth, W, H):
        ys, xs = np.nonzero(mask)
        return xs + 0.5 - W / 2, H / 2 - (ys + 0.5), depth[mask]

    # ------------------------------------------------------------------ solid parts
    def _shade_solid(self, part, depth, attr, mask, ao_map, shadow, s, pose, W, H):
        g = self.genome
        pm = part[mask]
        at = attr[mask]
        n = at[:, 0:3]
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        is_wing = np.isin(pm, WING_PARTS)
        wing_top = is_wing & (n[:, 2] >= 0)  # wing normals point up: facing us means we see its top
        two_sided = np.isin(pm, (*WING_PARTS, LOBE))
        n = np.where((two_sided & (n[:, 2] < 0))[:, None], -n, n)
        ao = ao_map[mask]
        px, py, pz = self._pixel_xyz(mask, depth, W, H)
        lit = shadow(px, py, pz, bias=1.5 + 0.02 * s * (1 - np.abs(n @ self._sun)))

        albedo = np.zeros((len(pm), 3))
        gloss = np.full(len(pm), g.gloss)
        silver = np.zeros(len(pm))
        body = np.isin(pm, (BODY, ARM))
        if body.any():
            albedo[body], gloss[body], silver[body] = self._body_albedo(at[body], pm[body], s, pose)
        if is_wing.any():
            albedo[is_wing], gloss[is_wing] = self._wing_albedo(at[is_wing], wing_top[is_wing], pm[is_wing], s)
        lobe = pm == LOBE
        if lobe.any():  # cephalic fins: dark skin, paler on the inner face and towards the tip
            col = _mix(self.c_base, self.c_base * 0.5 + self.c_belly * 0.5, smoothstep(0.3, 1.0, at[lobe, 4]) * 0.6)
            albedo[lobe] = col * (1.0 + 0.08 * fbm(at[lobe, 3] * 60, at[lobe, 6] * 4, 1.0, 2))[:, None]
        wrap = np.full(len(pm), self.wrap)
        transmit = np.where(is_wing, 0.25 if g.kind in ("squid", "octopus") else 0.08, 0.0)
        return self._light(albedo, n, lit, ao, gloss, silver, wrap, back=1.0, transmit=transmit)

    def _surface_coords(self, a, v, w):
        """Roughly isotropic coordinates on the body surface, in body lengths: (along, around)."""
        m = self.model
        ac = np.clip(a, 0.0, 1.0)
        girth = 0.5 * (m.top(ac) + m.bottom(ac)) + 0.5 * m.half_width(ac)
        ang = np.arctan2(v, w)
        return a, ang * girth * 0.65, ang

    def _body_albedo(self, at, pm, s, pose):
        g = self.genome
        c = g.colors
        a, v, w = at[:, 3], at[:, 4], at[:, 5]
        is_arm = pm == ARM
        u1, u2, ang = self._surface_coords(a, v, w)
        px_len = 1.0 / (self.scale)  # one final pixel, in body lengths
        seed = g.seed % 997

        # Countershading: dark back, pale belly, with a soft, slightly irregular boundary.
        col = np.broadcast_to(self.c_base, (len(a), 3)).copy()
        # Backs are a touch darker and flanks a touch lighter than the base colour.
        col *= (1.0 + 0.18 * (0.15 - np.clip(v, -1, 1)))[:, None]
        if c.belly_lightness > 0.2:
            level = c.belly_level if c.belly_level is not None else -0.05 - 0.2 * (1 - c.belly_lightness)
            soft = {"shark": 0.04, "whale": 0.07, "ray": 0.1}.get(g.kind, 0.16)
            rag = 0.0
            if g.kind in ("shark", "whale"):
                rag = 0.07 * fbm(u1 * 18.0, u2 * 6.0, seed) + 0.04 * np.sin(u1 * 31.0 + 4.0 * c.pattern_offset)
                # The pale belly stops short of the tail: the tail stalk is dark all round.
                rag = rag + 0.9 * smoothstep(0.72, 0.97, a)
            t = smoothstep(level + soft, level - soft, v + rag)
            if g.kind in ("squid", "octopus"):
                t = t * 0.6
            col = _mix(col, self.c_belly, t)
        else:
            t = np.zeros(len(a))
        belly_t = t

        # Pattern.
        m1, m2 = self._pattern(a, v, w, u1, u2, ang, pose, px_len)
        if c.accent_edge_color:  # a dark rim round each mark (clownfish bands)
            grown, _ = self._pattern(a, v, w, u1, u2, ang, pose, px_len, grow=max(0.008, 1.2 * px_len))
            col = _mix(col, self.c_accent_edge, np.clip(grown - m1, 0, 1))
        col = _mix(col, self.c_accent, m1)
        col = _mix(col, self.c_accent2, m2)

        gloss = np.full(len(a), g.gloss)
        silver = np.full(len(a), g.silver) * (0.35 + 0.65 * smoothstep(0.85, -0.2, v))  # less on the dark back
        silver *= 1.0 - 0.7 * np.maximum(m1, m2) * (c.pattern not in ("neon", "salmon", "trout"))
        if c.pattern == "neon":  # the neon stripe is iridescent
            silver = np.maximum(silver, 0.55 * m1)

        # Scales: overlapping rows, each with a darker trailing rim and its own sheen.
        if g.scales > 0 and not is_arm.all():
            sz = g.scales
            fade = smoothstep(1.2, 3.0, sz / px_len)
            if np.any(fade > 0):
                su, sv = u1 / sz, u2 / (sz * 0.75)
                row = np.floor(sv)
                su = su + 0.5 * (row % 2)
                fu, fv = su - np.floor(su), sv - row
                # Jitter each scale's outline a little so the rows don't read as a printed grid.
                fu = fu + 0.12 * (_hash(np.floor(su), row, seed + 1.0) - 0.5)
                rim = smoothstep(0.7, 0.97, fu + 0.55 * (fv - 0.5) ** 2 * 2)
                cell = _hash(np.floor(su), row, seed)
                keep = smoothstep(g.eye_pos + 0.08, g.eye_pos + 0.14, a) * (a < 0.99) * ~is_arm
                k = fade * keep * (0.6 + 0.4 * fbm(u1 * 12, u2 * 12, seed + 6.0, 2))
                col *= (1.0 - (0.1 * rim - 0.06 * (cell - 0.5)) * k)[:, None]
                silver = silver * (1.0 + (0.7 * (cell - 0.5) - 0.2 * rim) * k)
                gloss = gloss * (1.0 + (1.2 * (cell - 0.5) - 0.3 * rim) * k)
                # Lateral line: a row of tiny pores along the flank.
                ll = 0.3 * (1 - a) + 0.05
                line = smoothstep(px_len * 0.9, px_len * 0.3, np.abs(u2 - ll * 0.25)) * (a > g.eye_pos + 0.12) * \
                    (a < 0.97) * (np.abs(ang) < math.pi / 2)
                col *= (1.0 - 0.18 * line * fade)[:, None]

        # Skin: fine mottling everywhere, chromatophores on cephalopods, nicks and scars on big animals.
        grain = fbm(u1 * 40.0, u2 * 40.0, seed + 3.0, 3)
        mott = {"fish": 0.05, "shark": 0.07, "whale": 0.08, "ray": 0.08, "squid": 0.06, "octopus": 0.12,
                "urchin": 0.15}[g.kind]
        col *= (1.0 + mott * grain)[:, None]
        if g.kind in ("squid", "octopus"):
            dots = _hash(np.floor(u1 / 0.012), np.floor(u2 / 0.012), seed + 9.0)
            chrom = (dots > 0.8) * smoothstep(1.0, 2.5, 0.012 / px_len)
            col = _mix(col, col * np.array([0.55, 0.3, 0.25]), chrom * 0.7)
            gloss = gloss * (1.0 + 0.5 * grain)
        if g.kind == "urchin":  # the shell is studded with the knobs the spines grow from
            knob = _hash(np.floor(u1 / 0.03), np.floor(u2 / 0.03), seed)
            col *= (1.0 - 0.25 * (knob > 0.6))[:, None]
        if g.scars > 0:
            col = _mix(col, col * 1.6 + 0.04, self._scars(u1, u2, px_len) * g.scars)
        if g.knobs > 0:  # humpback tubercles on the head
            kn = _hash(np.floor(u1 / 0.018), np.floor(u2 / 0.018), seed + 5.0)
            fu = u1 / 0.018 - np.floor(u1 / 0.018) - 0.5
            fv = u2 / 0.018 - np.floor(u2 / 0.018) - 0.5
            bump = (kn > 0.55) * smoothstep(0.2, 0.05, fu * fu + fv * fv) * (a < 0.2) * (v > -0.2)
            # Each knob: a lit upper half and a soft shadow under it, so it reads as a bump, not a hole.
            col = _mix(col, col * 1.35 + 0.01, bump * g.knobs * smoothstep(0.0, -0.08, fv) * 0.7)
            col = _mix(col, col * 0.7, bump * g.knobs * smoothstep(0.0, 0.08, fv) * 0.7)
            # Barnacle clusters on the chin and throat.
            barn = (fbm(u1 * 30, u2 * 30, seed + 8.0, 2) > 0.45) * (a < 0.12) * (v < -0.2)
            col = _mix(col, np.array([0.55, 0.55, 0.5]), barn * g.knobs)

        # Head details: gill cover or gill slits, mouth, blowhole.
        if g.gill_slits == 1 and g.kind == "fish":
            ga = g.eye_pos + 0.1 + 0.035 * (1 - v ** 2)
            d = a - ga
            line = smoothstep(px_len * 0.9, 0.0, np.abs(d)) * (np.abs(v) < 0.75) * (np.abs(ang) < 1.9)
            behind = smoothstep(0.0, 0.02, d) * smoothstep(0.05, 0.02, d) * (np.abs(v) < 0.75)
            col = _mix(col, col * 0.45, line * 0.8)
            col = col * (1.0 + 0.1 * behind)[:, None]
        elif g.gill_slits > 1:
            gap = max(0.02, 1.6 * px_len)
            for i in range(int(g.gill_slits)):
                ga = g.eye_pos + 0.09 + i * gap + 0.012 * (1 - v ** 2)
                if g.kind == "ray":
                    continue
                d = a - ga
                slit = smoothstep(px_len * 0.8, 0.0, np.abs(d)) * (v > -0.5) * (v < 0.3 + 0.03 * i) * \
                    (np.abs(ang) < 1.6)
                lip = smoothstep(px_len * 1.6, px_len * 0.8, np.abs(d - px_len)) * (v > -0.5) * (v < 0.3)
                col = _mix(col, col * 0.35, slit * 0.85)
                col = col * (1.0 + 0.12 * lip)[:, None]
        if g.mouth > 0 and not is_arm.all():
            col = self._mouth(col, a, v, ang, px_len)
        if g.kind == "whale":  # blowhole on top of the head
            ba = g.eye_pos * 0.6 + 0.05 if g.beak else g.eye_pos + 0.03
            hole = smoothstep(0.012 + px_len * 0.5, 0.004, np.hypot(a - ba, (np.abs(ang - math.pi / 2)) * 0.02))
            col = _mix(col, col * 0.3, hole)
        if g.pleats:  # throat grooves running back from the chin
            groove = (np.mod(v * 14.0, 1.0) < 0.25) * (a > 0.03) * (a < 0.48) * (belly_t > 0.5) * \
                smoothstep(0.5, 0.4, a)
            col = _mix(col, col * 0.6, groove * smoothstep(1.0, 2.5, 0.07 / px_len))

        # Arms: pale suckers in rows on the inner side.
        if is_arm.any():
            inner = at[:, 6]
            arm_col = _mix(col, self.c_belly, smoothstep(0.1, 0.6, inner) * 0.7)
            per = 0.022
            ua = a / per
            fu = ua - np.floor(ua) - 0.5
            row = (np.floor(ua) % 2) * 0.5
            fw = np.mod(np.arctan2(v, w) * 1.2 + row, 1.0) - 0.5
            suck = smoothstep(0.2, 0.1, fu * fu + fw * fw) * smoothstep(0.35, 0.7, inner) * \
                smoothstep(1.0, 2.5, per / px_len)
            ring = smoothstep(0.03, 0.0, np.abs(fu * fu + fw * fw - 0.16)) * smoothstep(0.35, 0.7, inner)
            arm_col = _mix(arm_col, np.array([0.85, 0.72, 0.68]), suck * 0.8)
            arm_col = _mix(arm_col, arm_col * 0.6, ring * 0.5 * smoothstep(1.0, 2.5, per / px_len))
            col = np.where(is_arm[:, None], arm_col, col)
        return col, gloss, np.clip(silver, 0, 1)

    def _mouth(self, col, a, v, ang, px_len):
        g = self.genome
        a0 = max(0.012, g.eye_pos - (0.03 if g.teeth else 0.07))
        if g.kind == "fish":
            a0 = 0.0
        a1 = a0 + g.mouth
        t = np.clip((a - a0) / max(g.mouth, 1e-6), 0.0, 1.0)
        mv = g.mouth_height + (0.22 if g.teeth else 0.08) * t ** 1.3
        dv = (v - mv) * self.genome.height * 0.5  # in body lengths
        inside = (a >= a0 - px_len) & (a <= a1) & (np.abs(ang) < 1.7)
        width = px_len * 0.7 + 0.003
        if g.teeth:
            gape = 0.022 * (1 - t ** 2) * min(1.0, g.mouth / 0.1) + px_len
            mouth = inside & (dv < gape * 0.5) & (dv > -gape * 0.5)
            col = np.where(mouth[:, None], np.array([0.06, 0.02, 0.02]), col)
            # Interlocking triangular teeth along both jaws.
            per = max(0.008, 1.5 * px_len)
            ph = np.mod(a / per, 1.0)
            tri = np.abs(ph - 0.5) * 2
            upper = inside & (dv > gape * 0.5 - (1 - tri) * gape * 0.55) & (dv < gape * 0.5)
            lower = inside & (dv < -gape * 0.5 + tri * gape * 0.45) & (dv > -gape * 0.5)
            col = np.where((upper | lower)[:, None] & (t < 0.92)[:, None], np.array([0.85, 0.82, 0.75]), col)
            lip = inside & (np.abs(np.abs(dv) - gape * 0.5) < width)
            col = _mix(col, col * 0.5, lip * 0.6)
        else:
            line = inside * smoothstep(width, width * 0.3, np.abs(dv))
            col = _mix(col, col * 0.25, line)
        return col

    def _scars(self, u1, u2, px_len):
        """A few faint, healed scratches, curved and broken, unique to each animal."""
        rng = np.random.default_rng(self.genome.seed + 11)
        out = np.zeros(len(u1))
        width = max(0.002, 0.5 * px_len)
        for _ in range(10):
            c0 = np.array([rng.uniform(0.1, 0.9), rng.uniform(-0.08, 0.08)])
            ang = rng.uniform(-0.6, 0.6) + (math.pi / 2 if rng.random() < 0.3 else 0.0)
            ln = rng.uniform(0.015, 0.07)
            bend = rng.uniform(-8, 8)
            d = np.array([math.cos(ang), math.sin(ang)])
            rel_u, rel_v = u1 - c0[0], u2 - c0[1]
            along = rel_u * d[0] + rel_v * d[1]
            across = -rel_u * d[1] + rel_v * d[0] - bend * (along - ln / 2) ** 2  # a gentle curve
            inside = (along > 0) & (along < ln)
            broken = vnoise(along * 200.0, c0[0] * 50.0, self.genome.seed % 97) > -0.3
            fade = np.sin(np.clip(along / ln, 0, 1) * math.pi) ** 0.5
            out = np.maximum(out, inside * broken * fade * smoothstep(width, width * 0.2, np.abs(across))
                             * rng.uniform(0.3, 0.7))
        return out

    # ------------------------------------------------------------------ patterns
    def _pattern(self, a, v, w, u1, u2, ang, pose, px_len, grow=0.0):
        """Pattern coverage (accent, accent2), each 0..1 per body pixel. ``grow`` widens the
        marks by that many body lengths (for dark rims round them)."""
        g = self.genome
        c = g.colors
        kind = c.pattern
        z = np.zeros(len(a))
        eye = g.eye_pos
        seed = g.seed % 997
        body = (a > eye + 0.08) | (a < eye - 0.25)
        aa = max(px_len, 0.004)  # soft edge: about a pixel

        def band(x, lo, hi):
            return smoothstep(lo - aa, lo + aa, x) * smoothstep(hi + aa, hi - aa, x)

        if kind == "stripes":
            count = 3 + int(c.pattern_scale * 4)
            ph = np.mod((a - c.pattern_offset) * count, 1.0)
            face = np.ones(len(a)) if c.pattern_on_fins else body
            # Bars bow forward a little at the middle of the flank, like real ones.
            ph = np.mod(ph + 0.04 * (1 - v ** 2) * count * 0.3, 1.0)
            gr = grow * count
            return face * (a < 0.95) * band(np.mod(ph + gr, 1.0), 0.0, 0.32 * (1 - 0.15 * np.abs(v)) + 2 * gr) * \
                (v > -0.6 - grow * 4), z
        if kind == "bands":
            count = 3 + int(c.pattern_scale * 4)
            u = np.mod((a - c.pattern_offset) * count, 1.0)
            keep = body * (a < 0.95) * (v > -0.6)
            return keep * band(u, 0.0, 0.22), keep * band(u, 0.5, 0.72)
        if kind == "shimmer":
            count = 2 + c.pattern_scale * 3
            u = np.sin(2 * math.pi * (a * count + 0.35 * v + 0.2 * np.sin(4.0 * v + 3.0 * a) - pose.phase))
            keep = (v > -0.75) * (a < 0.98)
            return keep * smoothstep(0.45, 0.75, u), keep * smoothstep(-0.5, -0.8, u)
        if kind == "neon":
            blue = band(a, eye + 0.03, 0.97) * band(v, 0.02, 0.3)
            red = band(a, 0.45, 0.97) * band(v, -0.75, 0.02)
            return blue, red
        if kind == "whale_shark":
            cells = 10 + int(c.pattern_scale * 6)
            lines = smoothstep(0.06, 0.03, np.abs(np.mod(a * cells + c.pattern_offset, 1.0) - 0.05)) * (a > eye + 0.1)
            lines = lines * (0.5 + 0.5 * fbm(a * 40, v * 6, seed, 2))  # faint and broken, not a ruled grid
            angn = ang / math.pi
            ridges = smoothstep(0.04, 0.02, np.abs(np.mod(angn * 5.0, 1.0) - 0.5)) * (a > 0.25)
            ca = a * cells + c.pattern_offset
            cb = angn * 5.0 + 0.25
            ia, ib = np.floor(ca), np.floor(cb)
            jit = vnoise(ia * 3.1, ib * 2.3, seed)
            fa = ca - ia - 0.55 - 0.08 * jit  # centred in its cell, so never clipped at the border
            fb = cb - ib - 0.5 + 0.08 * jit
            rad = 0.03 + 0.015 * _hash(ia, ib, seed)
            spot = smoothstep(rad, rad * 0.7, fa * fa + (fb * 1.1) ** 2) * (_hash(ia + 2, ib, seed) > 0.12)
            head_spots = (a < eye + 0.1) * self._spots(a * 2.0, angn * 1.2, size=0.045)
            keep = (v > -0.35) * (a < 0.97)
            return keep * np.maximum(spot, head_spots) * 0.85, keep * np.maximum(lines, ridges) * 0.3
        if kind == "mackerel":
            # Atlantic mackerel: 20-30 thin, wavy, near-vertical dark lines over the green-blue back.
            count = 22 + int(c.pattern_scale * 8)
            u = (a - c.pattern_offset) * count
            wob = 0.35 * np.sin(v * 9.0 + a * 13.0) + 0.3 * fbm(a * 25, v * 5, seed)
            bars = band(np.mod(u + wob + 0.4 * v, 1.0), 0.0, 0.3) * smoothstep(0.15, 0.3, v) * \
                (a > eye + 0.12) * (a < 0.97)
            line = band(v, 0.0, 0.08) * (a > 0.25) * (a < 0.97)
            return bars, line * 0.6
        if kind in ("spots", "rings"):
            angn = ang / math.pi
            if kind == "rings":
                return body * self._spots(a, angn * 0.6, size=0.06, ring=True) * (v > -0.5), z
            if g.kind == "ray":  # many small white dots, as on a spotted eagle ray
                return self._spots(u1 * 2.6, u2 * 2.6, size=0.035, density=0.65) * (v > 0.1), z
            sp = self._spots(a, angn * 0.6, size=0.05)
            return body * sp * (v > -0.5), z
        if kind == "tiger":
            count = 6 + int(c.pattern_scale * 8)
            u = (a - c.pattern_offset) * count
            wob = 0.12 * np.sin(v * 5.0 + np.floor(u) * 2.3) + 0.12 * fbm(a * 20, v * 3, seed)
            bars = band(np.mod(u + wob, 1.0), 0.0, 0.28)
            fadeout = smoothstep(0.98, 0.6, a) * smoothstep(0.2, 0.32, a)
            return body * bars * fadeout * smoothstep(0.1 - 0.35 * a, 0.35 - 0.35 * a, v) * 0.85, z
        if kind == "orca":
            e = ((a - (eye + 0.12)) / 0.075) ** 2 + ((v - 0.32) / 0.17) ** 2
            patch = smoothstep(1.05, 0.95, e) * (np.abs(w) > 0.15)
            o = c.pattern_offset * 0.04
            edge = np.interp(a, [0.0, 0.08, 0.2, 0.32, 0.42, 0.5, 0.58, 0.66, 0.74, 0.8, 0.86],
                             [-0.25, -0.12, -0.35, -0.65, -0.75, -0.45, 0.05 + o, 0.12, -0.15, -0.6, -1.2])
            white = smoothstep(edge + 0.03, edge - 0.03, v) * (a < 0.86)
            d = g.dorsal
            saddle = band(a, d.start + 0.02, d.end + 0.14) * smoothstep(0.5, 0.6, v - 0.25 * np.abs(a - d.end - 0.05) / 0.1)
            # The saddle patch is streaky, not solid.
            saddle *= 0.65 + 0.35 * fbm(a * 60, v * 8, seed)
            return np.maximum(patch, white), saddle
        if kind == "lateral":
            return body * band(v, -0.1 - 0.2 * c.pattern_scale, 0.15), z
        if kind == "saddle":
            return smoothstep(0.4 - 0.3 * c.pattern_scale, 0.5 - 0.3 * c.pattern_scale, v) * (a > 0.05), z
        if kind == "gradient":
            start = 0.45 + 0.2 * c.pattern_offset
            return smoothstep(start, start + 0.25, a), z
        if kind in ("koi", "calico"):
            # Big irregular patches of colour (and smaller black ones), different on every fish.
            n1 = fbm(u1 * 7.0 + c.pattern_offset * 10, u2 * 7.0, seed, 4)
            thr = 0.35 - 0.6 * c.pattern_scale
            red = smoothstep(thr - 0.04, thr + 0.04, n1) * (v > -0.35 + 0.1 * n1)
            n2 = fbm(u1 * 11.0 + 5.0, u2 * 11.0, seed + 40, 4)
            black_thr = 0.45 if kind == "koi" else 0.25
            black = smoothstep(black_thr - 0.04, black_thr + 0.04, n2) * (v > -0.6)
            if c.accent2_color is None:
                black = black * 0
            return red * (a > 0.02), black * (a > eye + 0.02)
        if kind == "discus":
            # Wavy turquoise lines running nose to tail over a warm body.
            count = 6 + int(c.pattern_scale * 6)
            ph = np.mod(v * count * 0.5 + 0.35 * np.sin(a * 22.0 + v * 3.0) + 0.2 * fbm(a * 10, v * 6, seed), 1.0)
            return band(ph, 0.0, 0.42) * (a > eye - 0.05), z
        if kind == "trout":
            pink = band(v, -0.25, 0.12) * band(a, eye + 0.05, 0.97) * (0.7 + 0.3 * fbm(a * 20, v * 4, seed))
            sp = self._spots(a, ang / math.pi * 0.6, size=0.022, density=0.7) * (v > -0.35)
            return sp, pink
        if kind == "salmon":
            sp = self._spots(a, ang / math.pi * 0.6, size=0.016, density=0.45) * (v > 0.1) * (a < 0.8)
            return sp, z
        if kind == "blotches":
            n1 = fbm(u1 * (6.0 + 10.0 * c.pattern_scale), u2 * (6.0 + 10.0 * c.pattern_scale), seed, 4)
            return smoothstep(0.1, 0.35, n1) * 0.8, smoothstep(-0.2, -0.45, n1) * 0.6
        if kind == "tang":
            # Regal blue tang: the black "palette" band from the eye to the tail with its loop.
            c1 = 0.55 - 0.7 * ((a - 0.5) / 0.45) ** 2
            band1 = smoothstep(0.17, 0.11, np.abs(v - c1)) * band(a, eye + 0.03, 0.97)
            e = ((a - 0.6) / 0.2) ** 2 + ((v - 0.12) / 0.3) ** 2
            loop = smoothstep(0.32, 0.2, np.abs(e - 1.0)) * (a < 0.85)
            return np.maximum(band1, loop), z
        if kind == "dolphin_cape":
            # Common dolphin: dark cape over the back dipping into a V under the dorsal fin, a tan
            # hourglass patch in front and a grey one behind.
            cape_edge = 0.35 - 0.45 * np.exp(-((a - 0.48) / 0.08) ** 2)
            under = smoothstep(cape_edge + 0.04, cape_edge - 0.04, v) * (v > -0.35)
            tan = under * band(a, 0.12, 0.5)
            grey = under * band(a, 0.5, 0.86)
            return tan, grey
        if kind == "barracuda":
            count = 16 + int(c.pattern_scale * 6)
            bars = band(np.mod(a * count - 0.4 * v, 1.0), 0.0, 0.35) * (v > 0.15) * band(a, 0.2, 0.9)
            blot = self._spots(a, ang / math.pi * 0.6, size=0.03, density=0.4) * (v < -0.1) * (v > -0.6) * (a > 0.5)
            return bars * 0.7, blot
        if kind == "red_belly":  # red-bellied piranha: red throat and belly under silver flanks
            n1 = 0.08 * fbm(a * 15, v * 4, seed)
            return smoothstep(-0.05, -0.25, v + n1) * band(a, 0.04, 0.8), z
        if kind == "humpback":
            n1 = fbm(u1 * 9.0, u2 * 9.0, seed, 3)
            return smoothstep(-0.2 + n1 * 0.2, -0.35 + n1 * 0.2, v) * (a < 0.9), z
        return z, z

    def _spots(self, ua, ub, size=0.05, ring=False, density=0.5):
        """Hash-placed round spots; ``size`` is the spot radius as a fraction of a cell."""
        c = self.genome.colors
        cells = 5 + int(c.pattern_scale * 6)
        u, v = ua * cells + c.pattern_offset, ub * cells
        iu, iv = np.floor(u), np.floor(v)
        h = _hash(iu, iv, self.genome.seed % 997)
        fa = u - iu - 0.5 + (_hash(iu, iv + 7, 3) - 0.5) * 0.3
        fb = v - iv - 0.5 + (h - 0.5) * 0.4
        has = np.mod(h * 7.0, 1.0) < density
        r2 = fa * fa + fb * fb
        rad = size * (0.8 + 0.4 * _hash(iu + 3, iv, 5))
        edge = max(0.004, 0.6 / (self.scale * 0.6 / cells + 1e-6) * 0.05)
        if ring:
            hole = 0.012 if self.scale * 0.6 / cells > 3.0 else -1.0
            return has * smoothstep(rad + edge, rad - edge, r2) * smoothstep(hole - edge, hole + edge, r2)
        return has * smoothstep(rad + edge, rad - edge, r2)

    # ------------------------------------------------------------------ wings
    def _wing_albedo(self, at, top, pm, s):
        g = self.genome
        c = g.colors
        a, r, side, m = at[:, 3], at[:, 4], at[:, 5], at[:, 6]
        seed = g.seed % 997
        span = max(g.wings.span, g.hammer, 0.05)
        x, y = a, side * r * span  # (along, across) in body lengths
        col = np.broadcast_to(self.c_base, (len(a), 3)).copy()
        grain = fbm(x * 30.0, y * 30.0, seed + 2.0, 3)
        col *= (1.0 + 0.08 * grain)[:, None]
        if c.pattern in ("spots", "rings"):
            k = 2.6 if g.kind == "ray" else 1.6
            sp = self._spots(x * k, y * k, size=0.035 if g.kind == "ray" else 0.05, ring=c.pattern == "rings",
                             density=0.65 if g.kind == "ray" else 0.5)
            col = _mix(col, self.c_accent, sp)
        elif c.pattern == "chevron":
            # Manta shoulder patches: pale wedges on the back behind the head.
            wedge = smoothstep(0.62, 0.5, r + 0.6 * m) * smoothstep(0.08, 0.2, r) * smoothstep(0.85, 0.55, m)
            wedge *= 0.75 + 0.25 * fbm(x * 15, y * 15, seed)
            col = _mix(col, self.c_accent, wedge)
        elif c.pattern in ("shimmer", "tiger", "blotches"):
            n1 = fbm(x * 18.0, y * 18.0, seed, 3)
            col = _mix(col, self.c_accent, smoothstep(0.15, 0.4, n1) * 0.6)
        # Wing margins darken a touch towards the edge.
        col *= (1.0 - 0.15 * smoothstep(0.8, 1.0, r))[:, None]
        # Undersides are pale (rays, the hammerhead's head), edged with grey.
        under = ~top
        if c.belly_lightness > 0.25 and g.kind in ("ray", "shark"):
            ucol = np.broadcast_to(self.c_belly, (len(a), 3)).copy()
            ucol = _mix(ucol, self.c_base * 0.6 + self.c_belly * 0.4, smoothstep(0.75, 1.0, r) * 0.7)
            if c.belly_pattern:  # every manta has its own pattern of belly spots
                sp = self._spots(x * 2.4 + 3, y * 2.4, size=0.03, density=0.35) * smoothstep(0.7, 0.3, r)
                ucol = _mix(ucol, np.array([0.03, 0.03, 0.04]), sp)
            col = np.where(under[:, None], ucol, col)
        gloss = np.full(len(a), g.gloss)
        return col, gloss

    # ------------------------------------------------------------------ fins
    def _shade_fins(self, part, depth, attr, mask, shadow, s, W, H):
        g = self.genome
        c = g.colors
        pm = part[mask]
        at = attr[mask]
        n = at[:, 0:3]
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        back_face = n[:, 2] < 0
        n = np.where(back_face[:, None], -n, n)
        m, r, fa = at[:, 3], at[:, 4], at[:, 5]
        px_len = 1.0 / self.scale
        seed = g.seed % 997
        k = len(pm)
        skin_fins = g.kind in ("shark", "whale") or g.fin_alpha >= 0.98

        if skin_fins:
            col = np.broadcast_to(self.c_base, (k, 3)).copy()
            if not c.tail_body_color and c.fin_color:
                col = np.broadcast_to(self.c_fin, (k, 3)).copy()
            # Paired fins show the pale belly colour on their underside.
            paired = np.isin(pm, PAIRED)
            world_up = n @ self._up
            if c.belly_lightness > 0.25:
                under = paired & (world_up < 0)
                col = np.where(under[:, None], _mix(self.c_belly, col, 0.3), col)
            if c.fin_color and not c.tail_body_color:
                pass
            elif c.fin_color:  # sharks keep the fin colour as a slightly different tone of skin
                col = _mix(col, self.c_fin, 0.5 * (pm != CAUDAL))
            col *= (1.0 + 0.06 * fbm(fa * 40, r * 6, seed, 3))[:, None]
            # Skin fins darken towards their tips and trailing edges.
            col *= (1.0 - 0.22 * smoothstep(0.55, 1.0, r) - 0.1 * smoothstep(0.6, 1.0, m))[:, None]
            alpha = np.ones(k)
        else:
            col = np.broadcast_to(self.c_fin, (k, 3)).copy()
            if c.tail_body_color:
                col = np.where((pm == CAUDAL)[:, None], self.c_base, col)
            elif c.tail_color:
                col = np.where((pm == CAUDAL)[:, None], _lin(P.hex_rgb(c.tail_color)), col)
            # Fin rays: bony spokes from root to edge, more opaque than the membrane between.
            count = np.where(pm == CAUDAL, 18, np.where(np.isin(pm, PAIRED), 10, 16)).astype(float)
            span = np.where(pm == CAUDAL, g.tail_spread * g.height, 0.2)
            count = np.maximum(4, np.minimum(count, span / (2.2 * px_len)))
            ph = np.mod(m * count, 1.0)
            ray = smoothstep(0.6, 0.9, np.abs(ph - 0.5) * 2) * smoothstep(0.02, 0.15, r)
            ray *= smoothstep(1.2, 3.0, (span / count) / px_len)
            col = _mix(col, col * 0.62, ray * 0.6)
            # A milky, darker root and a paler, thinner outer edge.
            col = _mix(col, col * 0.75, smoothstep(0.25, 0.0, r) * 0.5)
            alpha = g.fin_alpha * (0.85 + 0.15 * smoothstep(0.4, 0.0, r)) + (1 - g.fin_alpha) * ray * 0.6
            alpha *= 1.0 - 0.35 * smoothstep(0.75, 1.0, r)
            alpha = np.clip(alpha, 0.15, 1.0)
        if c.pattern_on_fins:
            if c.pattern == "stripes":
                count = 3 + int(c.pattern_scale * 4)
                ph = np.mod((fa - c.pattern_offset) * count, 1.0)
                stripe = smoothstep(0.34, 0.3, ph) * np.isin(pm, (DORSAL, ANAL, CAUDAL))
                col = _mix(col, self.c_accent, stripe)
                alpha = np.maximum(alpha, stripe * 0.95)
            elif c.pattern in ("spots", "trout", "salmon", "koi", "calico"):
                sp = self._spots(fa * 1.5, (m * 0.3 + r * 0.25), size=0.05)
                col = _mix(col, self.c_accent if c.pattern != "trout" else self.c_accent, sp)
                alpha = np.maximum(alpha, sp * 0.9)
        if c.fin_tips > 0:
            tip = smoothstep(1.0 - c.fin_tips - 0.04, 1.0 - c.fin_tips + 0.04, r)
            col = _mix(col, self.c_accent, tip)
            alpha = np.maximum(alpha, tip)
        if c.fin_edge_color:
            edge = smoothstep(0.84, 0.94, r)
            col = _mix(col, self.c_fin_edge, edge)
            alpha = np.maximum(alpha, edge * 0.95)
        far = np.isin(pm, FAR_PARTS)
        col = np.where(far[:, None], col * 0.8, col)

        px, py, pz = self._pixel_xyz(mask, depth, W, H)
        lit = shadow(px, py, pz, bias=2.0 + 0.01 * s)
        gloss = np.full(k, g.gloss * (1.0 if skin_fins else 0.5))
        silver = np.zeros(k)
        wrap = np.full(k, 0.5 if not skin_fins else self.wrap)
        shaded = self._light(col, n, lit, np.zeros(k), gloss, silver, wrap, rim=0.5, back=1.0,
                             transmit=0.0 if skin_fins else 0.9 * (1 - g.fin_alpha) + 0.3)
        return shaded, alpha

    # ------------------------------------------------------------------ spines
    def _draw_spines(self, rgb, alpha, depth, lines, s, W, H):
        """Urchin spines and pufferfish spikes: tapering, shaded cylinders, depth-tested."""
        g = self.genome
        base_r = _spine_radius(g)
        cxy = np.array([W / 2, H / 2])
        p0 = np.stack([cxy[0] + lines[:, 0, 0] * s, cxy[1] - lines[:, 0, 1] * s, lines[:, 0, 2] * s], -1)
        p1 = np.stack([cxy[0] + lines[:, 1, 0] * s, cxy[1] - lines[:, 1, 1] * s, lines[:, 1, 2] * s], -1)
        d3 = lines[:, 1] - lines[:, 0]
        d3 = d3 / (np.linalg.norm(d3, axis=1, keepdims=True) + 1e-9)
        # A lit cylinder: brightest where it runs across the light.
        dl = d3 @ self._sun
        cyl = np.sqrt(np.clip(1 - dl ** 2, 0, 1))
        toward = np.clip(d3[:, 2], -1, 1)
        span = np.abs(p1[:, :2] - p0[:, :2]).max(axis=1)
        steps = int(min(400, max(4, span.max() * 1.2)))
        t = np.linspace(0.0, 1.0, steps)
        rad = base_r * s * (1.0 - 0.85 * t) ** 0.9  # (steps,)
        rmax = max(0.5, float(rad.max()))
        R = int(math.ceil(rmax))
        offs = np.array([(dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1)])
        if native.lib is not None:
            win_z = np.full(H * W, EMPTY_DEPTH)
            wk, wt, wox, woy = (np.full(H * W, -1, dtype=np.int32) for _ in range(4))
            native.lib.fg_spines(native.ptr(native.f64(p0)), native.ptr(native.f64(p1)), len(p0), steps,
                                 native.ptr(native.f64(rad)), R, native.ptr(native.f64(depth)),
                                 native.ptr(native.f64(alpha)), W, H, native.ptr(win_z), native.iptr(wk),
                                 native.iptr(wt), native.iptr(wox), native.iptr(woy))
            hit = np.nonzero(wk >= 0)[0]
            if not len(hit):
                return
            k_, t_ = wk[hit], wt[hit]
            o_ = (woy[hit] + R) * (2 * R + 1) + (wox[hit] + R)
            iy, ix, z = hit // W, hit % W, win_z[hit]
            win = slice(None)
            self._shade_spines(rgb, alpha, depth, k_, t_, o_, ix, iy, z, win, t, rad, offs, cyl, toward)
            return
        dist = np.hypot(offs[:, 0], offs[:, 1])
        pts = p0[:, None] + (p1 - p0)[:, None] * t[None, :, None]  # (k, steps, 3)
        ks, ts, os_ = np.nonzero(dist[None, None, :] <= np.maximum(rad, 0.5)[None, :, None] + 0.0 * pts[..., :1])
        ix = np.floor(pts[ks, ts, 0] + offs[os_, 0]).astype(int)
        iy = np.floor(pts[ks, ts, 1] + offs[os_, 1]).astype(int)
        z = pts[ks, ts, 2] + np.sqrt(np.maximum(rad[ts] ** 2 - dist[os_] ** 2, 0))
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ks, ts, os_, ix, iy, z = ks[ok], ts[ok], os_[ok], ix[ok], iy[ok], z[ok]
        vis = (z > depth[iy, ix] - 0.5) | (alpha[iy, ix] == 0)
        ks, ts, os_, ix, iy, z = ks[vis], ts[vis], os_[vis], ix[vis], iy[vis], z[vis]
        if not len(z):
            return
        pix = iy * W + ix
        order = np.lexsort((z, pix))
        last = np.ones(len(order), dtype=bool)
        last[:-1] = pix[order][1:] != pix[order][:-1]
        win = order[last]
        k_, t_, o_ = ks[win], ts[win], os_[win]
        self._shade_spines(rgb, alpha, depth, k_, t_, o_, ix, iy, z, win, t, rad, offs, cyl, toward)

    def _shade_spines(self, rgb, alpha, depth, k_, t_, o_, ix, iy, z, win, t, rad, offs, cyl, toward):
        g = self.genome
        base_col = _lin(self.palette.colors[P.FIN])
        tip_col = base_col * 0.55 + 0.45 * np.array([0.9, 0.88, 0.8])
        tt = t[t_]
        # Across the spine: brighter on the side facing the light (a cheap cylinder shading).
        across = (offs[o_, 0] * 0.0 + offs[o_, 1] * -1.0) / np.maximum(rad[t_], 0.5)
        col = base_col[None, :] * (1 - tt[:, None]) + tip_col[None, :] * tt[:, None]
        light = 0.25 + 0.75 * cyl[k_] * (0.75 + 0.25 * np.clip(across, -1, 1)) + 0.15 * np.clip(toward[k_], 0, 1)
        shade = col * (SUN * light[:, None] * 0.75 + SKY * 0.5)
        spec = (cyl[k_] ** 30) * g.gloss * 0.6
        shade = shade + spec[:, None]
        yy, xx = iy[win], ix[win]
        rgb[yy, xx] = shade
        alpha[yy, xx] = 1.0
        depth[yy, xx] = z[win]

    # ------------------------------------------------------------------ eyes
    def _draw_eyes(self, rgb, alpha, depth, eyes, axes, pose, s, W, H):
        seen = []
        if not self.model.eyes:
            return seen
        g = self.genome
        r = self.model.eyes[0].radius * s
        if g.eye_style == "dark":
            r *= 0.75
        r = max(r, 0.9 * self.ss)
        ceph = g.kind in ("squid", "octopus")
        for e, ax in zip(eyes, axes):
            if ax[2] < -0.15:
                continue
            ex, ey, ez = W / 2 + e[0] * s, H / 2 - e[1] * s, e[2] * s
            ix, iy = int(math.floor(ex)), int(math.floor(ey))
            if not (0 <= ix < W and 0 <= iy < H) or depth[iy, ix] > ez + max(1.5, r * 0.8):
                continue
            if ax[2] > 0.2:
                seen.append((ex, ey, r))
            R = int(math.ceil(r * 1.5)) + 1
            y0, y1, x0, x1 = max(0, iy - R), min(H, iy + R + 1), max(0, ix - R), min(W, ix + R + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = xx + 0.5 - ex, yy + 0.5 - ey
            # Foreshorten the disc when the eye turns away from us.
            axy = np.array([ax[0], -ax[1]])
            nrm = np.linalg.norm(axy)
            squash = max(0.2, float(ax[2]))
            if nrm > 1e-3:
                ux, uy = axy / nrm
                du = (dx * ux + dy * uy) / squash
                dv = -dx * uy + dy * ux
                lx, ly = du * ux - dv * uy, du * uy + dv * ux
            else:
                lx, ly = dx, dy
            rho = np.hypot(lx, ly) / r
            on = alpha[y0:y1, x0:x1] > 0
            region = rgb[y0:y1, x0:x1]
            # Socket: a soft dark ring round the eye.
            sock = smoothstep(1.35, 1.0, rho) * smoothstep(0.85, 1.0, rho) * on
            region *= (1.0 - 0.35 * sock * (0.6 + 0.4 * smoothstep(0.5, -1.0, dy / r)))[..., None]
            inside = (rho < 1.0) & on
            if not inside.any():
                continue
            fwd = math.cos(pose.yaw)
            nx, ny = lx / r, ly / r  # -1..1 across the eye, y down
            # Iris: darker towards the rim, upper part shadowed by the brow.
            # Iris: a dark limbal ring, a brighter collar round the pupil, fine radial fibres.
            iris = self.c_iris * (0.35 + 0.65 * smoothstep(1.0, 0.7, rho))[..., None]
            iris = iris * (1.0 + 0.35 * smoothstep(0.75, 0.55, rho) * smoothstep(0.45, 0.6, rho))[..., None]
            fib = vnoise(np.arctan2(ny, nx) * 9.0, rho * 3.0, g.seed % 97)
            iris = iris * (1.0 + 0.12 * fib)[..., None]
            # The eyeball is a sphere: lit from above, shadowed underneath and by the brow.
            sph = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
            iris = iris * (0.55 + 0.45 * sph * smoothstep(1.0, -0.6, ny))[..., None]
            col = iris
            # Pupil.
            pxo = nx - 0.05 * fwd
            if ceph:
                if g.kind == "squid" or g.arms.tentacles:
                    pup = smoothstep(0.52, 0.45, np.hypot(pxo, ny))
                else:  # octopus: a horizontal slit
                    pup = smoothstep(0.22, 0.16, np.abs(ny) + 0.25 * pxo ** 2) * (np.abs(pxo) < 0.7)
            elif g.eye_style == "dark":
                pup = smoothstep(0.72, 0.62, np.hypot(pxo, ny))
            else:
                pup = smoothstep(0.58, 0.5, np.hypot(pxo, ny))
            col = _mix(col, np.array([0.004, 0.006, 0.01]), pup)
            # Wet cornea: a catch-light up towards the sun, and a faint reflection of the water.
            hl = smoothstep(0.17, 0.07, np.hypot(nx + 0.3, ny + 0.36))
            hl2 = smoothstep(0.1, 0.04, np.hypot(nx - 0.32, ny - 0.38)) * 0.3
            col = col + np.array([1.2, 1.2, 1.15]) * (hl + hl2)[..., None]
            # The cornea mirrors the bright water above in its upper half.
            col = col + ENV_TOP * 0.06 * smoothstep(0.2, -0.9, ny)[..., None] * sph[..., None]
            edge = smoothstep(1.0, 0.85, rho)
            out = _mix(region, col, edge)
            region[inside] = out[inside]
            rgb[y0:y1, x0:x1] = region
        return seen


def _spine_radius(g: FishGenome) -> float:
    """Base radius of an urchin spine or pufferfish spike, in body lengths."""
    if g.puff_spines:
        return 0.012
    sp = g.spines
    return float(np.clip(0.9 / max(sp.count, 1) ** 0.5 * (0.6 + sp.length), 0.006, 0.05))


def _cleanup(part: np.ndarray) -> np.ndarray:
    """Remove lone stray pixels and fill pinholes (only needed without supersampling)."""
    filled = part != 0
    nb = sum(_shift(filled.astype(np.int8), dy, dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)))
    part = part.copy()
    part[filled & (nb == 0)] = 0
    holes = (~filled) & (nb >= 3)
    if holes.any():
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            np_ = _shift(part, dy, dx)
            fill = holes & (np_ != 0) & (part == 0)
            part[fill] = np_[fill]
    return part


def _shift(arr: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """arr shifted so out[y, x] = arr[y + dy, x + dx] (out-of-range -> fill)."""
    out = np.full_like(arr, fill)
    h, w = arr.shape[:2]
    if abs(dy) >= h or abs(dx) >= w:
        return out
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = arr[ys, xs]
    return out
