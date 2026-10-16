"""Turns posed 3D fish into palette-indexed pixel-art frames."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from . import palette as P
from .geometry import (ANAL, ARM, BODY, CAUDAL, DORSAL, FAR_PARTS, FIN_PARTS, SKIN_PARTS, WING_FAR, WING_PARTS,
                       FishModel, Pose)
from .genome import FishGenome
from .raster import EMPTY_DEPTH, rasterize

LIGHT = np.array([0.0, 1.0, 0.55]) / math.hypot(1.0, 0.55)
N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _shift(arr: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """arr shifted so out[y, x] = arr[y + dy, x + dx] (out-of-range -> fill)."""
    out = np.full_like(arr, fill)
    h, w = arr.shape[:2]
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = arr[ys, xs]
    return out


class FishRenderer:
    """Renders one fish at a fixed scale so every frame lines up."""

    def __init__(self, genome: FishGenome, width: int = 32, height: int = 32, fit_poses: Iterable[Pose] = ()):
        self.genome = genome
        self.width = width
        self.height = height
        self.model = FishModel(genome, detail=max(width, height))
        self.palette = P.build_palette(genome.colors)
        poses = list(fit_poses) or [Pose(phase=p / 8) for p in range(8)]
        self.scale = self._fit_scale(poses)

    def _fit_scale(self, poses: list[Pose]) -> float:
        """Largest scale at which every pose fits inside the canvas (with a
        1px margin for the outline), keeping the fish centred."""
        ext_x = ext_y = 1e-6
        for pose in poses:
            cam, _, _, _, lines = self.model.posed(pose)
            pts = np.concatenate([cam, lines.reshape(-1, 3)])
            ext_x = max(ext_x, np.abs(pts[:, 0]).max())
            ext_y = max(ext_y, np.abs(pts[:, 1]).max())
        sx = (self.width / 2 - 1.5) / ext_x
        sy = (self.height / 2 - 1.5) / ext_y
        return min(sx, sy)

    # ------------------------------------------------------------------
    def render_indexed(self, pose: Pose) -> np.ndarray:
        """Render a pose to an (h, w) array of palette slot indices."""
        g = self.genome
        model = self.model
        cam, normals, eyes, eye_axes, lines = model.posed(pose)
        s = self.scale
        cx, cy = self.width / 2, self.height / 2
        screen = np.stack([cx + cam[:, 0] * s, cy - cam[:, 1] * s, cam[:, 2] * s], -1)

        attrs = np.concatenate([normals, model.mesh.attrs], axis=1)
        part, depth, attr = rasterize(screen, attrs, model.mesh.tris, model.mesh.tri_part, self.width, self.height)
        part = self._cleanup(part)

        n = attr[..., 0:3]
        n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
        is_fin = np.isin(part, FIN_PARTS)
        is_wing = np.isin(part, WING_PARTS)
        # Wing normals point up, so a wing facing the camera shows its back.
        wing_top = is_wing & (n[..., 2] >= 0)
        # Fins and wings are two-sided: always shade the face we're looking at.
        n = np.where(((is_fin | is_wing) & (n[..., 2] < 0))[..., None], -n, n)
        light = n @ LIGHT

        out = np.zeros(part.shape, dtype=np.uint8)
        body = np.isin(part, (BODY, ARM))
        out[body] = self._shade_body(attr, light, body, pose)
        # Seen from above (rays), light the wings from the viewer so both wings shade alike
        # through a flap; from the side keep the overhead light.
        wing_light = n[..., 2] * 1.6 - 0.75 if g.camera_tilt > 0.5 else light
        out[is_wing] = self._shade_wing(attr, wing_light, is_wing, wing_top, part)
        out[is_fin] = P.FIN
        # Fins on the far side of the body and fins facing away from the light are darker.
        out[np.isin(part, FAR_PARTS)] = P.FIN_DARK
        out[is_fin & (light < 0.1)] = P.FIN_DARK
        # Darken where the tail joins the body so the two read as separate shapes.
        out[(part == CAUDAL) & (attr[..., 4] < 0.18)] = P.FIN_DARK
        self._fin_rays(out, part, attr)
        if g.colors.tail_body_color:  # tail in the body's own colours (sharks, tuna)
            tail = part == CAUDAL
            out[tail] = P.BASE
            out[tail & (light > 0.84)] = P.HIGHLIGHT
            out[tail & ((light < 0.1) | (attr[..., 4] < 0.18))] = P.SHADOW
        if g.colors.pattern_on_fins and g.colors.pattern == "stripes":
            # Bars carry on across the median fins and the tail (angelfish).
            median = np.isin(part, (DORSAL, ANAL, CAUDAL))
            out[median & self._stripes(attr[..., 5])] = P.ACCENT
        if g.colors.fin_tips > 0:
            out[is_fin & (attr[..., 4] > 1.0 - g.colors.fin_tips)] = P.ACCENT

        # Arms and wings crossing over other skin: shade the far side of the edge
        # so overlapping arms read as separate shapes instead of one blob.
        skin = np.isin(part, SKIN_PARTS)
        extra = np.isin(part, (ARM, *WING_PARTS))
        if extra.any():
            for dy, dx in N4:
                np_ = _shift(part, dy, dx)
                nd = _shift(depth, dy, dx, fill=EMPTY_DEPTH)
                behind = skin & np.isin(np_, SKIN_PARTS) & (extra | np.isin(np_, (ARM, *WING_PARTS))) & (nd - depth > 2.0)
                out[behind] = P.SHADOW

        # Depth edges: where a fin overlaps something behind it, or something
        # in front overlaps a fin, outline the fin's edge with the dark fin colour.
        thresh = 0.6
        for dy, dx in N4:
            np_ = _shift(part, dy, dx)
            nd = _shift(depth, dy, dx, fill=EMPTY_DEPTH)
            edge = is_fin & (np_ != 0) & (np_ != part) & (np.abs(depth - nd) > thresh)
            out[edge] = P.FIN_DARK
            # Body pixels sitting in front of a fin get a hard outline.
            front = body & np.isin(np_, FIN_PARTS) & (depth - nd > thresh * 2)
            out[front & np.isin(np_, FAR_PARTS)] = P.SHADOW

        self._draw_spines(out, depth, lines)
        self._draw_mouth(out, depth, pose)
        self._draw_eyes(out, depth, eyes, eye_axes, pose)
        self._outline(out)
        return out

    # ------------------------------------------------------------------
    def _cleanup(self, part: np.ndarray) -> np.ndarray:
        """Remove lone stray pixels and fill pinholes - these flicker badly in animation."""
        filled = part != 0
        nb = sum(_shift(filled.astype(np.int8), dy, dx) for dy, dx in N4)
        part = part.copy()
        part[filled & (nb == 0)] = 0
        holes = (~filled) & (nb >= 3)
        if holes.any():
            # Fill with the most common neighbouring part (good enough: take any neighbour).
            for dy, dx in N4:
                np_ = _shift(part, dy, dx)
                fill = holes & (np_ != 0) & (part == 0)
                part[fill] = np_[fill]
        return part

    def _shade_body(self, attr, light, mask, pose) -> np.ndarray:
        g = self.genome
        c = g.colors
        a = attr[..., 3][mask]
        v = attr[..., 4][mask]  # -1 belly .. 1 back
        w = attr[..., 5][mask]  # 1 near flank, -1 far flank
        li = light[mask]

        res = np.full(a.shape, P.BASE, dtype=np.uint8)
        res[li > 0.84] = P.HIGHLIGHT
        res[li < 0.18] = P.SHADOW

        pattern = self._pattern(a, v, w, mask, pose)
        over_belly = c.pattern in ("neon", "orca")  # markings that cover the belly too
        if not over_belly:
            res[pattern > 0] = pattern[pattern > 0]

        if c.belly_lightness > 0.25:
            level = c.belly_level if c.belly_level is not None else -0.05 - 0.2 * (1 - c.belly_lightness)
            if g.kind == "shark":  # countershading meets in a ragged line, not a ruler-straight one
                level = level + 0.08 * np.sin(a * 31.0 + 4.0 * c.pattern_offset) + 0.05 * np.sin(a * 13.0)
            belly = v < level
            res[belly] = P.BELLY
            res[belly & (li < -0.35)] = P.SHADOW
            if g.pleats:  # throat grooves running back from the chin
                res[belly & (a > 0.04) & (a < 0.5) & (np.mod(v * 9.0, 1.0) < 0.3)] = P.SHADOW
        if over_belly:
            res[pattern > 0] = pattern[pattern > 0]

        # Gill slit: a short curved line behind the eye, only when there is room.
        px_len = 1.0 / self.scale
        if g.gill_slits == 1 and self.scale > 18:
            ga = g.eye_pos + 0.11 + 0.03 * (1 - v ** 2)
            gill = (np.abs(a - ga) < px_len * 0.55) & (np.abs(v) < 0.6) & (np.abs(w) > 0.4)
            res[gill] = P.SHADOW
        elif g.gill_slits > 1 and self.scale > 12:
            # Shark gill slits: a row of short upright lines, one pixel apart at least.
            gap = max(0.022, 2.0 * px_len)
            for i in range(int(g.gill_slits)):
                ga = g.eye_pos + 0.09 + i * gap
                if ga > g.peak:
                    break
                ga = ga + 0.01 * (1 - v ** 2)
                gill = (np.abs(a - ga) < px_len * 0.55) & (v > -0.45) & (v < 0.35) & (np.abs(w) > 0.4)
                res[gill] = P.SHADOW
        return res

    def _shade_wing(self, attr, light, mask, top, part) -> np.ndarray:
        """Wings show the back colours (and pattern) on top and the belly colour underneath."""
        c = self.genome.colors
        a = attr[..., 3][mask]
        r = attr[..., 4][mask]
        side = attr[..., 5][mask]
        li = light[mask]
        res = np.full(a.shape, P.BASE, dtype=np.uint8)
        res[li > 0.84] = P.HIGHLIGHT
        res[li < 0.18] = P.SHADOW
        if c.pattern in ("spots", "rings"):
            # Spots in (along, across) body lengths so they stay round on wide wings.
            span = max(self.genome.wings.span, self.genome.hammer, 0.05)
            res[self._spots(a * 1.6, side * r * span * 1.6, rings=c.pattern == "rings")] = P.ACCENT
        under = ~top[mask]
        res[under] = P.BELLY if c.belly_lightness > 0.25 else P.SHADOW
        res[under & (part[mask] == WING_FAR) & (li < 0.3)] = P.SHADOW
        return res

    def _spots(self, ua_coord, ub_coord, rings: bool = False) -> np.ndarray:
        """Hash-placed round spots on a 2D surface parameterisation."""
        c = self.genome.colors
        cells = 5 + int(c.pattern_scale * 6)
        ua, ub = ua_coord * cells + c.pattern_offset, ub_coord * cells
        ia, ib = np.floor(ua), np.floor(ub)
        h = np.sin(ia * 12.9898 + ib * 78.233 + self.genome.seed) * 43758.5453
        jit = np.mod(h, 1.0)
        fa, fb = ua - ia - 0.5, ub - ib - 0.5 + (jit - 0.5) * 0.4
        has = np.mod(h * 7.0, 1.0) < 0.5
        d2 = fa * fa + fb * fb
        if rings:
            # Hollow rings when there are pixels enough to show the hole, else solid spots.
            hole = 0.012 if self.scale * 0.6 / cells > 3.0 else -1.0
            return has & (d2 < 0.06) & (d2 > hole)
        return has & (d2 < 0.05)

    def _stripes(self, a) -> np.ndarray:
        c = self.genome.colors
        count = 3 + int(c.pattern_scale * 4)
        return np.mod((a - c.pattern_offset) * count, 1.0) < 0.32

    def _pattern(self, a, v, w, mask, pose) -> np.ndarray:
        """Pattern colour slot per body pixel (0 = no pattern there)."""
        res = np.zeros(a.shape, dtype=np.uint8)
        slots = self._pattern_slots(a, v, w, mask, pose)
        if isinstance(slots, tuple):
            res[slots[0]] = P.ACCENT
            res[slots[1]] = P.ACCENT2
        else:
            res[slots] = P.ACCENT
        return res

    def _pattern_slots(self, a, v, w, mask, pose):
        """A mask for the accent colour, or (accent mask, accent2 mask)."""
        g = self.genome
        c = g.colors
        kind = c.pattern
        eye = g.eye_pos
        # Leave the face clean (cephalopods have their face behind the mantle).
        body = (a > eye + 0.08) | (a < eye - 0.25)
        if kind == "stripes":
            face = np.ones(a.shape, dtype=bool) if c.pattern_on_fins else body
            return face & (a < 0.95) & self._stripes(a) & (v > -0.55)
        if kind == "bands":
            # Alternating bars in the two pattern colours.
            count = 3 + int(c.pattern_scale * 4)
            u = np.mod((a - c.pattern_offset) * count, 1.0)
            keep = body & (a < 0.95) & (v > -0.6)
            return keep & (u < 0.22), keep & (u > 0.5) & (u < 0.72)
        if kind == "shimmer":
            # Iridescent bands of colour rippling down the body ("passing clouds"), moving with the swim cycle.
            count = 2 + c.pattern_scale * 3
            u = np.sin(2 * math.pi * (a * count + 0.35 * v + 0.2 * np.sin(4.0 * v + 3.0 * a) - pose.phase))
            keep = (v > -0.75) & (a < 0.98)
            return keep & (u > 0.55), keep & (u < -0.6)
        if kind == "neon":
            # Neon tetra: electric blue line from the eye to the tail, red over the lower rear half.
            blue = (a > eye + 0.03) & (a < 0.97) & (v > 0.02) & (v < 0.3)
            red = (a > 0.45) & (a < 0.97) & (v <= 0.02) & (v > -0.75)
            return blue, red
        if kind == "whale_shark":
            # Pale spots in rows between thin pale vertical lines, on the back and flanks.
            ang = np.arctan2(v, w) / math.pi
            cells = 10 + int(c.pattern_scale * 6)
            lines = (np.mod(a * cells + c.pattern_offset, 1.0) < 0.12) & (a > eye + 0.1)
            ridges = (np.abs(np.mod(ang * 5.0, 1.0) - 0.5) < 0.06) & (a > 0.25)
            fa = np.mod(a * cells + c.pattern_offset, 1.0) - 0.56
            fb = np.mod(ang * 5.0, 1.0) - 0.25
            spot = (fa * fa + (fb * 1.1) ** 2 < 0.045)
            head_spots = (a < eye + 0.1) & self._spots(a * 2.0, ang * 1.2)
            keep = (v > -0.35) & (a < 0.97)
            return keep & (spot | head_spots), keep & (lines | ridges)
        if kind == "mackerel":
            # Wavy dark bars over the upper back only, and a fine line along the flank.
            count = 10 + int(c.pattern_scale * 6)
            u = (a - c.pattern_offset) * count
            wobble = 0.3 * np.sin(v * 10.0 + np.floor(u) * 1.7)
            bars = (np.mod(u + wobble + 0.6 * v, 1.0) < 0.38) & (v > 0.2) & (a > eye + 0.12) & (a < 0.97)
            line = (np.abs(v - 0.05) < 0.06) & (a > 0.25) & (a < 0.97)
            return bars, line
        if kind in ("spots", "rings"):
            # Spots live on the surface: use (a, angle) so they wrap around the body.
            ang = np.arctan2(v, w) / math.pi  # -1..1
            return body & self._spots(a, ang * 0.6, rings=kind == "rings") & (v > -0.5)
        if kind == "tiger":
            # Broken dark bars over the back, fading out towards the belly and the tail.
            count = 6 + int(c.pattern_scale * 8)
            u = (a - c.pattern_offset) * count
            wobble = 0.12 * np.sin(v * 5.0 + np.floor(u) * 2.3)
            bars = np.mod(u + wobble, 1.0) < 0.3
            return body & bars & (a > 0.25) & (a < 0.98) & (v > 0.25 - 0.35 * a)
        if kind == "orca":
            # White eye patch, white chin and belly that narrows between the flippers, then the
            # white flank sweep rising up behind the dorsal fin; a grey saddle behind the fin.
            e = ((a - (eye + 0.12)) / 0.075) ** 2 + ((v - 0.32) / 0.17) ** 2 < 1.0
            o = c.pattern_offset * 0.04
            edge = np.interp(a, [0.0, 0.08, 0.2, 0.32, 0.42, 0.5, 0.58, 0.66, 0.74, 0.8, 0.86],
                             [-0.25, -0.12, -0.35, -0.65, -0.75, -0.45, 0.05 + o, 0.12, -0.15, -0.6, -1.2])
            white = (v < edge) & (a < 0.86)
            d = g.dorsal
            saddle = (a > d.start + 0.02) & (a < d.end + 0.14) & (v > 0.55 + 0.25 * np.abs(a - d.end - 0.05) / 0.1)
            return (e & (np.abs(w) > 0.2)) | white, saddle
        if kind == "lateral":
            return body & (v > -0.1 - 0.2 * c.pattern_scale) & (v < 0.15)
        if kind == "saddle":
            return (v > 0.45 - 0.3 * c.pattern_scale) & (a > 0.05)
        if kind == "gradient":
            # Dithered fade towards the tail: checkerboard in the transition band.
            ys, xs = np.nonzero(mask)
            checker = ((ys + xs) % 2) == 0
            start = 0.45 + 0.2 * c.pattern_offset
            return (a > start + 0.12) | ((a > start) & checker)
        return np.zeros(a.shape, dtype=bool)

    def _fin_rays(self, out, part, attr):
        """At larger sizes, hint at fin rays with alternating dark streaks."""
        if self.scale < 40 or self.genome.kind != "fish":  # sharks and whales have smooth fins
            return
        rays = (part == CAUDAL) | np.isin(part, FIN_PARTS)
        m, r = attr[..., 3], attr[..., 4]
        count = 7 if self.scale < 70 else 11
        streak = rays & (np.mod(m * count, 1.0) < 0.22) & (r > 0.3) & (out == P.FIN)
        out[streak] = P.FIN_DARK

    def _draw_spines(self, out, depth, lines):
        """Urchin spines: 1px depth-tested lines, light where they point up or at the viewer."""
        if not len(lines):
            return
        s = self.scale
        h, w = out.shape
        scr = np.stack([self.width / 2 + lines[..., 0] * s, self.height / 2 - lines[..., 1] * s, lines[..., 2] * s], -1)
        d = lines[:, 1] - lines[:, 0]
        slot = np.where((d[:, 1] * 0.6 + d[:, 2]) / (np.linalg.norm(d, axis=1) + 1e-9) > -0.1, P.FIN, P.FIN_DARK)
        span = np.abs(scr[:, 1, :2] - scr[:, 0, :2]).max()
        t = np.linspace(0.0, 1.0, int(span * 1.5) + 2)[None, :, None]
        pts = scr[:, None, 0] + (scr[:, None, 1] - scr[:, None, 0]) * t  # (k, steps, 3)
        ix, iy = np.floor(pts[..., 0]).astype(int), np.floor(pts[..., 1]).astype(int)
        z = pts[..., 2]
        sl = np.broadcast_to(slot[:, None], z.shape)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix, iy, z, sl = ix[ok], iy[ok], z[ok], sl[ok]
        vis = z > depth[iy, ix] - 0.5
        ix, iy, z, sl = ix[vis], iy[vis], z[vis], sl[vis]
        if not len(z):
            return
        # Nearest spine sample wins each pixel.
        pix = iy * w + ix
        order = np.lexsort((z, pix))
        last = np.ones(len(order), dtype=bool)
        last[:-1] = pix[order][1:] != pix[order][:-1]
        win = order[last]
        out[iy[win], ix[win]] = sl[win]
        depth[iy[win], ix[win]] = z[win]

    def _draw_mouth(self, out, depth, pose):
        """A dark mouth line along the jaw; with teeth, a zigzag of white teeth along it."""
        if self.genome.mouth <= 0:
            return
        mouths, facing = self.model.mouth_lines(pose)
        s = self.scale
        cx, cy = self.width / 2, self.height / 2
        h, w = out.shape
        for pts, ok in zip(mouths, facing):
            if not ok:
                continue
            pix: list[tuple[int, int]] = []
            for p0, p1 in zip(pts[:-1], pts[1:]):
                x0, y0, z0 = cx + p0[0] * s, cy - p0[1] * s, p0[2] * s
                x1, y1, z1 = cx + p1[0] * s, cy - p1[1] * s, p1[2] * s
                for t in np.linspace(0.0, 1.0, int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2):
                    ix, iy = int(math.floor(x0 + (x1 - x0) * t)), int(math.floor(y0 + (y1 - y0) * t))
                    if not (0 <= ix < w and 0 <= iy < h) or out[iy, ix] == P.TRANSPARENT:
                        continue
                    if z0 + (z1 - z0) * t < depth[iy, ix] - 1.5 or (pix and pix[-1] == (ix, iy)):
                        continue
                    if (ix, iy) not in pix:
                        pix.append((ix, iy))
            if len(pix) < 3:
                continue
            if not self.genome.teeth:
                for ix, iy in pix:
                    out[iy, ix] = P.OUTLINE
                continue
            # An open jaw: dark gape two pixels tall with interlocking upper and lower teeth
            # (a zigzag of white), edged top and bottom with a dark lip line.
            body = out != P.TRANSPARENT
            for k, (ix, iy) in enumerate(pix):
                rows = ((iy - 2, P.OUTLINE), (iy - 1, P.EYE if k % 2 else P.OUTLINE),
                        (iy, P.OUTLINE if k % 2 else P.EYE), (iy + 1, P.OUTLINE))
                if k == 0:  # the corner of the mouth at the front stays a plain line
                    rows = ((iy - 1, P.OUTLINE), (iy, P.OUTLINE))
                for ty, slot in rows:
                    if 0 <= ty < h and body[ty, ix]:
                        out[ty, ix] = slot

    def _draw_eyes(self, out, depth, eyes, axes, pose):
        if not self.model.eyes:
            return
        s = self.scale
        cx, cy = self.width / 2, self.height / 2
        r = self.model.eyes[0].radius * s
        h, w = out.shape
        if self.genome.eye_style == "dark":
            # Whales, sharks, rays: a plain dark eye, no white.
            for e, ax in zip(eyes, axes):
                if ax[2] < -0.2:
                    continue
                ex, ey, ez = cx + e[0] * s, cy - e[1] * s, e[2] * s
                ix, iy = int(math.floor(ex)), int(math.floor(ey))
                if not (0 <= ix < w and 0 <= iy < h) or depth[iy, ix] > ez + max(1.0, r):
                    continue
                rr = max(0.5, min(r, 3.0) * 0.6)
                for yy in range(iy - 3, iy + 4):
                    for xx in range(ix - 3, ix + 4):
                        if 0 <= xx < w and 0 <= yy < h and out[yy, xx] != P.TRANSPARENT \
                                and math.hypot(xx + 0.5 - ex, yy + 0.5 - ey) <= rr + 0.25:
                            out[yy, xx] = P.OUTLINE
            return
        for e, ax in zip(eyes, axes):
            if ax[2] < -0.2:  # facing away from the camera
                continue
            ex, ey, ez = cx + e[0] * s, cy - e[1] * s, e[2] * s
            ix, iy = int(math.floor(ex)), int(math.floor(ey))
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            if depth[iy, ix] > ez + max(1.0, r):  # hidden behind something
                continue
            # Pupil looks forward: towards the head direction on screen.
            fwd = math.cos(pose.yaw)
            if r < 1.1:
                # Tiny eye: a dark pupil with one white pixel behind it, looking forward.
                out[iy, ix] = P.OUTLINE
                bx = ix - (1 if fwd > 0.3 else -1 if fwd < -0.3 else 0)
                if bx != ix and 0 <= bx < w and out[iy, bx] not in (P.TRANSPARENT,) and r > 0.6:
                    out[iy, bx] = P.EYE
                continue
            rr = min(r, 3.5)
            for yy in range(iy - 4, iy + 5):
                for xx in range(ix - 4, ix + 5):
                    if not (0 <= xx < w and 0 <= yy < h) or out[yy, xx] == P.TRANSPARENT:
                        continue
                    d = math.hypot(xx + 0.5 - ex, yy + 0.5 - ey)
                    if d <= rr * 0.8:
                        out[yy, xx] = P.EYE
            # Pupil: one pixel (two/four at bigger sizes), nudged forward.
            px = ex + fwd * rr * 0.35
            pr = max(0.5, rr * 0.45)
            for yy in range(iy - 3, iy + 4):
                for xx in range(ix - 3, ix + 4):
                    if not (0 <= xx < w and 0 <= yy < h) or out[yy, xx] == P.TRANSPARENT:
                        continue
                    if math.hypot(xx + 0.5 - px, yy + 0.5 - ey) <= pr + 0.25:
                        out[yy, xx] = P.OUTLINE
            if r >= 2.6:  # ring the eye at large sizes
                for yy in range(iy - 5, iy + 6):
                    for xx in range(ix - 5, ix + 6):
                        if not (0 <= xx < w and 0 <= yy < h) or out[yy, xx] in (P.TRANSPARENT, P.EYE, P.OUTLINE):
                            continue
                        d = math.hypot(xx + 0.5 - ex, yy + 0.5 - ey)
                        if rr * 0.8 < d <= rr * 0.8 + 1.0:
                            out[yy, xx] = P.SHADOW

    def _outline(self, out: np.ndarray):
        filled = out != P.TRANSPARENT
        nb = np.zeros(out.shape, dtype=bool)
        for dy, dx in N4:
            nb |= _shift(filled, dy, dx, fill=False)
        out[(~filled) & nb] = P.OUTLINE

    # ------------------------------------------------------------------
    def to_rgba(self, indexed: np.ndarray) -> np.ndarray:
        lut = np.array([self.palette.rgba(i) for i in range(len(self.palette.colors))], dtype=np.uint8)
        return lut[indexed]

    def render(self, pose: Pose) -> np.ndarray:
        """Render a pose to an (h, w, 4) uint8 RGBA array."""
        return self.to_rgba(self.render_indexed(pose))
