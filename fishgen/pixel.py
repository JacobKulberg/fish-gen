"""Turns the realistic renders into pixel art.

The 3D renderer draws each frame with smooth lighting, anti-aliased edges and millions of
colours. This module takes all the frames of one sprite and makes them pixel art:

* hard edges: every pixel is either fully opaque or fully transparent
* one small palette shared by every frame (so colours don't flicker between frames),
  picked by k-means in the perceptual OKLab space, with a little extra contrast and
  colour first so the shading reads at small sizes
* lone pixels that only add noise are merged into the colour around them
* a selective outline: a dark, hue-matched rim round the silhouette that takes its colour
  from the pixels it borders, as pixel artists draw it
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
N8 = N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def auto_colors(width: int, height: int) -> int:
    """A palette size that suits the sprite: 10 colours at 16 px up to 24 at 128 px."""
    return int(np.clip(round(8 + max(width, height) / 8), 8, 32))


# ---------------------------------------------------------------------------- colour space
def _to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb8(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.round(s * 255).astype(np.uint8)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(rgb8: np.ndarray) -> np.ndarray:
    lms = _to_linear(rgb8.astype(np.float64)) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ np.linalg.inv(_M2).T) ** 3
    return _to_srgb8(lms @ np.linalg.inv(_M1).T)


# ---------------------------------------------------------------------------- helpers
def _shift(arr: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """out[y, x] = arr[y + dy, x + dx] (out of range -> fill)."""
    out = np.full_like(arr, fill)
    h, w = arr.shape[:2]
    out[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] = \
        arr[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
    return out


def kmeans(points: np.ndarray, weights: np.ndarray, k: int, iters: int = 40, seed: int = 0) -> np.ndarray:
    """Weighted k-means with k-means++ seeding (deterministic for a given seed)."""
    k = min(k, len(points))
    rng = np.random.default_rng(seed)
    centres = [points[np.argmax(weights)]]
    d2 = ((points - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 * weights
        if p.sum() <= 0:
            break
        centres.append(points[rng.choice(len(points), p=p / p.sum())])
        d2 = np.minimum(d2, ((points - centres[-1]) ** 2).sum(1))
    c = np.array(centres)
    for _ in range(iters):
        lab = np.argmin(((points[:, None] - c[None]) ** 2).sum(-1), axis=1)
        new = c.copy()
        for j in range(len(c)):
            m = lab == j
            if m.any():
                new[j] = np.average(points[m], axis=0, weights=weights[m])
        if np.allclose(new, c, atol=1e-5):
            break
        c = new
    return c


def _unique_colours(lab_pixels: np.ndarray, quant: float = 0.006):
    """Bucket near-identical colours so k-means runs on thousands of points, not millions."""
    key = np.round(lab_pixels / quant).astype(np.int64)
    uniq, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inverse.ravel(), lab_pixels)
    return sums / counts[:, None], counts, inverse.ravel()


# ---------------------------------------------------------------------------- main
def pixelate(frames: list[np.ndarray], colors: int | None = None, outline: bool = True,
             alpha_cut: float = 0.5, punch: float = 1.0,
             eyes: list[list[tuple[float, float, float]]] | None = None,
             mouths: list[np.ndarray | None] | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """Make a sprite's RGBA frames pixel art. Returns (frames, palette (n, 3) uint8).

    ``colors`` is the palette size for the animal itself (None picks one for the sprite
    size); the outline adds up to three dark shades on top. ``punch`` scales the extra
    contrast and colour given to the shading before it is quantised (0 = none). ``eyes`` lists
    each frame's visible eyes as (x, y, radius) so small eyes still get a clear pupil pixel;
    ``mouths`` gives each frame's maw and teeth coverage (h, w, 2) or None, so a gaping mouth
    comes out as a dark gap with crisp white teeth.
    """
    if not frames:
        return frames, np.zeros((0, 3), np.uint8)
    h, w = frames[0].shape[:2]
    exact = colors is not None  # a palette size asked for is kept to
    colors = colors or auto_colors(w, h)
    stack = np.stack(frames)
    solid = stack[..., 3] >= alpha_cut * 255
    solid = np.stack([_tidy_mask(m) for m in solid])

    lab_img = _see_through(to_oklab(stack[..., :3]), stack[..., 3] / 255.0)
    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as ex:  # numpy runs these outside the GIL
        lab_img = np.stack(list(ex.map(_smooth, lab_img, solid)))
    lab = _punch(lab_img[solid], punch)
    pts, counts, inverse = _unique_colours(lab)
    # sqrt weights keep small, distinct details (eyes, stripes) from being outvoted by big flat areas.
    centres = kmeans(pts, np.sqrt(counts), colors)
    mouth_cols = _mouth_colours(lab_img, mouths)
    extra = list(mouth_cols)
    if eyes is not None and any(r < 2.0 for frame in eyes for _, _, r in frame) and centres[:, 0].min() > 0.2:
        # A pale or bright animal has no dark enough colour for a pupil pixel: give it one, a
        # very dark shade of its own darkest colour.
        dark = centres[np.argmin(centres[:, 0])].copy()
        extra.append(np.array([0.15, dark[1] * 0.4, dark[2] * 0.4]))
    for c in extra:  # a small maw, a few teeth or a pupil would never win a colour of their own
        dist = np.linalg.norm(centres - c, axis=1)
        if dist.min() > 0.06:
            if exact:  # then it takes the place of the nearest colour instead
                centres[np.argmin(dist)] = c
            else:
                centres = np.vstack([centres, c])
    centres = centres[np.argsort(centres[:, 0])]  # dark to light
    idx_u = _nearest(pts, centres)
    index = np.full(solid.shape, -1, dtype=np.int32)
    index[solid] = idx_u[inverse]
    index = np.stack([_despeckle(f, centres) for f in index])
    if mouths is not None:
        maw_i, tooth_i = (int(_nearest(c[None], centres)[0]) for c in mouth_cols[:2]) if mouth_cols else (0, 0)
        for k, (f, mouth) in enumerate(zip(index, mouths)):
            if mouth is not None:
                _mark_mouth(f, mouth, maw_i, tooth_i, eyes[k] if eyes is not None else [])
    if eyes is not None:
        for k, (f, frame_eyes) in enumerate(zip(index, eyes)):
            _mark_eyes(f, frame_eyes, centres, mouths[k] if mouths is not None else None)

    palette_lab = centres
    if outline:
        index, palette_lab = _outline(index, palette_lab)

    pal8 = from_oklab(palette_lab)
    out = []
    for f in index:
        rgba = np.zeros((h, w, 4), np.uint8)
        on = f >= 0
        rgba[on, :3] = pal8[f[on]]
        rgba[on, 3] = 255
        out.append(rgba)
    return out, pal8


def _nearest(pts: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.argmin(((pts[:, None] - centres[None]) ** 2).sum(-1), axis=1).astype(np.int32)


def _see_through(lab: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pixel art has no half-transparent pixels, so thin translucent fins are drawn the way a
    pixel artist would: paler and softer in colour than the body, by how see-through they are."""
    clear = (1.0 - np.clip(alpha, 0.0, 1.0))[..., None]
    out = lab.copy()
    out[..., :1] += (0.92 - out[..., :1]) * clear * 0.45
    out[..., 1:] *= 1.0 - 0.45 * clear
    return out


def _punch(lab: np.ndarray, amount: float) -> np.ndarray:
    """Spread the lightness range and lift chroma a little: shading that is subtle at full
    resolution would otherwise collapse into one flat colour at 16-64 px."""
    if amount <= 0 or not len(lab):
        return lab
    out = lab.copy()
    L = out[:, 0]
    mid = np.median(L)
    out[:, 0] = np.clip(mid + (L - mid) * (1 + 0.15 * amount), 0.0, 1.0)
    out[:, 1:] *= 1 + 0.06 * amount
    return out


def _smooth(lab: np.ndarray, mask: np.ndarray, sigma: float = 0.045, passes: int = 2) -> np.ndarray:
    """Edge-preserving blur (a small bilateral filter): irons out fine skin texture and noise,
    which would only turn into speckle, while keeping stripes, spots, eyes and shading edges."""
    m = mask.astype(np.float64)
    for _ in range(passes):
        acc = lab * m[..., None]
        wsum = m.copy()
        for dy, dx in N8:
            nb = _shift(lab, dy, dx)
            nm = _shift(m, dy, dx)
            d2 = ((nb - lab) ** 2).sum(-1)
            w = nm * np.exp(-d2 / (2 * sigma * sigma)) * (0.6 if dy and dx else 1.0)
            acc += nb * w[..., None]
            wsum += w
        lab = np.where(mask[..., None], acc / np.maximum(wsum, 1e-9)[..., None], lab)
    return lab


def _tidy_mask(m: np.ndarray) -> np.ndarray:
    """Drop pixels with no opaque neighbour at all, and fill 1-px pinholes."""
    m = m.copy()
    n8 = sum(_shift(m.astype(np.int8), dy, dx) for dy, dx in N8)
    m &= n8 > 0
    n4 = sum(_shift(m.astype(np.int8), dy, dx) for dy, dx in N4)
    m |= n4 == 4
    return m


def _despeckle(index: np.ndarray, centres: np.ndarray, max_de: float = 0.09) -> np.ndarray:
    """Merge a pixel into its surroundings when all four neighbours share one other colour that
    is close to it. Strong contrasts (an eye, a spot) are kept."""
    out = index.copy()
    nb = np.stack([_shift(index, dy, dx, fill=-1) for dy, dx in N4])
    same = (nb == nb[0]).all(0) & (nb[0] >= 0) & (index >= 0) & (nb[0] != index)
    ys, xs = np.nonzero(same)
    if len(ys):
        a, b = index[ys, xs], nb[0][ys, xs]
        close = np.linalg.norm(centres[a] - centres[b], axis=1) < max_de
        out[ys[close], xs[close]] = b[close]
    return out


def _mark_eyes(index: np.ndarray, eyes, centres: np.ndarray, mouth: np.ndarray | None = None):
    """An eye a few pixels across survives the downscale; a smaller one blurs into the skin.
    Give those a pupil pixel, the way a pixel artist would: dark on a pale head, and on a head
    so dark that a dark pixel wouldn't show, a light glint instead."""
    h, w = index.shape
    darkest = 0  # the palette is sorted dark to light
    for x, y, r in eyes:
        if r >= 2.0:
            continue
        ix, iy = int(np.floor(x)), int(np.floor(y))
        if not (0 <= ix < w and 0 <= iy < h) or index[iy, ix] < 0:
            continue
        around = index[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2]
        around = around[around >= 0]
        skin = np.median(centres[around, 0])
        if skin - centres[darkest, 0] > 0.15:
            index[iy, ix] = darkest
            if mouth is not None:
                _skin_ring(index, ix, iy, centres, mouth[..., 0] > 0.1)
        else:
            index[iy, ix] = int(np.argmin(np.abs(centres[:, 0] - min(1.0, skin + 0.4))))


def _skin_ring(index: np.ndarray, ix: int, iy: int, centres: np.ndarray, mouth: np.ndarray):
    """Keep skin between a one-pixel pupil and a gaping mouth: dark mouth pixels right next to
    it would swallow it into one dark blob, so they take the skin colour from nearby."""
    h, w = index.shape
    win = index[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
    pupil = centres[index[iy, ix], 0]
    light = win[(win >= 0)]
    light = light[centres[light, 0] > pupil + 0.15]
    if not len(light):
        return
    skin = np.bincount(light).argmax()
    for dy, dx in N8:
        y, x = iy + dy, ix + dx
        if 0 <= y < h and 0 <= x < w and index[y, x] >= 0 and mouth[y, x] and \
                centres[index[y, x], 0] < pupil + 0.12:
            index[y, x] = skin


def _mouth_colours(lab: np.ndarray, mouths) -> list[np.ndarray]:
    """The colours of a gaping maw (made dark enough to read as a hole) and of its teeth
    (bright enough to stand out), or [] when no frame shows a mouth."""
    if not mouths or all(m is None for m in mouths):
        return []
    maws, teeth = [], []
    for img, m in zip(lab, mouths):
        if m is not None:
            maws.append(img[(m[..., 0] >= 0.6) & (m[..., 1] < 0.2)])
            teeth.append(img[m[..., 1] >= 0.5])
    maw, tooth = np.concatenate(maws), np.concatenate(teeth)
    if not len(maw):
        return []
    maw = maw.mean(0)
    maw[0] = min(maw[0], 0.3)
    tooth = tooth.mean(0) if len(tooth) else np.array([0.95, 0.0, 0.01])
    tooth[0] = max(tooth[0], 0.88)
    # A near-black for pupils, darker than the maw, so an eye by the mouth isn't lost in it.
    pupil = np.array([min(0.15, maw[0] - 0.12), 0.0, 0.0])
    return [maw, tooth, pupil]


def _mark_mouth(index: np.ndarray, mouth: np.ndarray, maw_i: int, tooth_i: int, eyes=()):
    """A gaping mouth drawn the pixel-art way: a dark maw with every tooth a pixel of white,
    which a downscale alone would smear into a pinkish blur. It never paints over an eye, nor
    the pixels right round one, so the eye can't merge into the mouth."""
    on = index >= 0
    yy, xx = np.mgrid[:index.shape[0], :index.shape[1]]
    for x, y, r in eyes:
        on &= np.maximum(np.abs(xx + 0.5 - x), np.abs(yy + 0.5 - y)) > max(r, 1.0) + 0.6
    maw, teeth = mouth[..., 0], mouth[..., 1]
    white = on & (maw >= 0.3) & (teeth >= 0.3)
    # Teeth stand apart: of a run of white pixels along the jaw, only the first stays white.
    white &= ~_shift(white, 0, -1, fill=False)
    index[on & (maw >= 0.45)] = maw_i
    index[white] = tooth_i


def _outline(index: np.ndarray, palette: np.ndarray, shades: int = 3):
    """Selective outline: each transparent pixel touching the sprite becomes a dark version of
    the colours it borders. The outline colours are reduced to a few shades."""
    frames = []
    cols = []
    for f in index:
        on = f >= 0
        acc = np.zeros(f.shape + (3,))
        n = np.zeros(f.shape)
        for dy, dx in N4:
            nf = _shift(f, dy, dx, fill=-1)
            hit = nf >= 0
            acc[hit] += palette[nf[hit]]
            n += hit
        rim = (~on) & (n > 0)
        frames.append(rim)
        cols.append(acc[rim] / n[rim][:, None])
    allc = np.concatenate(cols) if cols else np.zeros((0, 3))
    if not len(allc):
        return index, palette
    dark = allc.copy()
    dark[:, 0] = np.minimum(dark[:, 0] * 0.42, 0.3)  # much darker...
    dark[:, 1:] *= 0.45  # ...and greyer, as dark colours are (and stay inside the RGB range)
    dark[:, 2] -= 0.012  # with a nudge towards blue, the classic hue-shifted shadow
    pts, counts, inverse = _unique_colours(dark)
    oc = kmeans(pts, counts.astype(float), min(shades, len(pts)))
    oi = _nearest(pts, oc)[inverse] + len(palette)
    out = index.copy()
    start = 0
    for k, rim in enumerate(frames):
        cnt = int(rim.sum())
        out[k][rim] = oi[start:start + cnt]
        start += cnt
    return out, np.concatenate([palette, oc])
