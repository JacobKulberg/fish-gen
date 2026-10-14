"""Tiny z-buffered triangle rasterizer that samples exactly at pixel centres.

Sampling once per pixel (no anti-aliasing) is what keeps the output crisp
pixel art rather than a blurry downscale.
"""

from __future__ import annotations

import numpy as np

EMPTY_DEPTH = -1e9


def rasterize(
    screen: np.ndarray,  # (n, 3): x px, y px (down), depth (larger = closer)
    attrs: np.ndarray,  # (n, k)
    tris: np.ndarray,  # (t, 3)
    tri_part: np.ndarray,  # (t,)
    width: int,
    height: int,
):
    """Return (part (h,w) int, depth (h,w), attr (h,w,k))."""
    k = attrs.shape[1]
    part_buf = np.zeros(height * width, dtype=np.int32)
    depth_buf = np.full(height * width, EMPTY_DEPTH)
    attr_buf = np.zeros((height * width, k))

    v0, v1, v2 = screen[tris[:, 0]], screen[tris[:, 1]], screen[tris[:, 2]]
    area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])

    # Integer range of pixel centres (i + 0.5) inside each triangle's bounding box.
    lo = np.minimum(np.minimum(v0, v1), v2)
    hi = np.maximum(np.maximum(v0, v1), v2)
    x0 = np.clip(np.ceil(lo[:, 0] - 0.5), 0, width).astype(np.int64)
    x1 = np.clip(np.floor(hi[:, 0] - 0.5) + 1, 0, width).astype(np.int64)
    y0 = np.clip(np.ceil(lo[:, 1] - 0.5), 0, height).astype(np.int64)
    y1 = np.clip(np.floor(hi[:, 1] - 0.5) + 1, 0, height).astype(np.int64)
    bw, bh = np.maximum(x1 - x0, 0), np.maximum(y1 - y0, 0)
    count = bw * bh
    count[np.abs(area) < 1e-12] = 0
    total = int(count.sum())
    if total == 0:
        return part_buf.reshape(height, width), depth_buf.reshape(height, width), attr_buf.reshape(height, width, k)

    # Expand to one row per (triangle, candidate pixel) pair.
    tri = np.repeat(np.arange(len(tris)), count)
    start = np.cumsum(count) - count
    local = np.arange(total) - np.repeat(start, count)
    bw_r = bw[tri]
    px = x0[tri] + local % bw_r
    py = y0[tri] + local // bw_r
    fx, fy = px + 0.5, py + 0.5

    a0, a1, a2 = v0[tri], v1[tri], v2[tri]
    inv = 1.0 / area[tri]
    w0 = ((a1[:, 0] - fx) * (a2[:, 1] - fy) - (a1[:, 1] - fy) * (a2[:, 0] - fx)) * inv
    w1 = ((a2[:, 0] - fx) * (a0[:, 1] - fy) - (a2[:, 1] - fy) * (a0[:, 0] - fx)) * inv
    w2 = 1.0 - w0 - w1
    eps = -1e-7
    inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)

    tri, pix = tri[inside], (py * width + px)[inside]
    w0, w1, w2 = w0[inside], w1[inside], w2[inside]
    z = w0 * a0[inside, 2] + w1 * a1[inside, 2] + w2 * a2[inside, 2]

    # Z-test: for each pixel keep the closest fragment (largest z).
    order = np.lexsort((z, pix))
    last = np.ones(len(order), dtype=bool)
    last[:-1] = pix[order][1:] != pix[order][:-1]
    win = order[last]

    t, p = tri[win], pix[win]
    depth_buf[p] = z[win]
    part_buf[p] = tri_part[t]
    attr_buf[p] = (
        w0[win, None] * attrs[tris[t, 0]] + w1[win, None] * attrs[tris[t, 1]] + w2[win, None] * attrs[tris[t, 2]]
    )
    return part_buf.reshape(height, width), depth_buf.reshape(height, width), attr_buf.reshape(height, width, k)
