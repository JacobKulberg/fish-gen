"""Tiny z-buffered triangle rasterizer that samples exactly at pixel centres.

The renderer calls it at a supersampled resolution and averages blocks of samples
down afterwards, which is what anti-aliases the silhouette and every marking.
Triangles are processed in chunks so big sprites (whales) stay within memory.
"""

from __future__ import annotations

import numpy as np

from . import native

EMPTY_DEPTH = -1e9
CHUNK = 3_000_000  # candidate (triangle, pixel) pairs per batch


def rasterize(
    screen: np.ndarray,  # (n, 3): x px, y px (down), depth (larger = closer)
    attrs: np.ndarray,  # (n, k)
    tris: np.ndarray,  # (t, 3)
    tri_part: np.ndarray,  # (t,)
    width: int,
    height: int,
):
    """Return (part (h,w) int, depth (h,w), attr (h,w,k))."""
    if native.lib is not None:
        return _rasterize_c(screen, attrs, tris, tri_part, width, height)
    k = attrs.shape[1]
    part_buf = np.zeros(height * width, dtype=np.int32)
    depth_buf = np.full(height * width, EMPTY_DEPTH)
    tri_buf = np.full(height * width, -1, dtype=np.int64)
    bary_buf = np.zeros((height * width, 2))

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

    live = np.nonzero(count)[0]
    if len(live):
        # Split the live triangles into batches of roughly CHUNK candidate pixels.
        cum = np.cumsum(count[live])
        cuts = np.searchsorted(cum, np.arange(CHUNK, cum[-1], CHUNK))
        for batch in np.split(live, cuts):
            if len(batch):
                _batch(batch, count, x0, y0, bw, v0, v1, v2, area, width, part_buf, depth_buf, tri_buf, bary_buf,
                       tri_part)

    attr_buf = np.zeros((height * width, k))
    hit = tri_buf >= 0
    t = tri_buf[hit]
    b0, b1 = bary_buf[hit, 0], bary_buf[hit, 1]
    attr_buf[hit] = (b0[:, None] * attrs[tris[t, 0]] + b1[:, None] * attrs[tris[t, 1]]
                     + (1.0 - b0 - b1)[:, None] * attrs[tris[t, 2]])
    return part_buf.reshape(height, width), depth_buf.reshape(height, width), attr_buf.reshape(height, width, k)


def _batch(sel, count, x0, y0, bw, v0, v1, v2, area, width, part_buf, depth_buf, tri_buf, bary_buf, tri_part):
    c = count[sel]
    total = int(c.sum())
    tri = np.repeat(sel, c)
    start = np.cumsum(c) - c
    local = np.arange(total) - np.repeat(start, c)
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

    # Z-test: for each pixel keep the closest fragment (largest z), then merge with the buffer.
    order = np.lexsort((z, pix))
    last = np.ones(len(order), dtype=bool)
    last[:-1] = pix[order][1:] != pix[order][:-1]
    win = order[last]
    p = pix[win]
    closer = z[win] > depth_buf[p]
    win, p = win[closer], p[closer]
    depth_buf[p] = z[win]
    part_buf[p] = tri_part[tri[win]]
    tri_buf[p] = tri[win]
    bary_buf[p, 0] = w0[win]
    bary_buf[p, 1] = w1[win]


def _rasterize_c(screen, attrs, tris, tri_part, width, height):
    """Same as ``rasterize``, in C (fishgen/native/fast.c)."""
    n = width * height
    k = attrs.shape[1]
    part = np.zeros(n, dtype=np.int32)
    depth = np.full(n, EMPTY_DEPTH)
    tri_id = np.full(n, -1, dtype=np.int32)
    b0, b1 = np.zeros(n), np.zeros(n)
    tris, attrs = native.i32(tris), native.f64(attrs)
    native.lib.fg_raster(native.ptr(native.f64(screen)), native.iptr(tris), native.iptr(native.i32(tri_part)),
                         len(tris), width, height, native.iptr(part), native.ptr(depth), native.iptr(tri_id),
                         native.ptr(b0), native.ptr(b1))
    attr = np.empty((n, k))
    native.lib.fg_interp(native.ptr(attrs), k, native.iptr(tris), native.iptr(tri_id), native.ptr(b0),
                         native.ptr(b1), n, native.ptr(attr))
    return part.reshape(height, width), depth.reshape(height, width), attr.reshape(height, width, k)
