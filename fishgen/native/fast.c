/* The renderer's hot loops in plain C, loaded with ctypes by fishgen/native/__init__.py.
 *
 * Each function mirrors a numpy implementation in fishgen (raster.py, render.py) and
 * produces the same results; the numpy versions remain as the fallback when no C
 * compiler is available. All arrays are C-contiguous doubles / int32s.
 */

#include <math.h>
#include <stdint.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#define EMPTY_DEPTH (-1e9)

/* Z-buffered triangle rasterizer sampling at pixel centres (see raster.py).
 * screen: (n,3) x, y (down), depth (larger = closer). Buffers come pre-cleared:
 * depth = EMPTY_DEPTH, tri_id = -1. For each covered pixel keeps the closest
 * fragment's part, depth, triangle and first two barycentrics. */
API void fg_raster(const double *screen, const int32_t *tris, const int32_t *tri_part, int64_t ntri,
                   int32_t width, int32_t height, int32_t *part, double *depth, int32_t *tri_id,
                   double *b0, double *b1) {
    const double eps = -1e-7;
    for (int64_t t = 0; t < ntri; t++) {
        const double *a0 = screen + 3 * (int64_t)tris[3 * t];
        const double *a1 = screen + 3 * (int64_t)tris[3 * t + 1];
        const double *a2 = screen + 3 * (int64_t)tris[3 * t + 2];
        double area = (a1[0] - a0[0]) * (a2[1] - a0[1]) - (a1[1] - a0[1]) * (a2[0] - a0[0]);
        if (fabs(area) < 1e-12) continue;
        double inv = 1.0 / area;
        double lox = fmin(fmin(a0[0], a1[0]), a2[0]), hix = fmax(fmax(a0[0], a1[0]), a2[0]);
        double loy = fmin(fmin(a0[1], a1[1]), a2[1]), hiy = fmax(fmax(a0[1], a1[1]), a2[1]);
        double fx0 = ceil(lox - 0.5), fx1 = floor(hix - 0.5) + 1;
        double fy0 = ceil(loy - 0.5), fy1 = floor(hiy - 0.5) + 1;
        int64_t x0 = fx0 < 0 ? 0 : (fx0 > width ? width : (int64_t)fx0);
        int64_t x1 = fx1 < 0 ? 0 : (fx1 > width ? width : (int64_t)fx1);
        int64_t y0 = fy0 < 0 ? 0 : (fy0 > height ? height : (int64_t)fy0);
        int64_t y1 = fy1 < 0 ? 0 : (fy1 > height ? height : (int64_t)fy1);
        int32_t pt = tri_part[t];
        for (int64_t py = y0; py < y1; py++) {
            double fy = py + 0.5;
            for (int64_t px = x0; px < x1; px++) {
                double fx = px + 0.5;
                double w0 = ((a1[0] - fx) * (a2[1] - fy) - (a1[1] - fy) * (a2[0] - fx)) * inv;
                double w1 = ((a2[0] - fx) * (a0[1] - fy) - (a2[1] - fy) * (a0[0] - fx)) * inv;
                double w2 = 1.0 - w0 - w1;
                if (w0 < eps || w1 < eps || w2 < eps) continue;
                double z = w0 * a0[2] + w1 * a1[2] + w2 * a2[2];
                int64_t p = py * width + px;
                if (z > depth[p]) {
                    depth[p] = z;
                    part[p] = pt;
                    tri_id[p] = (int32_t)t;
                    b0[p] = w0;
                    b1[p] = w1;
                }
            }
        }
    }
}

/* Interpolate per-vertex attributes (n,k) at the rasterized pixels. */
API void fg_interp(const double *attrs, int32_t k, const int32_t *tris, const int32_t *tri_id, const double *b0,
                   const double *b1, int64_t npix, double *out) {
    for (int64_t p = 0; p < npix; p++) {
        int32_t t = tri_id[p];
        double *o = out + p * k;
        if (t < 0) {
            for (int32_t j = 0; j < k; j++) o[j] = 0.0;
            continue;
        }
        const double *v0 = attrs + (int64_t)tris[3 * t] * k;
        const double *v1 = attrs + (int64_t)tris[3 * t + 1] * k;
        const double *v2 = attrs + (int64_t)tris[3 * t + 2] * k;
        double w0 = b0[p], w1 = b1[p], w2 = 1.0 - w0 - w1;
        for (int32_t j = 0; j < k; j++) o[j] = w0 * v0[j] + w1 * v1[j] + w2 * v2[j];
    }
}

/* Screen-space ambient occlusion over a depth buffer (see FishRenderer._ssao). */
API void fg_ssao(const double *depth, int32_t height, int32_t width, double s, double *occ) {
    double radii[2] = {fmax(2.0, 0.02 * s), fmax(4.0, 0.05 * s)};
    int n = 0;
    int64_t total = (int64_t)height * width;
    for (int64_t p = 0; p < total; p++) occ[p] = 0.0;
    for (int r = 0; r < 2; r++) {
        double rad = radii[r];
        for (int k = 0; k < 8; k++) {
            double ang = k * M_PI / 4 + rad;
            int dx = (int)nearbyint(cos(ang) * rad), dy = (int)nearbyint(sin(ang) * rad);
            for (int32_t y = 0; y < height; y++) {
                int32_t yy = y + dy;
                for (int32_t x = 0; x < width; x++) {
                    int32_t xx = x + dx;
                    double nd = (yy >= 0 && yy < height && xx >= 0 && xx < width) ? depth[(int64_t)yy * width + xx]
                                                                                 : EMPTY_DEPTH;
                    double diff = nd - depth[(int64_t)y * width + x];
                    if (diff >= 6 * rad) continue;
                    double v = (diff - 0.3 * rad) / (1.5 * rad);
                    occ[(int64_t)y * width + x] += v < 0 ? 0 : (v > 1 ? 1 : v);
                }
            }
            n++;
        }
    }
    for (int64_t p = 0; p < total; p++) occ[p] /= n;
}

static inline double hash2(double i, double j, double seed) {
    double h = sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453;
    return h - floor(h);
}

static inline double vnoise1(double x, double y, double seed) {
    double xi = floor(x), yi = floor(y);
    double xf = x - xi, yf = y - yi;
    double u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    double a = hash2(xi, yi, seed), b = hash2(xi + 1, yi, seed);
    double c = hash2(xi, yi + 1, seed), d = hash2(xi + 1, yi + 1, seed);
    return 2.0 * (a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v) - 1.0;
}

API void fg_hash(const double *i, const double *j, int64_t n, double seed, double *out) {
    for (int64_t p = 0; p < n; p++) out[p] = hash2(i[p], j[p], seed);
}

API void fg_fbm(const double *x, const double *y, int64_t n, double seed, int32_t octaves, double *out) {
    double norm = 0.0, amp = 1.0;
    for (int32_t k = 0; k < octaves; k++) {
        norm += amp;
        amp *= 0.5;
    }
    for (int64_t p = 0; p < n; p++) {
        double xx = x[p], yy = y[p], total = 0.0;
        amp = 1.0;
        for (int32_t k = 0; k < octaves; k++) {
            total += amp * vnoise1(xx, yy, seed + 13.0 * k);
            xx = xx * 2.03 + 1.7;
            yy = yy * 2.03 - 0.9;
            amp *= 0.5;
        }
        out[p] = total / norm;
    }
}

/* Urchin spines / pufferfish spikes as depth-tested tapering discs swept along each line
 * (see FishRenderer._draw_spines). Per pixel keeps the closest visible fragment's depth,
 * spine k, step t and offset (ox, oy); win_k comes pre-filled with -1, win_z with EMPTY. */
API void fg_spines(const double *p0, const double *p1, int64_t nk, int32_t steps, const double *rad, int32_t R,
                   const double *depth, const double *alpha, int32_t width, int32_t height, double *win_z,
                   int32_t *win_k, int32_t *win_t, int32_t *win_ox, int32_t *win_oy) {
    for (int64_t k = 0; k < nk; k++) {
        const double *a = p0 + 3 * k, *b = p1 + 3 * k;
        for (int32_t t = 0; t < steps; t++) {
            double tt = steps > 1 ? (double)t / (steps - 1) : 0.0;
            double x = a[0] + (b[0] - a[0]) * tt, y = a[1] + (b[1] - a[1]) * tt, z0 = a[2] + (b[2] - a[2]) * tt;
            double r = rad[t], lim = r > 0.5 ? r : 0.5;
            for (int32_t oy = -R; oy <= R; oy++) {
                for (int32_t ox = -R; ox <= R; ox++) {
                    double d = sqrt((double)(ox * ox + oy * oy));
                    if (d > lim) continue;
                    int64_t ix = (int64_t)floor(x + ox), iy = (int64_t)floor(y + oy);
                    if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;
                    double h = r * r - d * d;
                    double z = z0 + (h > 0 ? sqrt(h) : 0.0);
                    int64_t p = iy * width + ix;
                    if (!(z > depth[p] - 0.5 || alpha[p] == 0)) continue;
                    if (z > win_z[p]) {
                        win_z[p] = z;
                        win_k[p] = (int32_t)k;
                        win_t[p] = t;
                        win_ox[p] = ox;
                        win_oy[p] = oy;
                    }
                }
            }
        }
    }
}

/* Linear 0..1 -> sRGB 0..1, clipped, in place. */
API void fg_srgb(double *c, int64_t n) {
    for (int64_t p = 0; p < n; p++) {
        double v = c[p];
        v = v < 0 ? 0 : (v > 1 ? 1 : v);
        c[p] = v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1 / 2.4) - 0.055;
    }
}

#ifdef FISHGEN_PYMODULE
/* Built by setuptools as an extension module: an empty module, so the shared library
 * installs next to the package; ctypes loads the functions above from it. */
#include <Python.h>
static struct PyModuleDef fast_module = {PyModuleDef_HEAD_INIT, "_fast", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__fast(void) { return PyModule_Create(&fast_module); }
#endif
