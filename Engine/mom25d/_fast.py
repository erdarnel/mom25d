"""Compiled (numba) kernels for the matrix fill.

Optional: without numba (or with MOM25D_NUMBA=0) the numpy code paths in
solver.py / hmatrix.py are used; both give the same Z to ~1e-15 relative
(only the summation order differs).

The kernels fuse what the numpy path does with large temporaries: source
point distances, the cubic-spline lookup of the smooth kernel parts, the
inner (source) quadrature, the outer (test) quadrature and the RWG
coefficients, for one block of test triangles against one group of source
triangles. The spline is evaluated exactly as scipy's PPoly does (interval
by binary search, then sum c[3-k] dx^k).

MOM25D_THREADS sets numba's thread count (default 8).
"""

import os

import numpy as np

ENABLED = False
if os.environ.get("MOM25D_NUMBA", "1") != "0":
    try:
        import numba as nb

        # numba's TBB layer refuses older TBB installs with a warning; prefer OpenMP
        nb.config.THREADING_LAYER_PRIORITY = ["omp", "workqueue", "tbb"]
        ENABLED = True
    except ImportError:  # pragma: no cover - numba is optional
        pass

FOUR_PI = 4 * np.pi


def spline_data(pair):
    """(x, c) of a kernel pair's cubic spline of the smooth parts [S_A, S_V]
    (c: (4, n-1, 2) complex), cached on the pair. None for zero pairs."""
    if getattr(pair, "zero", False):
        return None
    d = getattr(pair, "_fast_spl", None)
    if d is None:
        spl = pair._spl
        d = (np.ascontiguousarray(spl.x, float), np.ascontiguousarray(spl.c, complex))
        pair._fast_spl = d
    return d


if ENABLED:
    # all 22 (hyper)threads of the WSL box were 2x slower than 4-16
    nb.set_num_threads(min(int(os.environ.get("MOM25D_THREADS", 8)), nb.config.NUMBA_NUM_THREADS))

    @nb.njit(cache=True, inline="always")
    def _interval(x, r):
        # scipy find_interval: x[i] <= r < x[i+1], clipped to the end intervals
        n = x.size
        if r < x[1]:
            return 0
        if r >= x[n - 2]:
            return n - 2
        lo, hi = 1, n - 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if r < x[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    @nb.njit(cache=True, inline="always")
    def _spline2(x, c, r):
        i = _interval(x, r)
        dx = r - x[i]
        a = c[3, i, 0] + dx * 0.0
        v = c[3, i, 1] + dx * 0.0
        z = 1.0
        for k in range(1, 4):
            z = z * dx
            a = a + c[3 - k, i, 0] * z
            v = v + c[3 - k, i, 1] * z
        return a, v

    @nb.njit(parallel=True, cache=True)
    def group_vals(vals, IVout, sel, pts, w, Vs, cAs, cVs, static, S0, S1,
                   r, wo, Vt, cAt, cVt, x, c, kA, kV, dz, jwmu, inv_jwe):
        """Add the contributions of source triangles `sel` (local columns of
        vals) to vals[p, i, s, j] (test tri p of the block, test half-basis
        i, source tri s, source half-basis j) and IVout[p, s].

        pts, w (ns, L, 2) / (ns, L): source quadrature; Vs, cAs, cVs (ns, 3..):
        source vertices and RWG coefficients; static: S0 (ns, M), S1 (ns, M, 2)
        closed-form 1/R integrals (near sources) else full kernel by
        quadrature; r (M, 2) = test points (npb * K); wo (npb, K) weights."""
        npb, K = wo.shape
        ns, L = w.shape
        for s in nb.prange(ns):
            q = sel[s]
            for p in range(npb):
                IA = np.zeros((3, 3), np.complex128)
                IV = 0j
                for k in range(K):
                    m = p * K + k
                    rx, ry = r[m, 0], r[m, 1]
                    if static:
                        k0a = kA * S0[s, m]
                        k1x = kA * S1[s, m, 0]
                        k1y = kA * S1[s, m, 1]
                        k0v = kV * S0[s, m]
                    else:
                        k0a = 0j
                        k1x = 0j
                        k1y = 0j
                        k0v = 0j
                    for l in range(L):
                        px, py = pts[s, l, 0], pts[s, l, 1]
                        ddx, ddy = rx - px, ry - py
                        R = np.sqrt(ddx * ddx + ddy * ddy)
                        sa, sv = _spline2(x, c, R)
                        if not static:
                            if dz == 0.0:
                                inv = 1 / (FOUR_PI * R)
                            else:
                                inv = 1 / (FOUR_PI * np.sqrt(R * R + dz * dz))
                            sa = sa + kA * inv
                            sv = sv + kV * inv
                        wl = w[s, l]
                        saw = sa * wl
                        k0a += saw
                        k1x += saw * px
                        k1y += saw * py
                        k0v += sv * wl
                    wk = wo[p, k]
                    IV += wk * k0v
                    for i in range(3):
                        ax = wk * (rx - Vt[p, i, 0])
                        ay = wk * (ry - Vt[p, i, 1])
                        t1 = ax * k1x + ay * k1y
                        for j in range(3):
                            IA[i, j] += t1 - (ax * Vs[s, j, 0] + ay * Vs[s, j, 1]) * k0a
                IVout[p, q] = IV
                for i in range(3):
                    for j in range(3):
                        vals[p, i, q, j] = (jwmu * cAt[p, i] * cAs[s, j] * IA[i, j]
                                            + inv_jwe * cVt[p, i] * cVs[s, j] * IV)

    @nb.njit(cache=True)
    def scatter(Z, vals, rows, cols):
        """Z[rows[p, i], cols[s, j]] += vals[p, i, s, j] (index < 0: skipped)."""
        npb, _, ns, _ = vals.shape
        for p in range(npb):
            for i in range(3):
                a = rows[p, i]
                if a < 0:
                    continue
                for s in range(ns):
                    for j in range(3):
                        b = cols[s, j]
                        if b >= 0:
                            Z[a, b] += vals[p, i, s, j]

    @nb.njit(parallel=True, cache=True)
    def far_entries(out, ri, ci, Pr, Ur, Qr, Pc, Uc, Qc, x, c, kA, kV, dz, jwmu, inv_jwe):
        """out[ri[a], ci[b]] = jw mu sum U.U gA + 1/(jw eps) sum Q Q gV over
        the basis quadrature points (rows Pr/Ur/Qr, columns Pc/Uc/Qc)."""
        nr, kr = Qr.shape
        nc, kc = Qc.shape
        for ab in nb.prange(nr * nc):  # flat: ACA asks for single rows / columns
            a, b = ab // nc, ab % nc
            if True:
                za = 0j
                zv = 0j
                for k in range(kr):
                    for l in range(kc):
                        ddx = Pr[a, k, 0] - Pc[b, l, 0]
                        ddy = Pr[a, k, 1] - Pc[b, l, 1]
                        R = np.sqrt(ddx * ddx + ddy * ddy)
                        sa, sv = _spline2(x, c, R)
                        if dz == 0.0:
                            inv = 1 / (FOUR_PI * R)
                        else:
                            inv = 1 / (FOUR_PI * np.sqrt(R * R + dz * dz))
                        za += (Ur[a, k, 0] * Uc[b, l, 0] + Ur[a, k, 1] * Uc[b, l, 1]) * (sa + kA * inv)
                        zv += Qr[a, k] * Qc[b, l] * (sv + kV * inv)
                out[ri[a], ci[b]] = jwmu * za + inv_jwe * zv


_EMPTY_S0 = np.zeros((0, 0), complex)
_EMPTY_S1 = np.zeros((0, 0, 2), complex)


def chunk_vals(greens, lt, pb, r, groups, wo, V, cA, cV, src, omega, jwmu, inv_jwe):
    """vals (npb, 3, nS, 3) [Z entries per half-basis pair, already scaled]
    and IV (npb, nS) of test triangles pb against source triangles src
    (groups index into src), as the numpy path of assemble / HOperator."""
    npb, nS = pb.size, src.size
    vals = np.zeros((npb, 3, nS, 3), complex)
    IV = np.zeros((npb, nS), complex)
    Vt, cAt, cVt = V[pb], cA[pb], cV[pb]
    for ls, sel, pts, w, static in groups:
        pr = greens.pair(lt, ls)
        sd = spline_data(pr)
        if sd is None:
            continue
        q = src[sel]
        if static is None:
            S0, S1, st = _EMPTY_S0, _EMPTY_S1, False
        else:
            S0, S1, st = static[0], static[1], True
        group_vals(vals, IV, np.ascontiguousarray(sel), np.ascontiguousarray(pts), np.ascontiguousarray(w),
                   V[q], cA[q], cV[q], st, S0, S1, r, wo, Vt, cAt, cVt, sd[0], sd[1],
                   complex(pr.cA), complex(pr.cV), float(pr.dz), jwmu, inv_jwe)
    return vals, IV


if ENABLED:
    @nb.njit(parallel=True, cache=True)
    def aperture_near(pt, ps, toff, tpts, tw, soff, spts, sw, Va, Vs, x, c, out):
        """Aperture coupling, near pairs (aperture.py): for pair p (test
        triangle pt[p], source triangle ps[p]) out[p, i, j] = sum_k sum_l
        w_k w_l (r_k - a_i) . G(r_k - r'_l) . (r'_l - b_j), G = h_a Rot - h_s Q,
        with the fine rules (points tpts[toff[t]:toff[t+1]], ...) and the
        spline (x, c) of [h_a, h_s]."""
        for p in nb.prange(pt.size):
            t, s = pt[p], ps[p]
            for k in range(toff[t], toff[t + 1]):
                rx, ry, wk = tpts[k, 0], tpts[k, 1], tw[k]
                # A = sum_l w G r',  B = sum_l w G  (B row-major 2x2)
                Ax = 0j
                Ay = 0j
                Bxx = 0j
                Bxy = 0j
                Byx = 0j
                Byy = 0j
                for l in range(soff[s], soff[s + 1]):
                    dx = rx - spts[l, 0]
                    dy = ry - spts[l, 1]
                    r2 = dx * dx + dy * dy
                    rho = np.sqrt(r2)
                    ha, hs = _spline2(x, c, rho)
                    if r2 > 0:
                        c2 = (dx * dx - dy * dy) / r2
                        s2 = 2 * dx * dy / r2
                    else:
                        c2 = 0.0
                        s2 = 0.0
                    w = sw[l]
                    gxx = w * hs * s2
                    gxy = w * (ha - hs * c2)
                    gyx = w * (-ha - hs * c2)
                    gyy = -w * hs * s2
                    Bxx += gxx
                    Bxy += gxy
                    Byx += gyx
                    Byy += gyy
                    Ax += gxx * spts[l, 0] + gxy * spts[l, 1]
                    Ay += gyx * spts[l, 0] + gyy * spts[l, 1]
                for j in range(3):
                    bx, by = Vs[s, j, 0], Vs[s, j, 1]
                    vx = Ax - (Bxx * bx + Bxy * by)
                    vy = Ay - (Byx * bx + Byy * by)
                    for i in range(3):
                        out[p, i, j] += wk * ((rx - Va[t, i, 0]) * vx + (ry - Va[t, i, 1]) * vy)
