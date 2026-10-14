"""Hierarchical-matrix (H-matrix) operator and iterative solve for large models.

The dense solver stores Z (16 N^2 bytes) and factors it in O(N^3); beyond
~10^4 unknowns that is the limit. Here Z is never formed:

- the RWG unknowns are grouped in a cluster tree (bisection of their edge
  midpoints along the largest extent, `leaf` unknowns per leaf);
- cluster pairs that are well separated (max diameter <= eta * distance)
  are "far": their block is a low-rank product U V^T, built by adaptive
  cross approximation (ACA, partial pivoting) from a few rows / columns of
  entries evaluated with a 3-point rule on the full layered kernels, then
  recompressed (QR + SVD) to the tolerance; each unordered pair once (Z is
  symmetric);
- the other (leaf) pairs are "near": computed exactly by the same tiered
  integration as the dense `assemble` (singularity extraction, 7 / 3 / 1
  point rules), restricted to the neighbouring leaves, and stored as one
  sparse matrix;
- via and antipad unknowns (few) stay dense, computed exactly.

Solve: GMRES preconditioned by a sparse LU of the near-field matrix (all
right-hand sides, incl. lumped gaps). Memory and fill time grow roughly
as N log N.
"""

import time

import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import _fast
from .constants import EPS0, MU0
from .integrals import TRI3_BARY, TRI3_W, TRI7_W, potential_integrals, quad_points


class _Cluster:
    __slots__ = ("idx", "lo", "hi", "children", "id", "smax")

    def __init__(self, idx, pos, size):
        self.idx = idx
        self.lo, self.hi = pos[idx].min(0), pos[idx].max(0)
        self.smax = float(size[idx].max())  # largest triangle of its unknowns
        self.children = []

    @property
    def diam(self):
        return float(np.linalg.norm(self.hi - self.lo))


def _tree(idx, pos, size, leaf, out):
    c = _Cluster(idx, pos, size)
    c.id = len(out)
    out.append(c)
    if idx.size > leaf:
        ax = int(np.argmax(c.hi - c.lo))
        order = idx[np.argsort(pos[idx, ax], kind="stable")]
        h = order.size // 2
        c.children = [_tree(order[:h], pos, size, leaf, out), _tree(order[h:], pos, size, leaf, out)]
    return c


ADM_SIZES = 5.0  # far blocks must also be this many triangle sizes apart


def _dist(a, b):
    gap = np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return float(np.linalg.norm(gap))


def _partition(a, b, eta, far, near):
    """Block partition of (a, b). Symmetric by construction: which cluster
    gets split depends on the pair, not on its order (ties in diameter --
    common on regular grid meshes -- are broken by cluster id), so (a, b)
    and (b, a) produce mirrored blocks. An asymmetric partition covered
    parts of Z twice when far blocks are applied with their transpose."""
    d = _dist(a, b)
    # well separated, and far enough from the triangles' own size for the
    # far-field quadrature (on long thin cells a cluster's diameter alone
    # admitted pairs 2 cell lengths apart: 2e-3 error in S)
    if max(a.diam, b.diam) <= eta * d and d >= ADM_SIZES * max(a.smax, b.smax):
        far.append((a, b))
    elif not a.children and not b.children:
        near.append((a, b))
    elif a.children and (not b.children or (a.diam, a.id) > (b.diam, b.id)):
        for c in a.children:
            _partition(c, b, eta, far, near)
    else:
        for c in b.children:
            _partition(a, c, eta, far, near)


def _aca(row, col, m, n, tol, kmax):
    """Adaptive cross approximation with partial pivoting: block ~ U V^T.
    row(i) / col(j) return the exact i-th row / j-th column. Returns (U, V)
    or None if the rank would exceed kmax."""
    U = np.zeros((m, kmax), complex)
    V = np.zeros((n, kmax), complex)
    k = 0
    used_r, used_c = np.zeros(m, bool), np.zeros(n, bool)
    i, norm2 = 0, 0.0
    for _ in range(kmax):
        r = row(i) - V[:, :k] @ U[i, :k]
        used_r[i] = True
        rr = np.where(used_c, 0.0, np.abs(r))
        j = int(np.argmax(rr))
        if rr[j] == 0.0:
            free = np.nonzero(~used_r)[0]
            if not free.size:
                break
            i = int(free[0])
            continue
        v = r / r[j]
        c = col(j) - U[:, :k] @ V[j, :k]
        used_c[j] = True
        nu, nv = np.linalg.norm(c), np.linalg.norm(v)
        if k:
            norm2 += 2 * np.abs((c.conj() @ U[:, :k]) * (v.conj() @ V[:, :k])).sum()  # |vdot(u, c) vdot(v', v)|
        norm2 += (nu * nv) ** 2
        U[:, k], V[:, k] = c, v
        k += 1
        if nu * nv <= tol * np.sqrt(norm2):
            break
        cc = np.where(used_r, 0.0, np.abs(c))
        i = int(np.argmax(cc))
        if cc[i] == 0.0:
            break
    else:
        return None
    if not k:
        return np.zeros((m, 0), complex), np.zeros((n, 0), complex)
    return _recompress(U[:, :k], V[:, :k], tol)


def _recompress(U, V, tol):
    qu, ru = np.linalg.qr(U)
    qv, rv = np.linalg.qr(V)
    w, s, xh = np.linalg.svd(ru @ rv.T)
    k = max(1, int(np.sum(s > tol * s[0]))) if s.size and s[0] > 0 else 0
    return qu @ (w[:, :k] * s[:k]), qv @ xh[:k].T


class HOperator:
    """Compressed MoM operator for a Solver's mesh (same unknown order as
    the dense Z: RWG, via segments, antipad rings)."""

    SMALL = 64  # far blocks with a side <= SMALL are stored dense (ACA would not pay)
    CEN = 10.0  # centroid rule for far blocks this many triangle sizes apart (dense far tier)
    ACCEPT = 1.0  # a stalled GMRES solve is accepted at residual <= ACCEPT * tol
    PREC_SIZES = 6.0  # preconditioner: near entries within this many cell sizes (3: 30x more GMRES its)

    def __init__(self, solver, leaf=48, eta=1.0, tol=1e-4, near_factor=3.0, far_factor=10.0,
                 block=32, verbose=True):
        from .solver import AssemblyGeometry

        t0 = time.time()
        self.solver, self.tol, self.verbose = solver, tol, verbose
        m = solver.mesh
        st = solver.stack
        via_z = {n: st.level_z(n) for n in m.layer_names + list(getattr(m, "planes", []))}
        via_z["gnd"] = 0.0
        section = [st.section(n) for n in m.layer_names]
        self.geom = g = AssemblyGeometry(m, block, near_factor, far_factor, zl=solver.zl, via_z=via_z,
                                         section=section, blocks=False, walls=getattr(solver, "_walls", None))
        self.nb, self.nv = nb, nv = m.n_basis, m.n_vias
        self.nf = getattr(m, "n_frills", 0)
        self.na = getattr(m, "n_ap", 0)  # aperture M unknowns (aperture.py), after the frills
        self.N = m.n_unknowns
        # basis data for the far-field (3-point) entry evaluator
        V, A = m.verts, m.area
        P3 = quad_points(V, TRI3_BARY)
        w3 = TRI3_W[None, :] * A[:, None]
        tri = np.stack([m.tp, m.tm], 1)
        loc = np.stack([m.vp, m.vm], 1)
        cA = g.cA[tri, loc]
        cV = g.cV[tri, loc]
        self.pts = P3[tri].reshape(nb, 6, 2)
        self.bpos = np.c_[m.edge_mid, solver.zl[m.edge_layer]]
        rho = P3[tri] - V[tri, loc][:, :, None, :]  # (nb,2,3,2)
        self.u6 = (w3[tri][..., None] * cA[:, :, None, None] * rho).reshape(nb, 6, 2)
        self.q6 = (w3[tri] * cV[:, :, None]).reshape(nb, 6)
        # centroid (1-point) data for well-separated blocks: 4 kernel evaluations per entry
        cen = m.centroid[tri]  # (nb,2,2)
        self.pts1 = cen
        self.u1 = A[tri][..., None] * cA[..., None] * (cen - V[tri, loc])
        self.q1 = A[tri] * cV
        self.bsize = g.size[tri].max(1)  # largest triangle of each basis
        self.blayer = m.edge_layer
        self.wall = getattr(m, "wall", np.zeros(nb, bool))  # thick metal: halves on two levels
        self.hlayer = m.tri_layer[tri]  # (nb, 2) level of each half
        # cluster tree / block partition over the RWG unknowns
        pos = np.c_[m.edge_mid, solver.zl[m.edge_layer]]
        nodes = []
        root = _tree(np.arange(nb), pos, self.bsize, leaf, nodes)
        far, near = [], []
        _partition(root, root, eta, far, near)
        seen, self.far = set(), []
        for a, b in far:  # each unordered pair once
            key = (min(a.id, b.id), max(a.id, b.id))
            if key not in seen:
                seen.add(key)
                self.far.append((a, b))
        nbr = {}
        for a, b in near:
            nbr.setdefault(a.id, (a, []))[1].append(b)
        self._near_setup(nbr, near_factor, far_factor, block)
        self._pad_setup(near_factor, far_factor, block)
        if verbose:
            print(f"  H-matrix: {len(nodes)} clusters, {len(self.far)} far blocks, "
                  f"{len(near)} near leaf pairs, setup {time.time() - t0:.1f}s")

    # ------------------------------------------------------------ geometry
    def _chunks(self, test, src, near_factor, far_factor, block):
        """Test-triangle chunks (one layer each, <= block) against source
        triangles `src`, with near / mid / far groups (local source indices)
        and the closed-form near integrals -- as in AssemblyGeometry."""
        g, m = self.geom, self.solver.mesh
        tl, cen, size, V = m.tri_layer, m.centroid, g.size, g.V
        out = []
        for lt in np.unique(tl[test]):
            on = test[tl[test] == lt]
            for p0 in range(0, on.size, block):
                pb = on[p0:p0 + block]
                r = g.qpts[pb].reshape(-1, 2)
                groups = []
                for ls in np.unique(tl[src]):
                    if g.section is not None and g.section[ls] != g.section[lt]:
                        continue
                    loc = np.nonzero(tl[src] == ls)[0]
                    s = src[loc]
                    dz = abs(g.zl[lt] - g.zl[ls])
                    dist = np.linalg.norm(cen[pb][:, None, :] - cen[None, s, :], axis=-1)
                    rel = np.min(np.sqrt(dist**2 + dz**2) / np.maximum(size[pb][:, None], size[None, s]), axis=0)
                    nr, md, fr = rel < near_factor, (rel >= near_factor) & (rel < far_factor), rel >= far_factor
                    if nr.any():
                        I0, I1 = potential_integrals(V[s[nr]], r, dz)
                        static = (I0 / (4 * np.pi), (r[None] * I0[..., None] + I1) / (4 * np.pi))
                        groups.append((ls, loc[nr], g.qpts[s[nr]], g.qw[s[nr]], static))
                    for sel, pts, w in ((md, g.mpts, g.mw), (fr, g.fpts, g.fw)):
                        if sel.any():
                            groups.append((ls, loc[sel], pts[s[sel]], w[s[sel]], None))
                wo = g.qw[pb]
                Xw = (g.qpts[pb][:, None, :, :] - V[pb][:, :, None, :]) * wo[:, None, :, None]
                out.append((lt, pb, r, groups, wo, Xw))
        return out

    def _near_setup(self, nbr, near_factor, far_factor, block):
        m, b = self.solver.mesh, self.geom.b
        self.leaves = []
        for aid, (a, blist) in nbr.items():
            rows = a.idx
            cols = np.unique(np.concatenate([c.idx for c in blist]))
            test = np.unique(np.r_[m.tp[rows], m.tm[rows]])
            src = np.unique(np.r_[m.tp[cols], m.tm[cols]])
            rpos = -np.ones(self.nb, int)
            rpos[rows] = np.arange(rows.size)
            cpos = -np.ones(self.nb, int)
            cpos[cols] = np.arange(cols.size)
            chunks = self._chunks(test, src, near_factor, far_factor, block)
            bs = b[src]
            col_loc = np.where(bs >= 0, cpos[np.maximum(bs, 0)], -1)
            ch = []
            for lt, pb, r, groups, wo, Xw in chunks:
                bt = b[pb]
                row_loc = np.where(bt >= 0, rpos[np.maximum(bt, 0)], -1)
                ch.append((lt, pb, r, groups, wo, Xw, row_loc))
            self.leaves.append((rows, cols, src, col_loc, ch))

    def _pad_setup(self, near_factor, far_factor, block):
        g = self.geom
        self.pad_chunks = []
        if self.nv:
            alltri = np.arange(g.nt)
            self.pad_chunks = self._chunks(g.pad_tris, alltri, near_factor, far_factor, block)

    # ------------------------------------------------------------ assembly
    def _chunk(self, greens, lt, pb, r, groups, wo, Xw, src, need_A=True):
        """IA (npb, nS, 3, 3) and IV (npb, nS) of a test chunk against src."""
        from .solver import _cmatmul, _source_potentials

        K = TRI7_W.size
        nS, npb = src.size, pb.size
        K0A = np.zeros((nS, npb * K), complex)
        K1A = np.zeros((nS, npb * K, 2), complex)
        K0V = np.zeros((nS, npb * K), complex)
        for ls, sel, pts, w, static in groups:
            K0A[sel], K1A[sel], K0V[sel] = _source_potentials(greens.pair(lt, ls), r, pts, w, static)
        IV = np.einsum("qpk,pk->pq", K0V.reshape(nS, npb, K), wo)
        if not need_A:
            return None, IV
        K0A = K0A.reshape(nS, npb, K).transpose(1, 0, 2)
        K1A = K1A.reshape(nS, npb, K, 2).transpose(1, 0, 2, 3)
        T1 = _cmatmul(K1A.reshape(npb, nS, 2 * K), Xw.reshape(npb, 3, 2 * K).transpose(0, 2, 1))
        T2 = _cmatmul(K0A, Xw.transpose(0, 2, 1, 3).reshape(npb, K, 6)).reshape(npb, nS, 3, 2)
        IA = T1[..., None] - np.einsum("pqid,qjd->pqij", T2, self.geom.V[src], optimize=True)
        return IA, IV

    def _eval(self, greens, R, C, centroid=False):
        """Far-field entries Z[R, C]: 3-point rule on both sides, or the
        centroid rule (well-separated blocks, like the dense far tier)."""
        omega = greens.omega
        jwmu, inv_jwe = 1j * omega * MU0, 1 / (1j * omega * EPS0)
        if self.wall[R].any() or self.wall[C].any():
            return self._eval_halves(greens, R, C, centroid, jwmu, inv_jwe)
        out = np.zeros((R.size, C.size), complex)
        for lr in np.unique(self.blayer[R]):
            ri = np.nonzero(self.blayer[R] == lr)[0]
            for lc in np.unique(self.blayer[C]):
                ci = np.nonzero(self.blayer[C] == lc)[0]
                pr = greens.pair(lr, lc)
                if getattr(pr, "zero", False):
                    continue
                P, U, Q = (self.pts1, self.u1, self.q1) if centroid else (self.pts, self.u6, self.q6)
                if _fast.ENABLED:
                    x, c = _fast.spline_data(pr)
                    Rr, Cc = R[ri], C[ci]
                    _fast.far_entries(out, ri, ci, P[Rr], U[Rr], Q[Rr], P[Cc], U[Cc], Q[Cc], x, c,
                                      complex(pr.cA), complex(pr.cV), float(pr.dz), jwmu, inv_jwe)
                    continue
                pa, pb = P[R[ri]], P[C[ci]]  # (r,k,2), (c,l,2)
                d = pa[:, None, :, None, :] - pb[None, :, None, :, :]
                rho = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)  # (r,c,k,l)
                gA, gV = pr.gA(rho), pr.gV(rho)
                ua, ub = U[R[ri]], U[C[ci]]
                dot = np.einsum("rkd,cld->rckl", ua, ub)
                za = np.einsum("rckl,rckl->rc", dot, gA)
                zv = np.einsum("rk,cl,rckl->rc", Q[R[ri]], Q[C[ci]], gV)
                out[np.ix_(ri, ci)] = jwmu * za + inv_jwe * zv
        return out

    def _eval_halves(self, greens, R, C, centroid, jwmu, inv_jwe):
        """_eval for blocks holding side-wall bases (thick metal), whose two
        triangles lie on different levels: per pair of half-bases, each
        with its own level's kernels."""
        P, U, Q = (self.pts1, self.u1, self.q1) if centroid else (self.pts, self.u6, self.q6)
        k = P.shape[1] // 2
        out = np.zeros((R.size, C.size), complex)
        for a in (0, 1):
            Pa, Ua, Qa = (np.ascontiguousarray(X[R][:, a * k:(a + 1) * k]) for X in (P, U, Q))
            for c in (0, 1):
                Pc, Uc, Qc = (np.ascontiguousarray(X[C][:, c * k:(c + 1) * k]) for X in (P, U, Q))
                hr, hc = self.hlayer[R, a], self.hlayer[C, c]
                for lr in np.unique(hr):
                    ri = np.nonzero(hr == lr)[0]
                    for lc in np.unique(hc):
                        ci = np.nonzero(hc == lc)[0]
                        pr = greens.pair(lr, lc)
                        if getattr(pr, "zero", False):
                            continue
                        if _fast.ENABLED:
                            x, cc = _fast.spline_data(pr)
                            tmp = np.zeros((ri.size, ci.size), complex)
                            _fast.far_entries(tmp, np.arange(ri.size), np.arange(ci.size), Pa[ri], Ua[ri], Qa[ri],
                                              Pc[ci], Uc[ci], Qc[ci], x, cc, complex(pr.cA), complex(pr.cV),
                                              float(pr.dz), jwmu, inv_jwe)
                            out[np.ix_(ri, ci)] += tmp
                            continue
                        d = Pa[ri][:, None, :, None, :] - Pc[ci][None, :, None, :, :]
                        rho = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
                        gA, gV = pr.gA(rho), pr.gV(rho)
                        za = np.einsum("rckl,rckl->rc", np.einsum("rkd,cld->rckl", Ua[ri], Uc[ci]), gA)
                        zv = np.einsum("rk,cl,rckl->rc", Qa[ri], Qc[ci], gV)
                        out[np.ix_(ri, ci)] += jwmu * za + inv_jwe * zv
        return out

    def assemble(self, greens, zs=0.0, zs_via=None):
        """Near sparse matrix, far low-rank blocks and dense via / antipad
        parts at greens' frequency. zs: a number or one per RWG basis
        (zs_via: via walls, default zs); thick-metal side walls (geom.walls)
        go into the near matrix."""
        t0 = time.time()
        omega = greens.omega
        jwmu, inv_jwe = 1j * omega * MU0, 1 / (1j * omega * EPS0)
        g = self.geom
        cA, cV = g.cA, g.cV
        ii, jj, vv = [], [], []
        for rows, cols, src, col_loc, ch in self.leaves:
            blk = np.zeros((rows.size, cols.size), complex)
            for lt, pb, r, groups, wo, Xw, row_loc in ch:
                if _fast.ENABLED:
                    vals, _ = _fast.chunk_vals(greens, lt, pb, r, groups, wo, g.V, cA, cV, src, omega, jwmu,
                                               inv_jwe)
                    _fast.scatter(blk, vals, row_loc, col_loc)
                    continue
                IA, IV = self._chunk(greens, lt, pb, r, groups, wo, Xw, src)
                vals = (jwmu * cA[pb][:, None, :, None] * cA[src][None, :, None, :] * IA
                        + inv_jwe * cV[pb][:, None, :, None] * cV[src][None, :, None, :] * IV[:, :, None, None])
                Rl = np.broadcast_to(row_loc[:, None, :, None], vals.shape)
                Cl = np.broadcast_to(col_loc[None, :, None, :], vals.shape)
                ok = (Rl >= 0) & (Cl >= 0)
                np.add.at(blk, (Rl[ok], Cl[ok]), vals[ok])
            r_, c_ = np.meshgrid(rows, cols, indexing="ij")
            ii.append(r_.ravel())
            jj.append(c_.ravel())
            vv.append(blk.ravel())
        if np.any(zs):
            gi, gj, gg = g.gram
            ii.append(gi)
            jj.append(gj)
            vv.append((zs[gi] if np.ndim(zs) else zs) * gg)
        W = g.walls
        if W is not None:
            ii.append(W["rows"])
            jj.append(W["cols"])
            vv.append(jwmu / MU0 * W["L"])
            if np.any(zs):
                ii.append(W["idx"])
                jj.append(W["idx"])
                vv.append((zs[W["idx"]] if np.ndim(zs) else zs) * W["hl"])
        near = sp.coo_matrix((np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))),
                             shape=(self.nb, self.nb)).tocsr()
        self.near = 0.5 * (near + near.T)  # exact Galerkin Z is symmetric
        t1 = time.time()
        # far blocks
        self.lowrank = []
        dense = []  # blocks stored as they are: folded into one sparse matrix below
        nent = 0
        for a, b in self.far:
            R, C = a.idx, b.idx
            # centroid rule once the clusters are >= 8 triangle sizes apart
            cen = _dist(a, b) >= self.CEN * max(self.bsize[R].max(), self.bsize[C].max())
            res = None
            if min(R.size, C.size) > self.SMALL:
                res = _aca(lambda i: self._eval(greens, R[i:i + 1], C, cen)[0],
                           lambda j: self._eval(greens, R, C[j:j + 1], cen)[:, 0],
                           R.size, C.size, self.tol, max(2, min(R.size, C.size) // 3))
            if res is None:  # small or not low rank: store the block itself
                Bk = self._eval(greens, R, C, cen)
                dense.append((R, C, Bk))
                nent += Bk.size
            else:
                U, Vv = res
                self.lowrank.append((R, C, U, Vv))
                nent += U.size + Vv.size
        # dense far blocks (small or not low rank; most of them in a board with many
        # small clusters) as one sparse matrix F (each block once; the matvec adds
        # F x + F^T x): a single sparse product instead of thousands of small BLAS
        # calls per matvec (16 s -> 0.3 s at N 3.2e4, 6800 of 8200 far blocks dense).
        # Filled into preallocated int32 / complex arrays to keep the peak memory low.
        if dense:
            n = sum(B.size for _, _, B in dense)
            ri, ci = np.empty(n, np.int32), np.empty(n, np.int32)
            vals = np.empty(n, complex)
            k = 0
            while dense:
                R, C, B = dense.pop()
                m = B.size
                ri[k:k + m], ci[k:k + m], vals[k:k + m] = np.repeat(R, C.size), np.tile(C, R.size), B.ravel()
                k += m
            self.fard = sp.csr_matrix((vals, (ri, ci)), shape=(self.nb, self.nb))
            del ri, ci, vals
        else:
            self.fard = None
        t2 = time.time()
        # dense extra unknowns: via segments and antipad rings
        nx = self.nv + self.nf + self.na
        self.Zxb = np.zeros((nx - self.na, self.nb), complex)  # via / frill rows (dense)
        self.Zab = None  # aperture rows: sparse (na x nb)
        self.Zxx = np.zeros((nx, nx), complex)
        if self.nv:
            from .solver import _via_blocks

            IVpad = np.zeros((g.pad_tris.size, g.nt), complex)
            alltri = np.arange(g.nt)
            for lt, pb, r, groups, wo, Xw in self.pad_chunks:
                if _fast.ENABLED:
                    _, IV = _fast.chunk_vals(greens, lt, pb, r, groups, wo, g.V, cA, cV, alltri, omega, jwmu,
                                             inv_jwe)
                else:
                    _, IV = self._chunk(greens, lt, pb, r, groups, wo, Xw, alltri, need_A=False)
                IVpad[g.pad_pos[pb]] = IV
            Zvb, Zvv = _via_blocks(g, greens, IVpad, zs if zs_via is None else zs_via)
            self.Zxb[:self.nv] = Zvb
            self.Zxx[:self.nv, :self.nv] = Zvv
        if self.nf:
            from .antipad import FrillModel

            sol = self.solver
            if getattr(sol, "_frills", None) is None:
                sol._frills = FrillModel(sol)
            Cf, F = sol._frills.blocks(greens, omega)
            nb, nv = self.nb, self.nv
            nf = self.nf
            self.Zxb[nv:nv + nf] = -Cf[:nb].T
            self.Zxx[nv:nv + nf, :nv] = -Cf[nb:nb + nv].T
            self.Zxx[:nv, nv:nv + nf] = -Cf[nb:nb + nv]
            self.Zxx[nv:nv + self.nf, nv:nv + self.nf] = -0.5 * (F + F.T)
        if self.na:
            B, Y = self.solver.apertures().blocks(greens.freq)
            nb, a0 = self.nb, self.nv + self.nf
            self.Zab = B[:nb].T.tocsr()
            Bx = B[nb:].toarray()
            self.Zxx[a0:, :a0] = Bx.T
            self.Zxx[:a0, a0:] = Bx
            self.Zxx[a0:, a0:] = Y
        self.Zxx = 0.5 * (self.Zxx + self.Zxx.T)
        self.stats = dict(near_nnz=self.near.nnz, far_entries=nent, dense_equiv=self.N**2,
                          t_near=t1 - t0, t_far=t2 - t1, t_extra=time.time() - t2)
        self._prec = None
        return self

    # --------------------------------------------------------------- solve
    def matvec(self, x):
        nb = self.nb
        xb, xx = x[:nb], x[nb:]
        yb = self.near @ xb
        if getattr(self, "fard", None) is not None:
            yb += self.fard @ xb + self.fard.T @ xb
        for R, C, U, Vv in self.lowrank:
            if Vv is None:  # dense block (older operators)
                yb[R] += U @ xb[C]
                yb[C] += U.T @ xb[R]
            elif U.shape[1]:
                yb[R] += U @ (Vv.T @ xb[C])
                yb[C] += Vv @ (U.T @ xb[R])
        if xx.size:
            a0 = self.Zxb.shape[0]
            yb = yb + self.Zxb.T @ xx[:a0]
            yx = self.Zxx @ xx
            yx[:a0] += self.Zxb @ xb
            if self.Zab is not None:
                yb += self.Zab.T @ xx[a0:]
                yx[a0:] += self.Zab @ xb
            return np.r_[yb, yx]
        return yb

    def _preconditioner(self):
        if self._prec is None:
            nb, nx = self.nb, self.nv + self.nf + self.na
            # LU of the whole near matrix filled in badly (99 s at N 10^4);
            # the dominant local couplings are enough to precondition
            nr = self.near.tocoo()
            d = np.linalg.norm(self.bpos[nr.row] - self.bpos[nr.col], axis=1)
            keep = d <= self.PREC_SIZES * np.maximum(self.bsize[nr.row], self.bsize[nr.col])
            Pm = sp.csr_matrix((nr.data[keep], (nr.row[keep], nr.col[keep])), shape=nr.shape)
            if nx:  # couple the extras through their strongest entries only
                Zxb = self.Zxb.copy()
                thr = 1e-3 * np.abs(Zxb).max(axis=1, keepdims=True)
                Zxb[np.abs(Zxb) < thr] = 0
                Xs = sp.csr_matrix(Zxb)
                if self.Zab is not None:
                    Za = self.Zab.tocoo()
                    rmax = np.zeros(Za.shape[0])
                    np.maximum.at(rmax, Za.row, np.abs(Za.data))
                    keep = np.abs(Za.data) >= 1e-3 * rmax[Za.row]
                    Xs = sp.vstack([Xs, sp.csr_matrix((Za.data[keep], (Za.row[keep], Za.col[keep])),
                                                      shape=Za.shape)])
                Pm = sp.bmat([[Pm, Xs.T], [Xs, sp.csr_matrix(self.Zxx)]])
            t = time.time()
            try:
                lu = spla.splu(Pm.tocsc(), permc_spec="COLAMD")
                self._prec = lu.solve
                kind = "sparse LU"
            except (MemoryError, RuntimeError):
                ilu = spla.spilu(Pm.tocsc(), drop_tol=1e-5, fill_factor=20)
                self._prec = ilu.solve
                kind = "ILU"
            self.stats["precond"] = f"{kind} {time.time() - t:.1f}s ({Pm.nnz / 1e6:.1f} M nnz)"
        return self._prec

    def solve(self, B, rtol=1e-6, maxiter=3):
        """Z^-1 B (columns of B) by preconditioned GMRES (restart 200, at most
        maxiter restarts: 600 iterations by default). Columns normally take
        30-120 iterations; one that stalls just above rtol (seen: 1.6e-6 after
        2936 iterations, 18 min at N 3.2e4) is accepted if its true residual
        is within the H-matrix accuracy, else it raises."""
        N = self.N
        A = spla.LinearOperator((N, N), matvec=self.matvec, dtype=complex)
        M = spla.LinearOperator((N, N), matvec=self._preconditioner(), dtype=complex)
        B = np.asarray(B, complex)
        X = np.zeros_like(B)
        its = []
        for k in range(B.shape[1]):
            n_it = [0]

            def cb(_):
                n_it[0] += 1
            x, info = spla.gmres(A, B[:, k], M=M, rtol=rtol, restart=200, maxiter=maxiter,
                                 callback=cb, callback_type="pr_norm")
            if info > 0:
                # the H-matrix itself is only accurate to ~self.tol: accept a solution
                # whose true residual reaches that, with a warning
                r = np.linalg.norm(self.matvec(x) - B[:, k]) / np.linalg.norm(B[:, k])
                if r > self.ACCEPT * self.tol:
                    raise RuntimeError(f"GMRES did not converge ({n_it[0]} iterations, rhs {k}, "
                                       f"residual {r:.1e})")
                warnings.warn(f"GMRES stopped at residual {r:.1e} (target {rtol:g}) after {n_it[0]} "
                              f"iterations, rhs {k}: accepted (H-matrix tolerance {self.tol:g})",
                              RuntimeWarning, stacklevel=2)
            X[:, k] = x
            its.append(n_it[0])
            if self.verbose:
                print(f"    GMRES rhs {k + 1}/{B.shape[1]}: {n_it[0]} iterations", flush=True)
        self.stats["gmres_iterations"] = its
        return X

    def summary(self):
        s = self.stats
        za = self.Zab.nnz if getattr(self, "Zab", None) is not None else 0
        ratio = (s["near_nnz"] + s["far_entries"] + self.Zxb.size + self.Zxx.size + za) / s["dense_equiv"]
        return (f"H-matrix: near {s['near_nnz'] / 1e6:.1f} M + far {s['far_entries'] / 1e6:.1f} M entries "
                f"({100 * ratio:.1f} % of dense), near {s['t_near']:.1f}s far {s['t_far']:.1f}s "
                f"extra {s['t_extra']:.1f}s, {s.get('precond', '')}, "
                f"GMRES its {s.get('gmres_iterations', '')}")
