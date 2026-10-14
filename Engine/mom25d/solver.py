"""MPIE / RWG method-of-moments solver and port de-embedding."""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar
from scipy.special import hankel2, kv

from . import _fast, analytic
from .constants import C0, EPS0, MU0
from .greens import GroundedSlabGreens
from .integrals import TRI3_BARY, TRI3_W, TRI7_BARY, TRI7_W, potential_integrals, quad_points


@dataclass
class Substrate:
    """Grounded dielectric slab with conductors on top.

    eps_r, h, tand : substrate permittivity, thickness [m], loss tangent
    sigma, t       : conductor conductivity [S/m] and thickness [m];
                     sigma=None means perfect conductors
    ground_loss    : include the ground plane's conductor loss (same sigma, t);
                     False gives the older strip-only loss
    """

    eps_r: float
    h: float
    tand: float = 0.0
    sigma: float = None
    t: float = 35e-6
    ground_loss: bool = True

    @property
    def lossless(self):
        return not self.tand and not self.sigma

    def surface_impedance(self, freq):
        if not self.sigma:
            return 0.0
        w = 2 * np.pi * freq
        kc = np.sqrt(1j * w * MU0 * self.sigma)
        # one-sided skin-effect impedance of a slab, -> 1/(sigma t) at DC
        return kc / self.sigma / np.tanh(kc * self.t)


    def to_stackup(self):
        from .layered import Stackup

        return Stackup.microstrip(self.eps_r, self.h, self.tand, self.sigma, self.t, self.ground_loss)


# ---------------------------------------------------------------------- vias
def _filament_mutual(l, d):
    """Partial mutual inductance of two parallel, aligned filaments of length
    l at distance d (with d = radius: partial self inductance of a tube)."""
    return MU0 * l / (2 * np.pi) * (np.arcsinh(l / d) - np.sqrt(1 + (d / l) ** 2) + d / l)


def _filament_mutual_z(a1, a2, b1, b2, d):
    """Partial mutual inductance of parallel z-directed filaments spanning
    [a1, a2] and [b1, b2] at lateral distance d (Grover)."""
    f = lambda s: s * np.arcsinh(s / d) - np.sqrt(s * s + d * d)  # noqa: E731
    return MU0 / (4 * np.pi) * (f(a2 - b1) - f(a2 - b2) - f(a1 - b1) + f(a1 - b2))


def via_inductance(vias, h, k=0.0, spans=None):
    """Partial inductance matrix [H] of vertical via currents, including
    their images in the ground plane (quasi-static).

    vias  : objects with .xy and .drill
    h     : via length for grounding vias from z = h to 0 (if spans is None)
    spans : [(z_bottom, z_top), ...] per via, for vias between any levels
    k     : substrate wavenumber for a retarded kernel (grounding vias only;
            the solver keeps k = 0, see CLAUDE.md)
    """
    xy = np.array([v.xy for v in vias], float).reshape(-1, 2)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, [0.5 * v.drill for v in vias])
    if spans is not None:
        z = np.asarray(spans, float).reshape(-1, 2)
        a1, a2 = z[:, None, 0], z[:, None, 1]
        b1, b2 = z[None, :, 0], z[None, :, 1]
        # segment + image of the other segment (vertical images keep direction)
        return _filament_mutual_z(a1, a2, b1, b2, d) + _filament_mutual_z(a1, a2, -b2, -b1, d)
    L = 0.5 * _filament_mutual(2 * h, d)
    if k:
        # smooth remainder Int Int (e^{-jkR} - 1)/R dz dz' over [-h, h]^2
        t, w = np.polynomial.legendre.leggauss(12)
        z, w = h * t, h * w
        dz = z[:, None] - z[None, :]
        R = np.sqrt(d[..., None, None] ** 2 + dz ** 2)
        kr = k * R
        rem = np.where(np.abs(kr) > 1e-8, np.expm1(-1j * kr) / np.where(R > 0, R, 1), -1j * k)
        L = L + 0.5 * MU0 / (4 * np.pi) * np.einsum("...ij,i,j->...", rem, w, w)
    return L


def via_impedance(vias, spans, omega, sections):
    """Impedance matrix [ohm] of the vertical via currents (vector-potential
    part), for via segments in a stack split into sections by PEC planes.

    spans    : [(z_bottom, z_top), ...] per segment (absolute z)
    sections : [(z_bottom, z_top, k_tem or None), ...] (Stackup.via_sections)

    Each segment is cut into its pieces per section; pieces in the same
    section couple, the planes shield sections from each other (antipads
    are not modelled). Open section (air above): quasi-static partial
    inductances of piece + image in the plane below (= via_inductance).
    Closed section (PEC below and above, height d): parallel-plate modes,

        Z_ab = w mu0 l_a l_b / (4 d) H0(2)(k rho)                  (TEM)
             + jw mu0 / (pi d) sum_n K0(q_n rho) c_n(a) c_n(b)    (TM_n)

    c_n = Int_piece cos(n pi z / d) dz, q_n = sqrt((n pi / d)^2 - k^2).
    The TEM term is the radial wave a via launches between the planes; its
    real part is power lost to the (infinite) plane cavity. Self terms use
    rho = drill / 2.
    """
    xy = np.array([v.xy for v in vias], float).reshape(-1, 2)
    rho = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(rho, [0.5 * v.drill for v in vias])
    spans = np.asarray(spans, float).reshape(-1, 2)
    nv = len(spans)
    Z = np.zeros((nv, nv), complex)
    for zb, zt, k in sections:
        a1 = np.clip(spans[:, 0], zb, zt) - zb  # piece in local coordinates
        a2 = np.clip(spans[:, 1], zb, zt) - zb
        on = np.nonzero(a2 - a1 > 1e-12 * (zt - zb))[0]
        if not on.size:
            continue
        a1, a2, r = a1[on], a2[on], rho[np.ix_(on, on)]
        d = zt - zb
        if k is None:
            p1, p2 = a1[:, None], a2[:, None]
            q1, q2 = a1[None, :], a2[None, :]
            L = _filament_mutual_z(p1, p2, q1, q2, r) + _filament_mutual_z(p1, p2, -q2, -q1, r)
            Zs = 1j * omega * L
        else:
            la = a2 - a1
            Zs = omega * MU0 / (4 * d) * np.outer(la, la) * hankel2(0, k * r)
            nmax = int(np.ceil(40 * d / (np.pi * r.min()))) + 1
            n = np.arange(1, nmax + 1)
            kn = n * np.pi / d
            c = (np.sin(kn * a2[:, None]) - np.sin(kn * a1[:, None])) / kn  # (P, n)
            if np.abs(c).max() > 1e-12 * d:  # pieces filling the gap have no TM_n part
                qn = np.sqrt(kn**2 - k**2 + 0j)
                K = kv(0, np.multiply.outer(r, qn))
                Zs = Zs + 1j * omega * MU0 / (np.pi * d) * np.einsum("ijn,in,jn->ij", K, c, c)
        Z[np.ix_(on, on)] += Zs
    return Z


# ------------------------------------------------------- thick-metal side walls
WALL_CUTOFF = 40.0  # wall-wall mutual inductances kept within this many wall heights


def wall_impedance(mesh, stack):
    """Vertical-current part of the side-wall bases of thick metal
    (Mesh.wall, Layout.add_thickness), or None without walls.

    Wall basis m carries the vertical current l_m x_m, uniform over its
    l_m x h strip (h = distance between the layer and its twin level). As
    for vias, horizontal and vertical currents do not couple (quasi-static
    PEEC); wall-wall pairs get jw l_m l_n M_mn with M the partial mutual
    inductance of parallel z filaments (self: at the strip's geometric mean
    distance 0.2235 l) plus their first images in the PEC planes bounding
    the stack section, kept within WALL_CUTOFF * h. Conductor loss: Zs h l
    per wall.

    Returns dict(rows, cols, L [H, incl. l l], hl = h l per wall, idx)."""
    from scipy.spatial import cKDTree

    w = np.nonzero(getattr(mesh, "wall", np.zeros(0, bool)))[0]
    if not w.size:
        return None
    lay = mesh.edge_layer[w]
    twin = mesh.tri_layer[mesh.tm[w]]
    zl = np.array([stack.level_z(n) for n in mesh.layer_names])
    zb, zt = np.minimum(zl[lay], zl[twin]), np.maximum(zl[lay], zl[twin])
    sec = np.array([stack.section(mesh.layer_names[k]) for k in lay])
    zlo = stack.z[np.array(stack.bounds)[sec]]
    zhi = np.array([stack.z[stack.bounds[s + 1]] if stack.section_top_pec(s) else np.nan for s in sec])
    xy, l = mesh.edge_mid[w], mesh.edge_len[w]
    h = zt - zb
    pairs = cKDTree(xy).query_pairs(WALL_CUTOFF * h.max(), output_type="ndarray")
    i = np.r_[np.arange(w.size), pairs[:, 0]]
    j = np.r_[np.arange(w.size), pairs[:, 1]]
    keep = sec[i] == sec[j]
    i, j = i[keep], j[keep]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    d[: w.size] = 0.2235 * l  # self: geometric mean distance of a strip
    # Walls of different metal levels can share an x-y position (e.g. an L1 and an L2 strip edge
    # on the same grid line of a Manhattan mesh): d = 0 makes the filament formula 0 * inf = NaN.
    # Treat close pairs as strips: no closer than the geometric mean distance of the shorter one.
    d = np.maximum(d, 0.2235 * np.minimum(l[i], l[j]))
    a1, a2, b1, b2 = zb[i], zt[i], zb[j], zt[j]
    M = _filament_mutual_z(a1, a2, b1, b2, d)
    M += _filament_mutual_z(a1, a2, 2 * zlo[j] - b2, 2 * zlo[j] - b1, d)  # vertical images keep direction
    up = ~np.isnan(zhi[j])
    M[up] += _filament_mutual_z(a1[up], a2[up], 2 * zhi[j][up] - b2[up], 2 * zhi[j][up] - b1[up], d[up])
    L = l[i] * l[j] * M
    off = i != j
    return dict(idx=w, rows=np.r_[w[i], w[j][off]], cols=np.r_[w[j], w[i][off]], L=np.r_[L, L[off]],
                hl=h * l)


# ---------------------------------------------------------------------- assembly
def _cmatmul(a, b):
    """complex @ real via two real products (numpy's mixed-type batched
    matmul falls back to a very slow non-BLAS loop)."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return np.matmul(a.real, b) + 1j * np.matmul(a.imag, b)


def _source_potentials(kern, r, pts, w, near_static=None):
    """Inner integrals over source triangles at observation points r (M,2).

    kern : kernel pair (smooth(rho) -> S_A, S_V; leading cA, cV / (4 pi R),
           R = sqrt(rho^2 + dz^2))
    Returns K0A (Nq,M), K1A (Nq,M,2), K0V (Nq,M):
        K0A = Int gA dS',  K1A = Int r' gA dS',  K0V = Int gV dS'.
    Near sources (near_static given): the 1/R part is the precomputed
    closed-form (K0, K1) pair, the smooth remainder is done by quadrature.
    Far sources: plain quadrature of the full kernel.
    """
    d = r[None, :, None, :] - pts[:, None, :, :]  # (Nq,M,L,2)
    R = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    SA, SV = kern.smooth(R)
    if near_static is not None:
        S0, S1 = near_static
        K0A, K1A, K0V = kern.cA * S0, kern.cA * S1, kern.cV * S0
    else:
        inv = 1 / (4 * np.pi * (R if not kern.dz else np.sqrt(R * R + kern.dz**2)))
        SA = SA + kern.cA * inv
        SV = SV + kern.cV * inv
        K0A = K1A = K0V = 0.0
    SAw = SA * w[:, None, :]
    K0A = K0A + SAw.sum(-1)
    K1A = K1A + _cmatmul(SAw, pts)
    K0V = K0V + np.einsum("qml,ql->qm", SV, w)
    return K0A, K1A, K0V


def _morton_order(xy):
    q = ((xy - xy.min(0)) / max(np.ptp(xy, 0).max(), 1e-30) * 65535).astype(np.uint64)

    def spread(v):
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x55555555)
        return v

    return np.argsort(spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)), kind="stable")


class AssemblyGeometry:
    """Frequency-independent part of `assemble` for one mesh: quadrature
    points, RWG coefficients, test-block ordering (blocks stay on one metal
    layer), near/mid/far source classification per source layer and the
    closed-form 1/R integrals of near pairs. Build it once and pass it to
    every `assemble` call of a sweep.

    zl : height of each mesh layer (mesh.layer_names order); only the
         differences matter. Default: all layers at the same height.
    via_z : {layer / plane name: z} incl. "gnd": 0, for via segment lengths.
    section : stack section of each mesh layer; layers of different
         sections (separated by a solid plane) do not interact.
    """

    def __init__(self, mesh, block=32, near_factor=3.0, far_factor=10.0, zl=None, via_z=None,
                 section=None, blocks=True, walls=None):
        V, A = mesh.verts, mesh.area
        self.walls = walls  # wall_impedance(mesh, stack) for thick metal
        self.nt, self.nb = nt, nb = len(A), mesh.n_basis
        self.V = V
        self.qpts = qpts = quad_points(V, TRI7_BARY)  # (Nt,7,2)
        self.qw = qw = TRI7_W[None, :] * A[:, None]
        mpts = quad_points(V, TRI3_BARY)  # (Nt,3,2) mid-range inner rule
        mw = TRI3_W[None, :] * A[:, None]
        fpts = mesh.centroid[:, None, :]  # (Nt,1,2) far-field inner rule
        fw = A[:, None]
        size = np.max(np.linalg.norm(V - np.roll(V, 1, axis=1), axis=-1), axis=1)
        cen = mesh.centroid
        tl = mesh.tri_layer
        nl = len(mesh.layer_names)
        zl = np.zeros(nl) if zl is None else np.asarray(zl, float)
        self.levels = np.arange(nl)

        self.b = b = mesh.tri_basis
        lab = np.where(b >= 0, mesh.edge_len[np.maximum(b, 0)], 0.0)
        self.cA = cA = mesh.tri_sign * lab / (2 * A[:, None])  # RWG amplitude  l/(2A)
        self.cV = mesh.tri_sign * lab / A[:, None]  # divergence      l/A

        # half-basis -> basis map (3Nt x Nb)
        valid = (b >= 0).ravel()
        P = sp.csr_matrix((np.ones(valid.sum()), (np.nonzero(valid)[0], b.ravel()[valid])),
                          shape=(3 * nt, nb))
        self.PT = P.T.tocsr()

        self.zl, self.size, self.section = zl, size, section
        self.mpts, self.mw, self.fpts, self.fw = mpts, mw, fpts, fw
        self.blocks = []
        for lt in range(nl if blocks else 0):  # blocks=False: skip the O(N^2) dense setup (hmatrix.py)
            on_t = np.nonzero(tl == lt)[0]
            if not on_t.size:
                continue
            order = on_t[_morton_order(cen[on_t])]
            for p0 in range(0, order.size, block):
                pb = order[p0:p0 + block]
                r = qpts[pb].reshape(-1, 2)  # (npb*K, 2)
                groups = []
                for ls in range(nl):
                    src = np.nonzero(tl == ls)[0]
                    if not src.size or (section is not None and section[ls] != section[lt]):
                        continue
                    dz = abs(zl[lt] - zl[ls])
                    dist = np.linalg.norm(cen[pb][:, None, :] - cen[None, src, :], axis=-1)
                    dist = np.sqrt(dist**2 + dz**2)
                    rel = np.min(dist / np.maximum(size[pb][:, None], size[None, src]), axis=0)
                    near, mid, far = src[rel < near_factor], src[(rel >= near_factor) & (rel < far_factor)], \
                        src[rel >= far_factor]
                    if near.size:
                        I0, I1 = potential_integrals(V[near], r, dz)
                        static = (I0 / (4 * np.pi), (r[None] * I0[..., None] + I1) / (4 * np.pi))
                        groups.append((ls, near, qpts[near], qw[near], static))
                    for sel, pts, w in ((mid, mpts, mw), (far, fpts, fw)):
                        if sel.size:
                            groups.append((ls, sel, pts[sel], w[sel], None))
                # per test triangle p (batched): Xw[p,i,k,:] = w_k (r_k - a_i)
                wo = qw[pb]
                Xw = (qpts[pb][:, None, :, :] - V[pb][:, :, None, :]) * wo[:, None, :, None]
                self.blocks.append((lt, pb, r, groups, wo, Xw))

        # conductor-loss Gram matrix Int_T (r-a_i).(r-a_j) dS, times cA cA
        X = qpts[:, None, :, :] - V[:, :, None, :]  # (Nt,3,K,2)
        G = np.einsum("tk,tikd,tjkd->tij", qw, X, X)
        ii = np.broadcast_to(b[:, :, None], G.shape)
        jj = np.broadcast_to(b[:, None, :], G.shape)
        ok = (ii >= 0) & (jj >= 0)
        self.gram = (ii[ok], jj[ok], (cA[:, :, None] * cA[:, None, :] * G)[ok])

        # via segments: each drains its upper triangles (+1/A divergence) and
        # feeds its lower ones (-1/A); Dvia[pad row, k]
        segs = getattr(mesh, "via_segments", [])
        self.nv = nv = len(segs)
        if nv:
            self.vias = [mesh.vias[g["via"]] for g in segs]
            tris = [g[e] for g in segs for e in ("top_tris", "bottom_tris") if g[e] is not None]
            self.pad_tris = pad = np.unique(np.concatenate(tris)) if tris else np.zeros(0, int)
            self.pad_pos = -np.ones(nt, int)
            self.pad_pos[pad] = np.arange(pad.size)
            self.Dvia = np.zeros((pad.size, nv))
            # end charges per level: (level index, sign) -- for the Im gV(0) fix
            self.via_ends = []
            for k, g in enumerate(segs):
                ends = []
                t = g["top_tris"]
                if t is not None:  # None: the segment starts on a plane
                    self.Dvia[self.pad_pos[t], k] += 1 / A[t].sum()
                    ends.append((tl[t[0]], 1.0))
                t = g["bottom_tris"]
                if t is not None:
                    self.Dvia[self.pad_pos[t], k] -= 1 / A[t].sum()
                    ends.append((tl[t[0]], -1.0))
                self.via_ends.append(ends)
            if via_z is None:
                via_z = {n: 0.0 for n in mesh.layer_names}
                via_z["gnd"] = None
            self.via_spans = None
            if all(via_z.get(n) is not None for g in segs for n in (g["top"], g["bottom"])):
                self.via_spans = [(via_z[g["bottom"]], via_z[g["top"]]) for g in segs]
            ok = b >= 0
            self.Cdiv = sp.csr_matrix((self.cV[ok], (np.nonzero(ok)[0], b[ok])), shape=(nt, nb))


def assemble(mesh, greens, zs=0.0, block=32, near_factor=3.0, far_factor=10.0, geom=None, zs_via=None):
    """Galerkin MPIE impedance matrix Z (N x N) for RWG basis functions on
    all metal layers (N = Nb RWG edges, followed by one unknown per vertical
    via segment).

    Z_mn = jw mu0 <f_m, gA * f_n> + 1/(jw eps0) <div f_m, gV * div f_n>
           + Zs <f_m, f_n>

    with the kernels of the (test layer, source layer) pair, greens.pair().
    A via segment is a vertical current between two layers (or a layer and
    the ground plane); on each layer it is a half-RWG that drains / feeds
    the triangles inside the hole with a uniform charge. Its charges couple
    to everything through gV like RWG charges; its vertical current only
    couples to other vertical currents (`via_impedance`: quasi-static
    partial inductances of segment + image under an open top, parallel-plate
    modes between two planes) -- horizontal and vertical currents
    have no mutual inductance in the quasi-static Lorenz gauge (PEEC), and
    the radial current on the pads is neglected in the vector-potential
    term. Valid for k1 h << 1. The via wall adds Zs * length / (pi drill).

    zs: a number, or one value per RWG basis (thick metal); zs_via: the
    via-wall value (default zs, which must then be a number). Side walls of
    thick metal (geom.walls) add their vertical-current terms.

    Source triangles closer than near_factor * (triangle size) to a block of
    test triangles get singularity extraction and a 7-point inner rule, those
    closer than far_factor * size a 3-point rule on the full kernel, and the
    rest a single centroid point (3D distances between layers). `geom` (an
    AssemblyGeometry for this mesh) skips the frequency-independent setup.
    """
    geom = geom or AssemblyGeometry(mesh, block, near_factor, far_factor)
    omega = greens.omega
    nt, nb, V, b = geom.nt, geom.nb, geom.V, geom.b
    cA, cV = geom.cA, geom.cV
    K = TRI7_W.size
    jwmu = 1j * omega * MU0
    inv_jwe = 1 / (1j * omega * EPS0)
    nv = geom.nv
    Z = np.zeros((nb + nv, nb + nv), complex)
    Zb = Z[:nb, :nb]  # RWG block (view)
    if nv:
        IVpad = np.zeros((geom.pad_tris.size, nt), complex)

    if _fast.ENABLED:
        alltri = np.arange(nt)
    for lt, pb, r, groups, wo, Xw in geom.blocks:
        npb = pb.size
        if _fast.ENABLED:
            vals, IV = _fast.chunk_vals(greens, lt, pb, r, groups, wo, V, cA, cV, alltri, omega, jwmu, inv_jwe)
            _fast.scatter(Zb, vals, b[pb], b)
            if nv:
                pos = geom.pad_pos[pb]
                sel = pos >= 0
                IVpad[pos[sel]] = IV[sel]
            continue
        K0A = np.zeros((nt, npb * K), complex)
        K1A = np.zeros((nt, npb * K, 2), complex)
        K0V = np.zeros((nt, npb * K), complex)
        for ls, sel, pts, w, static in groups:
            K0A[sel], K1A[sel], K0V[sel] = _source_potentials(greens.pair(lt, ls), r, pts, w, static)

        K0A = K0A.reshape(nt, npb, K).transpose(1, 0, 2)  # (npb,Nt,K)
        K1A = K1A.reshape(nt, npb, K, 2).transpose(1, 0, 2, 3)
        # IA[p,q,i,j] = sum_k w_k (r_k - a_i) . (K1A - b_j K0A)
        T1 = _cmatmul(K1A.reshape(npb, nt, 2 * K), Xw.reshape(npb, 3, 2 * K).transpose(0, 2, 1))
        T2 = _cmatmul(K0A, Xw.transpose(0, 2, 1, 3).reshape(npb, K, 6)).reshape(npb, nt, 3, 2)
        IA = T1[..., None] - np.einsum("pqid,qjd->pqij", T2, V, optimize=True)
        IV = np.einsum("qpk,pk->pq", K0V.reshape(nt, npb, K), wo)
        if nv:
            pos = geom.pad_pos[pb]
            sel = pos >= 0
            IVpad[pos[sel]] = IV[sel]

        vals = (jwmu * cA[pb][:, None, :, None] * cA[None, :, None, :] * IA
                + inv_jwe * cV[pb][:, None, :, None] * cV[None, :, None, :] * IV[:, :, None, None])
        # (npb,Nt,3,3) -> rows (npb*3) x cols (Nt*3)
        H = vals.transpose(0, 2, 1, 3).reshape(npb * 3, nt * 3)
        HB = (geom.PT @ H.T).T  # (npb*3, Nb)
        rows = b[pb].ravel()
        ok = rows >= 0
        np.add.at(Zb, rows[ok], HB[ok])

    if np.any(zs):
        ii, jj, g = geom.gram
        np.add.at(Z, (ii, jj), (zs[ii] if np.ndim(zs) else zs) * g)
    W = geom.walls
    if W is not None:
        np.add.at(Z, (W["rows"], W["cols"]), jwmu / MU0 * W["L"])
        if np.any(zs):
            Z[W["idx"], W["idx"]] += (zs[W["idx"]] if np.ndim(zs) else zs) * W["hl"]
    if nv:
        Z[nb:, :nb], Z[nb:, nb:] = _via_blocks(geom, greens, IVpad, zs if zs_via is None else zs_via)
        Z[:nb, nb:] = Z[nb:, :nb].T
    # the exact Galerkin matrix is symmetric (reciprocity); averaging removes
    # the small asymmetry left by outer quadrature of near-singular terms
    return 0.5 * (Z + Z.T)


def _via_blocks(geom, greens, IVpad, zs=0.0):
    """Via rows of Z: (nv x Nb) coupling to the RWG unknowns and the (nv x nv)
    via block, from IVpad = <1_T, gV * 1_T'> for the pad triangles T against
    all triangles T'."""
    omega = greens.omega
    jwmu = 1j * omega * MU0
    inv_jwe = 1 / (1j * omega * EPS0)
    nv = geom.nv
    D = geom.Dvia
    Y = (geom.Cdiv.T @ IVpad.T).T  # (Npad, Nb): <1_pad, gV * div f_n>
    Zvb = inv_jwe * (D.T @ Y)
    Zvv = inv_jwe * (D.T @ IVpad[:, geom.pad_tris] @ D)
    # A segment's end charges with its vertical current form a short
    # vertical dipole that hardly radiates. The constant Im gV(0) (space
    # + surface wave) acting on its non-neutral layer charges would be a
    # spurious ~0.1 ohm radiation resistance per via (neutral RWG charges
    # cancel it exactly, so only via-via pairs carry it): remove it.
    im0 = {}
    for k, ek in enumerate(geom.via_ends):
        for m, em in enumerate(geom.via_ends):
            acc = 0.0
            for la, sa in ek:
                for lb, sb in em:
                    if (la, lb) not in im0:
                        pr = greens.pair(la, lb)
                        rho = 0.0 if pr.dz else 1e-6 * greens.h
                        im0[la, lb] = pr.gV(np.array([rho]))[0].imag
                    acc += sa * sb * im0[la, lb]
            Zvv[k, m] -= inv_jwe * 1j * acc
    if geom.via_spans is None:
        Zvv += jwmu / MU0 * via_inductance(geom.vias, greens.h)
        lengths = np.full(nv, greens.h)
    else:
        vs = getattr(greens, "via_sections", None)
        sections = vs() if vs is not None else [(0.0, greens.h, None)]
        Zvv += via_impedance(geom.vias, geom.via_spans, omega, sections)
        lengths = np.array([z1 - z0 for z0, z1 in geom.via_spans])
    if zs:
        Zvv += np.diag([zs * l / (np.pi * v.drill) for l, v in zip(lengths, geom.vias)])
    return Zvb, Zvv


# ------------------------------------------------------------------------- ports
class PortModel:
    """Delta-gap excitation plus current / voltage sampling for one port."""

    def __init__(self, mesh, port, h):
        self.port = port
        margin = port.margin if port.margin is not None else max(1.5 * port.width, 4 * h)
        s = mesh.cut_positions(port)
        s = s[(s >= margin - 1e-12) & (s <= port.s_gap - margin + 1e-12)]
        if s.size < 4:
            raise ValueError(f"port {port.name}: feed line too short or mesh too coarse "
                             f"({s.size} sample cuts in the fitting window "
                             f"[{margin * 1e3:.3g}, {(port.s_gap - margin) * 1e3:.3g}] mm)")
        self.s = s
        self.cuts = [mesh.cut_edges(port, si) for si in s]
        idx, sgn = mesh.cut_edges(port, port.s_gap)
        if idx.size == 0:
            raise ValueError(f"port {port.name}: no mesh edges on the gap cut")
        self.excitation = np.zeros(mesh.n_unknowns)
        self.excitation[idx] = sgn * mesh.edge_len[idx]
        self.len = mesh.edge_len
        # line voltage (scalar potential w.r.t. ground) is sampled at points
        # across the strip, half a cell on either side of every cut; averaging
        # both sides cancels the ripple from the alternating cell diagonals
        allcuts = mesh.cut_positions(port)
        k = np.searchsorted(allcuts, s)
        lo = s - allcuts[np.maximum(k - 1, 0)]
        hi = allcuts[np.minimum(k + 1, allcuts.size - 1)] - s
        d = 0.5 * np.minimum(np.where(lo > 0, lo, np.inf), np.where(hi > 0, hi, np.inf))
        frac = np.array([-0.3, -0.1, 0.1, 0.3])
        along = np.stack([s - d, s + d], 1)  # (Ns,2)
        pts = port.point(along[:, :, None], frac * port.width)  # (Ns,2,nfrac,2)
        self.vpts = pts.reshape(-1, 2)
        self.nper = 2 * frac.size

    def currents(self, x):
        """Total current toward the circuit at each sample cut; x (Nb, Nexc)."""
        return np.array([(sg * self.len[i]) @ x[i] for i, sg in self.cuts])

    def voltages(self, phi):
        """phi (Ns*nfrac, Nexc) potentials at vpts -> (Ns, Nexc) cut averages."""
        return phi.reshape(self.s.size, self.nper, -1).mean(1)


# relative weight of the voltage residual in the gamma fit (the voltage data
# is what keeps gamma well conditioned when the fitting window is short)
VOLTAGE_WEIGHT = 1.0


def fit_waves(s_list, I_list, V_list, gamma0, lossless=False, v_weight=VOLTAGE_WEIGHT,
              bounds=(0.7, 1.3)):
    """Fit transmission-line waves to sampled currents and voltages.

        I(s) = A e^{gamma s} + B e^{-gamma s}            (toward the circuit)
        V(s) = Z0 (A e^{gamma s} - B e^{-gamma s})

    All data sets share gamma and Z0; A, B are per data set. Fitting V and I
    together pins gamma and Z0 down even when the window is electrically
    short (dI/ds ~ gamma V / Z0, dV/ds ~ gamma Z0 I).

    lossless=True constrains gamma = j*beta (the microstrip mode is bound, so
    any fitted attenuation would be spurious and bias the de-embedding);
    beta is searched in bounds x Im(gamma0).

    Returns gamma, Z0, [(A, B), ...], relative residual.
    """
    scale = abs(gamma0.imag)
    Inrm = sum(np.sum(np.abs(I) ** 2) for I in I_list)
    Vnrm = sum(np.sum(np.abs(V) ** 2) for V in V_list)

    def solve(g):
        coefs, res, U_all, V_all = [], 0.0, [], []
        for s, I, V in zip(s_list, I_list, V_list):
            ep, em = np.exp(g * s), np.exp(-g * s)
            M = np.stack([ep, em], 1)
            c, *_ = np.linalg.lstsq(M, I, rcond=None)
            coefs.append(c)
            res += np.sum(np.abs(M @ c - I) ** 2) / Inrm
            U_all.append(c[0] * ep - c[1] * em)
            V_all.append(V)
        U, V = np.concatenate(U_all), np.concatenate(V_all)
        z0 = np.vdot(U, V) / np.vdot(U, U)
        res += v_weight * np.sum(np.abs(V - z0 * U) ** 2) / Vnrm
        return coefs, z0, res

    if lossless:
        lo, hi = bounds
        opt = minimize_scalar(lambda b: solve(1j * scale * b)[2], bounds=(lo, hi),
                              method="bounded", options=dict(xatol=1e-12))
        if min(opt.x - lo, hi - opt.x) < 1e-3:
            warnings.warn(f"beta fit hit its search bound ({opt.x:.4f} x the Kirschning-Jansen "
                          "estimate): gamma, Z0 and S at this frequency are unreliable",
                          RuntimeWarning, stacklevel=2)
        g = 1j * scale * opt.x
    else:
        obj = lambda p: solve(scale * (p[0] + 1j * p[1]))[2]  # noqa: E731
        p0 = np.array([gamma0.real, gamma0.imag]) / scale
        opt = minimize(obj, p0, method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-16, maxiter=4000))
        g = scale * (opt.x[0] + 1j * opt.x[1])
    coefs, z0, res = solve(g)
    return g, z0, coefs, np.sqrt(res)


@dataclass
class SweepResult:
    freqs: np.ndarray
    S: np.ndarray  # (F,P,P), referenced to each port's line impedance z0
    z0: np.ndarray  # (F,P) reference (line) impedances used for S
    gamma: np.ndarray  # (F,P) fitted propagation constants
    z0_line: np.ndarray  # (F,P) line impedances extracted by the MoM
    fit_residual: np.ndarray  # (F,P)
    port_names: list
    currents: list = None  # per frequency: (Nb, P) RWG coefficients
    waves: np.ndarray = None  # (F,P,P) incident current-wave amplitudes A

    def matched_currents(self, fi, port):
        """RWG coefficients at frequency index fi for a unit incident current
        wave on `port` with all other ports terminated in their line
        impedance (a superposition of the delta-gap solutions)."""
        if self.currents is None:
            raise ValueError("run sweep(..., keep_currents=True)")
        e = np.zeros(self.waves.shape[1])
        e[port] = 1.0
        return self.currents[fi] @ np.linalg.solve(self.waves[fi], e)

    @property
    def eeff(self):
        k0 = 2 * np.pi * self.freqs / C0
        return (self.gamma.imag / k0[:, None]) ** 2

    def renormalize(self, zref=50.0):
        """S-parameters referenced to a common real impedance zref."""
        out = np.empty_like(self.S)
        I = np.eye(self.S.shape[1])
        for f, S in enumerate(self.S):
            sq = np.diag(np.sqrt(self.z0[f]))
            Zm = sq @ (I + S) @ np.linalg.inv(I - S) @ sq
            out[f] = (Zm - zref * I) @ np.linalg.inv(Zm + zref * I)
        return out

    def connect(self, model, pairs, z0=None, name=None):
        """Connect S-parameter blocks between ports of this result.

        model : lumped.Touchstone or a .sNp path (e.g. an amplifier whose
                pads are two ports of the layout)
        pairs : [(layout port name or index, block port number 1..n), ...]
        z0    : common reference impedance (default the block's)
        Returns a network.Network on the remaining ports (layout ports first,
        then unconnected block ports named '<name>.<k>')."""
        from .lumped import Touchstone
        from .network import Network, connect

        if isinstance(model, str):
            model = Touchstone(model)
        z0 = z0 or getattr(model, "z0", 50.0)
        name = name or getattr(model, "path", "block").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        SA = self.renormalize(z0)
        SB = np.array([model.s(f) for f in self.freqs])
        na = SA.shape[-1]
        pa = [(self.port_names.index(a) if isinstance(a, str) else int(a), int(b) - 1) for a, b in pairs]
        S, idx = connect(SA, SB, pa)
        names = [self.port_names[i] if i < na else f"{name}.{i - na + 1}" for i in idx]
        net = Network(self.freqs, S, names, z0)
        net._source = (self, model, pa, idx, z0)  # for Network.currents
        return net

    def wave_currents(self, fi, a50, z0=50.0):
        """RWG coefficients at frequency index fi for incident power waves
        a50 (one per port, reference z0) at the layout ports."""
        if self.currents is None:
            raise ValueError("run sweep(..., keep_currents=True)")
        a50 = np.asarray(a50, complex)
        b50 = self.renormalize(z0)[fi] @ a50
        V = np.sqrt(z0) * (a50 + b50)  # port voltage and current (into the layout)
        I = (a50 - b50) / np.sqrt(z0)
        A = 0.5 * (I + V / self.z0[fi])  # incident current waves of the line reference
        return self.currents[fi] @ np.linalg.solve(self.waves[fi], A)

    def db(self, i, j, S=None):
        S = self.S if S is None else S
        return 20 * np.log10(np.abs(S[:, i, j]))

    def write_touchstone(self, path, zref=50.0):
        """Write S renormalized to zref (None: raw, referenced to line z0)."""
        S = self.S if zref is None else self.renormalize(zref)
        z = zref if zref is not None else float(np.mean(self.z0))
        n = S.shape[1]
        with open(path, "w") as fh:
            fh.write(f"! mom25d 2.5D MoM. ports: {', '.join(self.port_names)}\n")
            if zref is None:
                fh.write("! referenced to the port line impedances (see z0)\n")
            fh.write(f"# HZ S RI R {z:g}\n")
            for f, Sf in zip(self.freqs, S):
                # Touchstone 2-port order is S11 S21 S12 S22 (column-major)
                vals = Sf.T.ravel() if n == 2 else Sf.ravel()
                fh.write(f"{f:.9e} " + " ".join(f"{v.real:.9e} {v.imag:.9e}" for v in vals) + "\n")


class Solver:
    def __init__(self, mesh, substrate, z0=None, verbose=True, method="auto", v_weight=None,
                 z0_recip_min=None):
        """mesh      : Mesh built from a Layout with ports
        substrate : Substrate (one dielectric, copper on "top") or Stackup
                    (any number of layers; every mesh layer must be one of its
                    metal levels)
        z0        : None -> S is referenced to the MoM-extracted line impedance
                    of each port (recommended); or a sequence of fixed
                    per-port reference impedances.
        method    : "dense" (Z + LU), "hmatrix" (compressed Z + GMRES,
                    hmatrix.py) or "auto": hmatrix above HMATRIX_N unknowns
                    or when a dense Z would not fit in memory.
        v_weight  : weight of the line voltage in the wave fit (fit_waves;
                    default VOLTAGE_WEIGHT = 1). 0 fits the feed currents
                    only: use it when some port's line voltage is not valid
                    (with Port(z0_from="reciprocity")) and the feeds are long
                    enough (>~ lambda_g / 2) for I alone to fix gamma.
        z0_recip_min : reciprocity sets Z0 ratios only between ports coupled
                    by at least this much (min(|SI_ij|, |SI_ji|)); default None:
                    every coupling, weighted by its square. Couplings that are
                    not reciprocal in the solution (e.g. to a port whose fit
                    window sees the near field of a discontinuity) then
                    override good line-voltage estimates: a 79-ohm port came
                    out at 90-116 ohm from couplings |SI| <= 0.2 that were
                    non-reciprocal by up to 0.28. Set it between the weak and
                    the strong couplings (0.3 there)."""
        from .layered import Stackup

        self.mesh = mesh
        self.sub = substrate
        self.stack = substrate if isinstance(substrate, Stackup) else substrate.to_stackup()
        st = self.stack
        missing = [n for n in mesh.layer_names if n not in st.metals]
        if missing:
            raise ValueError(f"mesh layers {missing} are not metal levels of the stackup "
                             f"{list(st.metals)} (declare inner planes with Layout.add_plane)")
        planes = list(getattr(mesh, "planes", []))
        missing = [n for n in planes if n not in st.planes]
        if missing:
            raise ValueError(f"layout planes {missing} are not planes of the stackup {list(st.planes)}")
        # one dielectric with copper on top: the original single-slab kernels
        self.single = len(st.layers) == 1 and st.top == "air" and mesh.layer_names == ["top"]
        self.zl = np.array([st.level_z(n) for n in mesh.layer_names])
        self.thick = {}  # thick metal: mesh layer index -> copper thickness [m]
        for li, lj in enumerate(getattr(mesh, "twin_layer", [])):
            if lj < 0:
                continue
            a, b = mesh.layer_names[li], mesh.layer_names[lj]
            if st.section(a) != st.section(b):
                raise ValueError(f"thick metal {a!r}: twin level {b!r} is in another stack section")
            za, zb = sorted((st.level_z(a), st.level_z(b)))
            between = [n for n in list(st.metals) + list(st.planes) if za < st.level_z(n) < zb]
            if between:
                raise ValueError(f"thick metal {a!r} / {b!r}: levels {between} lie between them")
            self.thick[li] = self.thick[lj] = zb - za
        self._walls = wall_impedance(mesh, st) if self.thick else None
        order = sorted(mesh.layer_names + planes, key=st.level_z)
        if getattr(mesh, "level_order", None) != order:
            mesh.assign_levels(order)
        self.ports = mesh.layout.ports
        # internal ports (Layout.add_gap_port / add_via_port): S-parameter ports
        # with a fixed reference impedance, excited and terminated across a delta gap
        self.iports = list(getattr(mesh.layout, "internal_ports", []))
        if not self.ports and not self.iports:
            raise ValueError("layout has no ports")
        self.pm = [PortModel(mesh, p, st.level_h(p.layer)) for p in self.ports]
        # lumped components: internal delta-gap ports, loaded after the solve;
        # via ports add their own columns after the lumped gaps (gap ports are lumped gaps)
        self.lumped = list(getattr(mesh.layout, "lumped", []))
        self.components = list(getattr(mesh.layout, "components", []))
        self.Lm = mesh.lumped_vectors() if self.lumped else None
        self.iport_index = []  # column of each internal port in Lm
        if self.iports:
            names = [q.name for q in self.lumped]
            V = mesh.internal_port_vectors()
            extra = [k for k, ip in enumerate(self.iports) if ip["kind"] == "via"]
            if extra:
                self.Lm = V[:, extra] if self.Lm is None else np.hstack([self.Lm, V[:, extra]])
            col = {k: len(names) + i for i, k in enumerate(extra)}
            self.iport_index = [names.index(ip["name"]) if ip["kind"] == "gap" else col[k]
                                for k, ip in enumerate(self.iports)]
        self.values = {}  # per-sweep overrides of lumped values
        if method not in ("auto", "dense", "hmatrix"):
            raise ValueError("method must be 'auto', 'dense' or 'hmatrix'")
        if method == "auto":
            from .freqinterp import _available_memory

            n = mesh.n_unknowns
            method = "hmatrix" if (n > self.HMATRIX_N or 3 * 16 * n * n > 0.6 * _available_memory()) else "dense"
        self.method = method
        self._hop = None
        self.z0_fixed = None if z0 is None else np.asarray(z0, float)
        v = mesh.nodes
        self.rho_max = 1.02 * np.hypot(*(v.max(0) - v.min(0)))
        self.verbose = verbose
        if v_weight is None and getattr(mesh, "n_ap", 0):
            # plane apertures: the line voltage ignores the aperture field and is not
            # proportional to the true voltage along a feed over an opening (fit residual
            # 1e-2 -> 1e-3 without it); fit the currents only (feeds >~ lambda_g / 2)
            v_weight = 0.0
        self.v_weight = VOLTAGE_WEIGHT if v_weight is None else float(v_weight)
        self.z0_recip_min = z0_recip_min
        self._geom = None  # AssemblyGeometry, built on the first solve
        # ports with the same line width and layer share gamma and Z0
        self.groups = {}
        for p, port in enumerate(self.ports):
            hp = st.level_h(port.layer)  # same line: width, layer and coplanar grounds
            key = (round(port.width / hp, 9), port.layer,
                   port.ground.key(hp) if port.ground is not None else None, port.z0_from)
            self.groups.setdefault(key, []).append(p)

    @property
    def lossless(self):
        return self.stack.lossless

    def surface_impedance(self, freq):
        return self.stack.surface_impedance(freq)

    def _zs(self, freq):
        """(zs for the RWG bases, zs for via walls). Thick metal: each sheet
        of a thick layer is one face of the copper, a slab of half its
        thickness (DC: the two sheets in parallel give 1/(sigma t))."""
        z = self.stack.surface_impedance(freq)
        if not self.thick or not z:
            return z, z
        m = self.mesh
        per = np.full(len(m.layer_names), z, complex)
        for li, t in self.thick.items():
            per[li] = self.stack.surface_impedance(freq, 0.5 * t)
        return per[m.edge_layer], z

    def greens(self, freq):
        st = self.stack
        zg = st.surface_impedance(freq) if st.ground_loss else 0.0
        if self.single:
            d = st.layers[0]
            return GroundedSlabGreens(freq, d.eps_r, d.t, d.tand, self.rho_max, zs_ground=zg)
        from .layered import make_greens

        return make_greens(freq, st, self.rho_max, zs_ground=zg, levels=self.mesh.layer_names)

    def geometry(self):
        if self._geom is None:
            st = self.stack
            names = self.mesh.layer_names + list(getattr(self.mesh, "planes", []))
            via_z = {n: st.level_z(n) for n in names}
            via_z["gnd"] = 0.0
            section = [st.section(n) for n in self.mesh.layer_names]
            self._geom = AssemblyGeometry(self.mesh, zl=self.zl, via_z=via_z, section=section,
                                          walls=self._walls)
        return self._geom

    def impedance(self, freq, greens=None):
        """The MoM impedance matrix at freq (with antipad frill rows/columns
        after the via segments when vias pass planes through antipads)."""
        g = greens or self.greens(freq)
        zs, zv = self._zs(freq)
        Z = assemble(self.mesh, g, zs, geom=self.geometry(), zs_via=zv)
        nf = getattr(self.mesh, "n_frills", 0)
        if nf:
            from .antipad import FrillModel

            if getattr(self, "_frills", None) is None:
                self._frills = FrillModel(self)
            C, F = self._frills.blocks(g, g.omega)
            nj = Z.shape[0]
            Zx = np.zeros((nj + nf, nj + nf), complex)
            Zx[:nj, :nj] = Z
            del Z
            # Z = -<f, E(J)>, so the frill field enters the current rows as -C V;
            # the frill rows (H continuity) are negated to keep Z symmetric
            Zx[:nj, nj:] = -C
            Zx[nj:, :nj] = -C.T
            Zx[nj:, nj:] = -0.5 * (F + F.T)
            Z = Zx
        if getattr(self.mesh, "n_ap", 0):
            B, Y = self.apertures().blocks(freq)
            nj, na = Z.shape[0], Y.shape[0]
            Zx = np.zeros((nj + na, nj + na), complex)
            Zx[:nj, :nj] = Z
            del Z
            B = B.toarray()
            Zx[:nj, nj:] = B
            Zx[nj:, :nj] = B.T
            Zx[nj:, nj:] = 0.5 * (Y + Y.T)
            return Zx
        return Z

    def apertures(self):
        """ApertureModel of the layout's plane apertures (aperture.py)."""
        if getattr(self, "_apm", None) is None:
            from .aperture import ApertureModel

            self._apm = ApertureModel(self)
        return self._apm

    def solve_currents(self, freq, excitations=None, greens=None, Z=None, interp=None):
        """RWG (+ via segment) coefficients for the given excitations
        (default: one column per port, 1 V delta gap). Z: a precomputed
        impedance matrix (overwritten); interp: a FreqInterp supplying it."""
        t0 = time.time()
        g = greens or self.greens(freq)
        t1 = time.time()
        how = "fill"
        if excitations is None:
            excitations = self.port_excitations()
        if self.method == "hmatrix" and Z is None:
            from .hmatrix import HOperator

            if self._hop is None:
                self._hop = HOperator(self, verbose=self.verbose)
            self._hop.assemble(g, *self._zs(freq))
            t2 = time.time()
            rhs = excitations if self.Lm is None else np.hstack([excitations, self.Lm])
            # each GMRES solve is one column: solve repeated columns once (an internal
            # port's excitation is its own lumped-gap column)
            _, first, inv = np.unique(rhs.T, axis=0, return_index=True, return_inverse=True)
            X = self._hop.solve(rhs[:, first])[:, np.ravel(inv)]
            ne = excitations.shape[1]
            x = X if self.Lm is None else self._load_lumped(X[:, :ne], X[:, ne:], freq)
            if self.verbose:
                print(f"  f = {freq / 1e9:8.4f} GHz  N = {self.mesh.n_unknowns}  green {t1 - t0:.1f}s  "
                      f"{self._hop.summary()}  total {time.time() - t0:.1f}s")
            return x
        if Z is None and interp is not None:
            Z, how = interp.impedance(freq), "interp"
        elif Z is None:
            Z = self.impedance(freq, g)
        t2 = time.time()
        if self.Lm is not None:  # also solve for the lumped gaps, then load them
            ne = excitations.shape[1]
            X = sla.lu_solve(sla.lu_factor(Z, overwrite_a=True, check_finite=False),
                             np.hstack([excitations, self.Lm]))
            x = self._load_lumped(X[:, :ne], X[:, ne:], freq)
        else:
            x = sla.lu_solve(sla.lu_factor(Z, overwrite_a=True, check_finite=False), excitations)
        if self.verbose:
            nv = f" ({self.mesh.n_vias} via segments)" if self.mesh.n_vias else ""
            print(f"  f = {freq / 1e9:8.4f} GHz  N = {self.mesh.n_unknowns}{nv}  "
                  f"green {t1 - t0:.1f}s  {how} {t2 - t1:.1f}s  solve {time.time() - t2:.1f}s")
        return x

    def port_excitations(self):
        """One column per port: 1 V delta gap at each feed, then a 1 V source
        (in series with z_ref) at each internal port."""
        cols = [pm.excitation for pm in self.pm]
        cols += [self.Lm[:, c] for c in self.iport_index]
        return np.stack(cols, 1)

    OPEN = 1e12  # ohm: an unloaded lumped gap
    HMATRIX_N = 7000  # "auto": compressed solver above this many unknowns

    def lumped_impedance(self, freq):
        """D: (n_lumped x n_lumped) impedance of the components across the
        lumped gaps at freq (open gaps: OPEN)."""
        from .lumped import as_value, parse

        nl = 0 if self.Lm is None else self.Lm.shape[1]
        D = np.zeros((nl, nl), complex)
        names = [q.name for q in self.lumped]
        used = set()
        for ip, c in zip(self.iports, self.iport_index):  # internal ports: terminated in z_ref
            D[c, c] = ip["z_ref"]
            used.add(c)
        for model, gaps in self.components:
            ix = [names.index(n) for n in gaps]
            D[np.ix_(ix, ix)] = model.z(freq)
            used.update(ix)
        for k, q in enumerate(self.lumped):
            v = self.values.get(q.name, q.value)
            if k in used and (v is None or k in self.iport_index):
                continue
            if isinstance(v, str):
                v = parse(v)
            D[k, k] = self.OPEN if v is None else as_value(v).z(freq)
        return D

    def _load_lumped(self, Xe, Xi, freq):
        """Solution with the components across the lumped gaps: (Z + L D L^T)^-1
        e = Xe - Xi D (I + L^T Xi D)^-1 L^T Xe (Woodbury), Xi = Z^-1 L."""
        D = self.lumped_impedance(freq)
        M = self.Lm.T @ Xi
        y = self.Lm.T @ Xe
        return Xe - Xi @ (D @ np.linalg.solve(np.eye(len(D)) + M @ D, y))

    def lumped_currents(self, x):
        """Current a -> b through each lumped gap, (n_lumped, Nexc)."""
        return self.Lm.T @ x

    def charges(self, x, omega):
        """Surface charge density per triangle, (Nt, Nexc)."""
        m = self.mesh
        b = m.tri_basis
        lab = np.where(b >= 0, m.edge_len[np.maximum(b, 0)], 0.0)
        cV = m.tri_sign * lab / m.area[:, None]
        div = np.einsum("ti,tie->te", cV, np.where((b >= 0)[..., None], x[np.maximum(b, 0)], 0))
        # a via segment drains its upper triangles and feeds its lower ones
        for k, g in enumerate(m.via_segments):
            t = g["top_tris"]
            if t is not None:
                div[t] += x[m.n_basis + k] / m.area[t].sum()
            t = g["bottom_tris"]
            if t is not None:
                div[t] -= x[m.n_basis + k] / m.area[t].sum()
        return -div / (1j * omega)

    def via_currents(self, x):
        """Current flowing down each via segment (mesh.via_segments), (Nseg, Nexc)."""
        return x[self.mesh.n_basis:self.mesh.n_basis + self.mesh.n_vias]

    def frill_voltages(self, x):
        """Voltage between via barrel and plane at each antipad (mesh.frills)."""
        return x[self.mesh.n_basis + self.mesh.n_vias:]

    def line_voltage(self, greens, q, pts, layer="top"):
        """Voltage to the plane below (ground or an inner plane) at in-plane
        points of a metal layer from the triangle charges q (Nt, Nexc) of
        all layers."""
        m = self.mesh
        li = m.layer_names.index(layer)
        out = 0.0
        for ls in range(len(m.layer_names)):
            on = np.nonzero(m.tri_layer == ls)[0]
            if not on.size:
                continue
            pr = greens.pair(li, ls)
            if getattr(pr, "zero", False):
                continue
            V = m.verts[on]
            qp = quad_points(V, TRI7_BARY)  # (Nt,K,2)
            qw = TRI7_W[None, :] * m.area[on][:, None]
            d = pts[None, :, None, :] - qp[:, None, :, :]
            SW = pr.smooth_w(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2))  # (Nt,M,K)
            K0W = np.einsum("qml,ql->qm", SW, qw)
            if pr.cW:
                I0, _ = potential_integrals(V, pts, pr.dz)  # (Nt,M)
                K0W = K0W + pr.cW * I0 / (4 * np.pi)
            out = out + (K0W.T @ q[on]) / EPS0
        return out

    def current_density(self, x):
        """Surface current density vector at triangle centroids, (Nt, 2, Nexc)."""
        m = self.mesh
        b = m.tri_basis
        lab = np.where(b >= 0, m.edge_len[np.maximum(b, 0)], 0.0)
        cA = m.tri_sign * lab / (2 * m.area[:, None])
        rv = m.centroid[:, None, :] - m.verts  # (Nt,3,2)
        xb = np.where((b >= 0)[..., None], x[np.maximum(b, 0)], 0)  # (Nt,3,Nexc)
        return np.einsum("ti,tid,tie->tde", cA, rv, xb)

    def memory_estimate(self, interp=None):
        """Peak memory [bytes] of a dense sweep: Z, its LU and a fill
        workspace (3 N x N complex128), plus the FreqInterp anchors kept at
        once (order + 1 of them, complex128 or complex64). None for the
        H-matrix method (its memory depends on the compression)."""
        if self.method == "hmatrix":
            return None
        n = int(self.mesh.n_unknowns)
        need = 3 * 16 * n * n
        if interp is not None:
            item = np.dtype(interp.dtype or np.complex128).itemsize
            need += (interp.order + 1) * item * n * n
        return need

    def _wait_memory(self, need, wait):
        """Before the first fill: compare the estimate with MemAvailable. wait
        <= 0: warn only; wait > 0: poll every 30 s for up to `wait` seconds,
        then raise MemoryError (other jobs on a shared machine can grow after
        the solver chose its method)."""
        from .freqinterp import _available_memory

        if need is None:
            return
        avail = _available_memory()
        if need <= avail:
            return
        msg = f"estimated {need / 1e9:.1f} GB > available {avail / 1e9:.1f} GB (N = {self.mesh.n_unknowns})"
        if not wait or wait <= 0:
            if self.verbose:
                print(f"  memory: {msg}; continuing (sweep(wait_memory=s) waits instead)", flush=True)
            return
        if self.verbose:
            print(f"  memory: {msg}; waiting up to {wait / 60:.0f} min", flush=True)
        t0 = time.time()
        while time.time() - t0 < wait:
            time.sleep(min(30.0, max(1.0, wait / 20)))
            avail = _available_memory()
            if need <= avail:
                if self.verbose:
                    print(f"  memory: {avail / 1e9:.1f} GB available after {(time.time() - t0) / 60:.1f} min",
                          flush=True)
                return
        raise MemoryError(f"{msg} after waiting {wait / 60:.0f} min")

    def _checkpoint_key(self, freqs, values):
        m = self.mesh
        return np.array([len(freqs), float(np.sum(freqs)), float(m.n_unknowns), float(np.sum(m.nodes)),
                         float(np.sum(m.nodes ** 2)), float(len(m.tris)),
                         float(sum(hash(str(v)) % 1000003 for v in sorted((values or {}).items())))])

    def sweep(self, freqs, keep_currents=False, interp="auto", values=None, checkpoint=None, wait_memory=None):
        """S-parameters at freqs.

        interp : "auto" (default) -> interpolate Z between anchor fills
                 (FreqInterp with its defaults) when the sweep has at least
                 twice as many frequencies as anchors, else fill every point;
                 a FreqInterp -> use it; None -> exact fill at every point.
        values : {lumped name: value} overriding the layout's components.
        checkpoint : path of an .npz written after every frequency; a sweep
                 restarted with the same path, frequencies, mesh and values
                 resumes from it (skips the frequencies already done; with
                 keep_currents every frequency is solved again).
        wait_memory : seconds to wait for enough MemAvailable before the
                 first fill (memory_estimate), then MemoryError; None takes
                 $MOM25D_WAIT_MEMORY (default 0: warn only)."""
        import os

        freqs = np.atleast_1d(np.asarray(freqs, float))
        self.values = dict(values or {})
        if self.method == "hmatrix" or getattr(self.mesh, "n_ap", 0):
            interp = None  # FreqInterp stores dense matrices (and has no aperture rows)
        if isinstance(interp, str):
            if interp != "auto":
                raise ValueError("interp must be 'auto', None or a FreqInterp")
            from .freqinterp import FreqInterp

            interp = FreqInterp().setup(self, freqs)
            if len(freqs) < 2 * len(interp.anchors):
                interp = None
        elif interp is not None:
            interp.setup(self, freqs)
        if interp is not None:
            if self.verbose:
                print(f"  FreqInterp: {len(interp.anchors)} anchor fills for {len(freqs)} frequencies "
                      f"(order {interp.order}, k_e = {interp.k_ratio:.3f} k0)")
        if wait_memory is None:
            wait_memory = float(os.environ.get("MOM25D_WAIT_MEMORY", 0) or 0)
        nf, npt, ni = len(freqs), len(self.ports), len(self.iports)
        n = npt + ni  # feed ports, then internal ports
        S = np.zeros((nf, n, n), complex)
        gam = np.zeros((nf, n), complex)
        zl = np.zeros((nf, n), complex)
        zr = np.zeros((nf, n))
        res = np.zeros((nf, n))
        cur = []
        waves = np.zeros((nf, n, n), complex)
        zi = np.array([ip["z_ref"] for ip in self.iports], float)
        layers = list(dict.fromkeys(p.layer for p in self.ports))
        done_mask = np.zeros(nf, bool)
        key = self._checkpoint_key(freqs, self.values) if checkpoint else None
        if checkpoint and os.path.exists(checkpoint) and not keep_currents:
            try:
                ck = np.load(checkpoint)
                if ck["key"].shape == key.shape and np.allclose(ck["key"], key, rtol=1e-12, atol=0) \
                        and ck["S"].shape == S.shape:
                    done_mask = ck["done"].astype(bool)
                    S[:], gam[:], zl[:], zr[:], res[:], waves[:] = (ck[k] for k in ("S", "gam", "zl", "zr", "res", "waves"))
                    if self.verbose:
                        print(f"  checkpoint: resuming, {int(done_mask.sum())} of {nf} frequencies already done",
                              flush=True)
                elif self.verbose:
                    print("  checkpoint: file does not match this sweep; starting afresh", flush=True)
            except Exception as exc:  # unreadable (e.g. killed while writing): start afresh
                if self.verbose:
                    print(f"  checkpoint: unreadable ({exc}); starting afresh", flush=True)
                done_mask[:] = False
        if not done_mask.all():
            self._wait_memory(self.memory_estimate(interp), wait_memory)
        t_start = time.time()
        n_new = 0
        for fi, f in enumerate(freqs):
            if done_mask[fi]:
                continue
            if self.verbose and n_new:
                done = time.time() - t_start
                eta = done / n_new * (nf - fi)
                if interp is not None and interp.n_fills:
                    # anchor fills come in bursts: count them separately
                    per_fill = interp.fill_time / interp.n_fills
                    per_point = (done - interp.fill_time) / n_new
                    eta = per_point * (nf - fi) + per_fill * (len(interp.anchors) - interp.n_fills)
                print(f"  [{fi}/{nf}] elapsed {done / 60:5.1f} min, ETC {eta / 60:5.1f} min "
                      f"(done at {time.strftime('%H:%M', time.localtime(time.time() + eta))})",
                      flush=True)
            g = self.greens(f)
            x = self.solve_currents(f, greens=g, interp=interp)
            if keep_currents:
                cur.append(x)
            q = self.charges(x, g.omega)
            phi = [None] * npt
            for lay in layers:
                ps = [p for p in range(npt) if self.ports[p].layer == lay]
                vp = self.line_voltage(g, q, np.concatenate([self.pm[p].vpts for p in ps]), lay)
                for p, v in zip(ps, np.split(vp, np.cumsum([len(self.pm[p].vpts) for p in ps])[:-1])):
                    phi[p] = v
            Acoef = np.zeros((n, n), complex)
            Bcoef = np.zeros((n, n), complex)
            for members in self.groups.values():
                port = self.ports[members[0]]
                hp = self.stack.level_h(port.layer)
                b0 = analytic.beta(port.width, hp, self.stack.eps_below(port.layer), f)
                # dielectric above the line (buried / stripline) raises eps_eff
                # beyond the microstrip estimate: search a wider range
                bnd = (0.7, 1.3) if not self.stack.covered(port.layer) else (0.7, 1.8)
                s_list, I_list, V_list, keys = [], [], [], []
                for p in members:
                    Ip = self.pm[p].currents(x)  # (Ns, Nexc)
                    Vp = self.pm[p].voltages(phi[p])
                    for e in range(n):
                        s_list.append(self.pm[p].s)
                        I_list.append(Ip[:, e])
                        V_list.append(Vp[:, e])
                        keys.append((p, e))
                gm, z0, coefs, r = fit_waves(s_list, I_list, V_list, 1e-3 * b0 + 1j * b0,
                                             lossless=self.lossless, bounds=bnd, v_weight=self.v_weight)
                for (p, e), (a, bb) in zip(keys, coefs):
                    Acoef[p, e], Bcoef[p, e] = a, bb
                for p in members:
                    gam[fi, p], zl[fi, p], res[fi, p] = gm, z0, r
            if ni:  # internal ports: V = Vs - z_ref I, power waves a = Vs / 2 sqrt(z), b = (V - z I) / 2 sqrt(z)
                Ii = self.Lm[:, self.iport_index].T @ x  # (ni, n) current through each source
                Vs = np.zeros((ni, n))
                Vs[:, npt:] = np.eye(ni)
                Acoef[npt:] = Vs / (2 * zi[:, None])  # current-normalized: a / sqrt(z)
                Bcoef[npt:] = -(Vs - 2 * zi[:, None] * Ii) / (2 * zi[:, None])  # -b / sqrt(z)
                zl[fi, npt:] = zi
            waves[fi] = Acoef
            SI = -Bcoef @ np.linalg.inv(Acoef)  # S for current-normalized waves
            if npt and self.z0_fixed is None:
                zr[fi, :npt] = self._reciprocal_z0(SI[:npt, :npt], zl[fi, :npt].real)
                if ni:
                    zr[fi, :npt] = self._anchor_z0(SI, zr[fi, :npt], zi)
            elif npt:
                zr[fi, :npt] = self.z0_fixed
            zr[fi, npt:] = zi
            d = np.sqrt(zr[fi])
            S[fi] = d[:, None] * SI / d[None, :]
            done_mask[fi] = True
            n_new += 1
            if self.verbose:  # machine-readable progress (simulation manager ETC)
                print(f"PROGRESS {int(done_mask.sum())}/{nf}", flush=True)
            if checkpoint:  # atomic: a kill while writing leaves the previous file
                tmp = checkpoint + ".tmp.npz"
                np.savez(tmp, key=key, done=done_mask, S=S, gam=gam, zl=zl, zr=zr, res=res, waves=waves)
                os.replace(tmp, checkpoint)
        return SweepResult(freqs, S, zr, gam, zl, res, [p.name for p in self.ports] + [ip["name"] for ip in self.iports],
                           cur if keep_currents else None, waves)

    def _anchor_z0(self, SI, zfeed, zi, prior=1e-4):
        """Feed reference impedances refined by reciprocity with the internal
        ports, whose reference z_ref is exact: per feed group, the
        coupling-weighted mean of log(z_i |SI_ij / SI_ji|) over internal
        ports i and members j, with the feed-only estimate as a weak prior."""
        npt = len(zfeed)
        z = np.array(zfeed, float)
        cmin = getattr(self, "z0_recip_min", None)
        for members in self.groups.values():
            num, den = prior * np.log(np.mean(z[members])), prior
            for j in members:
                for k, zk in enumerate(zi):
                    i = npt + k
                    w = min(abs(SI[i, j]), abs(SI[j, i])) ** 2
                    if w > 0 and (cmin is None or w >= cmin**2):
                        num += w * np.log(zk * abs(SI[i, j] / SI[j, i]))
                        den += w
            z[members] = np.exp(num / den)
        return z

    def _reciprocal_z0(self, SI, zphi, prior=1e-4):
        """Port reference impedances that make S reciprocal.

        With current-normalized waves S = D SI D^-1, D = diag(sqrt(Z0)), so
        reciprocity fixes the ratios Z0_i / Z0_j = SI_ji / SI_ij between port
        groups that couple. The absolute level (and the ratio for groups that
        barely couple) comes from the scalar-potential estimate zphi, which is
        exact in the quasi-static limit.

        Groups of ports with z0_from="reciprocity" (invalid line voltage) are
        left out of that fit; each then gets the coupling-weighted mean of
        Z0_j SI_ji / SI_ij over the "voltage" ports j.
        """
        groups = list(self.groups.values())
        recip = np.array([self.ports[m[0]].z0_from == "reciprocity" for m in groups])
        if recip.all():
            raise ValueError("every port has z0_from='reciprocity': at least one needs 'voltage'")
        cmin = getattr(self, "z0_recip_min", None)
        if not recip.any():
            return self._voltage_z0(SI, zphi, groups, prior, cmin)
        gid = np.empty(len(self.ports), int)
        for k, members in enumerate(groups):
            gid[members] = k
        vport = ~recip[gid]
        z = np.array(zphi, float)
        vidx = np.nonzero(vport)[0]  # voltage ports, in port order
        vgroups = [np.nonzero(np.isin(vidx, g))[0] for g, r in zip(groups, recip) if not r]
        z[vidx] = self._voltage_z0(SI[np.ix_(vidx, vidx)], z[vidx], vgroups, prior, cmin)
        for k in np.nonzero(recip)[0]:
            num = den = 0.0
            for i in groups[k]:
                for j in np.nonzero(vport)[0]:
                    w = min(abs(SI[i, j]), abs(SI[j, i])) ** 2
                    if w > 0 and (cmin is None or w >= cmin**2):
                        num += w * np.log(z[j] * abs(SI[j, i] / SI[i, j]))
                        den += w
            if den == 0:
                raise ValueError(f"port group {k} (z0_from='reciprocity') does not couple to any voltage port")
            z[groups[k]] = np.exp(num / den)
        return z

    @staticmethod
    def _voltage_z0(SI, zphi, groups, prior, cmin=None):
        """Reciprocity + scalar-potential fit of Z0 over the given groups."""
        if len(groups) == 1:
            return zphi
        gid = np.empty(len(zphi), int)
        for k, members in enumerate(groups):
            gid[members] = k
        rows, rhs, wts = [], [], []
        for i in range(len(gid)):
            for j in range(i + 1, len(gid)):
                if gid[i] != gid[j]:
                    w = min(abs(SI[i, j]), abs(SI[j, i])) ** 2
                    if w > 0 and (cmin is None or w >= cmin**2):
                        r = np.zeros(len(groups))
                        r[gid[i]], r[gid[j]] = 1, -1
                        rows.append(r)
                        rhs.append(np.log(abs(SI[j, i] / SI[i, j])))
                        wts.append(w)
        for k, members in enumerate(groups):
            r = np.zeros(len(groups))
            r[k] = 1
            rows.append(r)
            rhs.append(np.log(np.mean(zphi[members])))
            wts.append(prior)
        sw = np.sqrt(wts)
        logz, *_ = np.linalg.lstsq(np.array(rows) * sw[:, None], np.array(rhs) * sw, rcond=None)
        # keep the absolute level of the scalar-potential estimate
        logz += np.mean(np.log(zphi)) - np.mean(logz[gid])
        return np.exp(logz[gid])
