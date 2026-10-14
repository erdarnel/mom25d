"""Apertures (slots, keep-outs of any shape) in solid inner planes.

A plane declared in the Stackup (`planes`) is infinite and splits the stack
into sections that do not couple (layered.SectionedGreens). An aperture
(`Layout.add_aperture`) is a region of the plane without copper: it is closed
with PEC and replaced by an equivalent magnetic surface current on each side,

    M = -n x E_a  (n: plane normal into the section),  +M above, -M below,

so the tangential E of the aperture is continuous by construction. M is
expanded in RWG functions on a triangle mesh of the aperture (boundary edges
excluded: M . n = 0 on the rim <-> E along the plate edge = 0). Testing the
continuity of tangential H across the aperture with the same functions,

    <f, H_up(J_up) + H_up(M)> = <f, H_dn(J_dn) + H_dn(-M)>,

gives with Z = -<f, E(J)> (solver.assemble) the symmetric system

    [ Z    B ] [I]   [e]          B = [D_up^T ; -D_dn^T],
    [ B^T  Y ] [m] = [0],         D_side[m, n] = <f_m, H_side(f_n)>,
                                  Y = <f, H_up(f)> + <f, H_dn(f)>,

using reciprocity <f_n, E(f_m^M)> = -<f_m^M, H(f_n)> inside each section.

Spectral kernels (Michalski-Zheng TL form; u, v = unit vectors along and
across k_rho; I^e, I^h, y^e, y^h normalised as in layered.py, where the TL
currents equal the physical ones):

- H at the plane from a horizontal unit current at level i:
      H~ = (u v I^h - v u I^e) . J~,   I = TL current at the plane (+z)
  -> spatial  G = h_a(rho) [[0, 1], [-1, 0]] - h_s(rho) Q(phi),
      h_a = S0{(I^h + I^e) / 2},  h_s = S2{(I^h - I^e) / 2},
      Q = [[-sin 2phi, cos 2phi], [cos 2phi, sin 2phi]],
  S_n{f} = (1/2pi) Int f J_n(k rho) k dk. The static part c e^{-k dz} is
  removed in closed form (S0: dz / R^3, S2: (R-dz)^2 (2R+dz) / (rho^2 R^3)).
  Checked: a current over a PEC half-space gives twice its Biot-Savart field.
- H at the plane from M on it (input admittances y looking into the section):
      H~ = -(Y^h u u + Y^e v v) . M~
  -> <f_m, H(f_n)> = -jw eps0 <f_m, g_e f_n> - 1/(jw mu0) <div f_m, g_q div f_n>,
      g_e~ = y^e,  g_q~ = (y^h + k0^2 y^e) / k^2    (= 1/u for a half-space),
  the dual of formulation C: assembled by solver.assemble with gA = g_e,
  gV = g_q, times -eps0 / mu0.
- Vertical currents (via segments, thick-metal side walls) in a section:
  quasi-static Biot-Savart with images in the section's planes (the via
  convention of solver.via_impedance).

Thick plane (two planes lo / hi at the faces of a foil, same aperture in
both): NOT supported yet. The first attempt shared one M between the faces
(thin slot with PEC walls: Ampere across it, dH_t/dz = jw eps M, adds
-jw eps t <f_m, f_n> to Y); that drops the E_z drop across the foil inside the
opening, i.e. the opening's interior collapses to zero height: with an 18 um
L2 the line Z0 came out 4 ohm low (2-D FD: thickness changes it by <= 0.3
ohm). A correct model needs one M per face, the interior section (closed PPW
of the foil thickness) coupling them, and vertical wall currents along the
opening's rim (E_z = 0 on the plate edge). Meanwhile model the foil as one
plane at its top face.

Limitation: the line voltage used for port Z0 (Solver.line_voltage) is the
voltage to the plane below and ignores the aperture field: under a feed that
runs over an aperture it is only proportional to the true voltage. gamma is
still right; Z0 must come from reciprocity with an internal port (gap / via
port; done automatically) or be given (Solver(z0=...)).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.special import hankel2, j0, jv

from .constants import C0, EPS0, MU0
from .greens import _csqrt_re_pos, _gauss_panels
from .integrals import TRI3_BARY, TRI3_W, TRI7_BARY, TRI7_W, quad_points
from .layered import _Pair

N_IMAGES = 60  # vertical-current images per side in a closed section


# ------------------------------------------------------------------ spectral
def _lines(sec, kr, k0):
    """TL data of a section Stackup (ground at interface 0, air or PEC above
    interface N) at spectral points kr: u (N,K) and per polarisation
    ("h": TE, "e": TM) the layer admittances y, the admittances looking down
    (ydn[k], k = 1..N) and up (yup[k], k = 0..N-1; yup[N] = air) at each
    interface."""
    eps = np.array([d.eps_r * (1 - 1j * d.tand) for d in sec.layers])
    t = np.array([d.t for d in sec.layers])
    N = len(t)
    u = np.stack([_csqrt_re_pos(kr**2 - k0**2 * e) for e in eps])
    x = u * t[:, None]
    ex = np.exp(-2 * x)
    T = (1 - ex) / (1 + ex)
    pec_top = sec.top == "pec"
    out = {}
    u0 = np.sqrt(kr**2 - k0**2)
    for pol, y, yair in (("h", u, u0), ("e", eps[:, None] / u, 1.0 / u0)):
        ydn = np.empty((N + 1, kr.size), complex)
        ydn[0] = np.inf
        ydn[1] = y[0] / T[0]
        for k in range(2, N + 1):
            yl, yk, tk = ydn[k - 1], y[k - 1], T[k - 1]
            ydn[k] = yk * (yl + yk * tk) / (yk + yl * tk)
        yup = np.empty((N + 1, kr.size), complex)
        yup[N] = np.inf if pec_top else yair
        for k in range(N - 1, -1, -1):
            yk, tk = y[k], T[k]  # layer k+1
            if k == N - 1 and pec_top:
                yup[k] = yk / tk
            else:
                yl = yup[k + 1]
                yup[k] = yk * (yl + yk * tk) / (yk + yl * tk)
        out[pol] = (y, ydn, yup)
    return u, x, T, out, eps, t


def _voltages(sec, kr, k0, src):
    """TL voltages at every interface for a unit shunt current at interface
    src (both polarisations): {pol: V (N+1, K)} plus the _lines data."""
    u, x, T, L, eps, t = _lines(sec, kr, k0)
    N = len(t)
    S = 2 * np.exp(-x) / (1 + np.exp(-2 * x))
    pec_top = sec.top == "pec"
    V = {}
    for pol, (y, ydn, yup) in L.items():
        v = np.zeros((N + 1, kr.size), complex)
        v[src] = 1.0 / (ydn[src] + yup[src])
        for k in range(src, 1, -1):
            v[k - 1] = v[k] * S[k - 1] / (1 + ydn[k - 1] / y[k - 1] * T[k - 1])
        for k in range(src, N):
            if k + 1 == N and pec_top:
                break
            v[k + 1] = v[k] * S[k] / (1 + yup[k + 1] / y[k] * T[k])
        V[pol] = v
    return V, (u, x, L, eps, t)


def _plane_currents(sec, kr, k0, src, side):
    """TL currents (+z) at the aperture plane for a unit shunt source at
    interface src: (I^h, I^e). side "up": the plane is the section's ground
    (z = 0); "dn": its PEC cover (z = top)."""
    V, (u, x, L, eps, t) = _voltages(sec, kr, k0, src)
    N = len(t)
    out = []
    for pol in ("h", "e"):
        y = L[pol][0]
        if side == "up":  # V(z) = V1 sinh(u z) / sinh(u t1): I(0) = -y1 V1 / sinh(u1 t1)
            k, sgn = 0, -1.0
            vk = V[pol][1]
        else:  # V(z) = V_{N-1} sinh(u (z_N - z)) / sinh(u tN): I(z_N) = yN V_{N-1} / sinh
            k, sgn = N - 1, 1.0
            vk = V[pol][N - 1]
        csch = 2 * np.exp(-x[k]) / (1 - np.exp(-2 * x[k]))
        out.append(sgn * y[k] * vk * csch)
    return out


def _input_admittance(sec, kr, k0, side):
    """(y^h, y^e) looking into the section from the aperture plane."""
    _, _, _, L, _, _ = _lines(sec, kr, k0)
    N = len(sec.layers)
    return tuple(L[p][2][0] if side == "up" else L[p][1][N] for p in ("h", "e"))


def _path(k0, kmax_r, tmin, rho_max, order=12):
    """Sommerfeld integration path as LayeredGreens._build_path: a lifted
    sine contour over the branch points / poles, then the real axis."""
    T = 1.3 * kmax_r + k0
    a = min(0.5 * k0, 1.0 / rho_max)
    n1 = int(np.clip(np.ceil(2 * T / a), 100, 3000))
    t, w = _gauss_panels(0.0, T, n1, order)
    kr1 = t + 1j * a * np.sin(np.pi * t / T)
    dk1 = w * (1 + 1j * a * np.pi / T * np.cos(np.pi * t / T))
    kmax = max(40 * kmax_r, 20.0 / tmin, T * 1.5)
    dk = min(np.pi / rho_max, 0.5 / tmin, kmax_r)
    n2 = int(np.ceil((kmax - T) / dk))
    t2, w2 = _gauss_panels(T, kmax, n2, 8)
    return (kr1, dk1, False), (t2.astype(complex), w2.astype(complex), True)


def _hankel(rho, f, path, n):
    """(1/2pi) Int f(k) J_n(k rho) k dk over the path; f: callable kr -> (M, K)."""
    out = 0.0
    for kr, w, real in path:
        F = f(kr) * (w * kr / (2 * np.pi))[None]  # (M, K)
        res = np.empty((rho.size, F.shape[0]), complex)
        chunk = max(1, int(4e6 // kr.size))
        for s in range(0, rho.size, chunk):
            arg = np.outer(rho[s:s + chunk], kr.real if real else kr)
            if n == 0:
                J = j0(arg.real) if real else jv(0, arg)
            else:
                J = jv(n, arg.real) if real else jv(n, arg)
            res[s:s + chunk] = J @ F.T
        out = out + res
    return out.T  # (M, rho)


def _rho_table(rho_max, tmin, lam):
    r = [0.0]
    while r[-1] < rho_max:
        r.append(r[-1] + min(lam / 80, max(tmin / 25, r[-1] / 25)))
    return np.array(r)


class _CouplingKernel:
    """h_a(rho), h_s(rho) of one (metal level, aperture side) pair."""

    def __init__(self, dz, rho, ha, hs):
        self.dz = dz
        self._spl = CubicSpline(rho, np.stack([ha, hs], -1))
        self.rho_max = rho[-1]

    def __call__(self, rho):
        S = self._spl(rho)
        return S[..., 0], S[..., 1]


class ApertureKernels:
    """Kernels of one aperture at one frequency.

    stack  : the full Stackup; lo, hi: plane names (equal for a thin plane)
    levels : {"up": [metal level names], "dn": [...]} in the adjacent sections
    """

    def __init__(self, freq, stack, lo, hi, levels, rho_max):
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.k0 = k0 = self.omega / C0
        self.stack = stack
        self.sides = {}
        secs = {}
        for side, plane in (("up", hi), ("dn", lo)):
            b = stack.bounds.index(stack.planes[plane])
            s = b if side == "up" else b - 1
            secs[side] = (s, stack.section_stack(s))
        eps_all = [d.eps_r for _, sec in secs.values() for d in sec.layers]
        t_all = [d.t for _, sec in secs.values() for d in sec.layers]
        kmax_r = k0 * np.sqrt(max(eps_all))
        tmin = min(t_all)
        path = _path(k0, kmax_r, tmin, rho_max)
        lam = 2 * np.pi / kmax_r
        rho = _rho_table(rho_max, tmin, lam)
        self.rho = rho
        S_e = S_q = 0.0
        c_e = c_q = 0.0
        for side, (s, sec) in secs.items():
            N = len(sec.layers)
            adj = sec.layers[0] if side == "up" else sec.layers[N - 1]
            ea = adj.eps_r * (1 - 1j * adj.tand)
            ka = k0 * np.sqrt(ea)
            ce, cq = 2 * ea, 2.0

            def self_rem(kr, sec=sec, side=side, ce=ce, cq=cq, ka=ka):
                yh, ye = _input_admittance(sec, kr, k0, side)
                ux = np.sqrt(kr**2 - ka**2)
                ge = ye - ce / (2 * ux)
                gq = (yh + k0**2 * ye) / kr**2 - cq / (2 * ux)
                return np.stack([ge, gq])

            res = _hankel(rho, self_rem, path, 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(rho > 0, np.expm1(-1j * ka * rho) / (4 * np.pi * rho), -1j * ka / (4 * np.pi))
            S_e = S_e + ce * f + res[0]
            S_q = S_q + cq * f + res[1]
            c_e, c_q = c_e + ce, c_q + cq
            # coupling kernels for the metal levels of this section
            kern = {}
            zp = 0.0 if side == "up" else float(sec.z[-1])
            for name in levels.get(side, []):
                src = sec.metals[name]
                dz = abs(float(sec.z[src]) - zp)
                Kb = 15.0 / tmin
                Ih, Ie = _plane_currents(sec, np.array([Kb + 0j]), k0, src, side)
                ca = complex(-0.5 * (Ih[0] + Ie[0]) * np.exp(Kb * dz))
                cs = complex(-0.5 * (Ih[0] - Ie[0]) * np.exp(Kb * dz))

                def rem(kr, sec=sec, src=src, side=side, dz=dz, ca=ca, cs=cs):
                    Ih, Ie = _plane_currents(sec, kr, k0, src, side)
                    e = np.exp(-kr * dz)
                    return np.stack([0.5 * (Ih + Ie) + ca * e, 0.5 * (Ih - Ie) + cs * e])

                ra = _hankel(rho, lambda kr: rem(kr)[:1], path, 0)[0]
                rs = _hankel(rho, lambda kr: rem(kr)[1:], path, 2)[0]
                R = np.sqrt(rho**2 + dz**2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s2 = np.where(rho > 0, (R - dz) ** 2 * (2 * R + dz) / (rho**2 * R**3), 0.0)
                ha = ra - ca * dz / (2 * np.pi * R**3)
                hs = rs - cs * s2 / (2 * np.pi)
                kern[name] = _CouplingKernel(dz, rho, ha, hs)
            self.sides[side] = dict(section=s, stack=sec, kern=kern, zp=zp)
        self.self_pair = _Pair(0.0, c_e, c_q, 0.0, CubicSpline(rho, np.stack([S_e, S_q], -1)),
                               CubicSpline(rho, np.zeros_like(rho, dtype=complex)))
        # thick plane: slot filled with the dielectric between the two planes
        self.t_slot, self.eps_slot = 0.0, 1.0
        if lo != hi:
            k_lo, k_hi = stack.planes[lo], stack.planes[hi]
            lay = stack.layers[k_lo:k_hi]
            self.t_slot = float(stack.z[k_hi] - stack.z[k_lo])
            self.eps_slot = sum(d.t * d.eps_r * (1 - 1j * d.tand) for d in lay) / self.t_slot


class _SelfGreens:
    """Minimal greens object for solver.assemble on the aperture mesh."""

    def __init__(self, omega, pair):
        self.omega = omega
        self._pair = pair
        self.h = 1.0

    def pair(self, i, j):
        return self._pair


# ------------------------------------------------------------- quadrature
def _subdivide(V, level):
    """Split triangles (N,3,2) into 4^level congruent pieces: (N*4^level,3,2)."""
    for _ in range(level):
        a, b, c = V[:, 0], V[:, 1], V[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        V = np.stack([np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))],
                     1).reshape(-1, 3, 2)
    return V


def _fine_rule(V, area, size, dz, max_level=2, frac=1.0):
    """Per triangle: quadrature points and weights fine enough for a kernel
    of width dz (sub-triangles <= dz, 7 points each). Returns lists."""
    lev = np.clip(np.ceil(np.log2(np.maximum(size / (frac * dz), 1.0))), 0, max_level).astype(int)
    pts, wts = [None] * len(V), [None] * len(V)
    for L in np.unique(lev):
        sel = np.nonzero(lev == L)[0]
        sub = _subdivide(V[sel], L).reshape(sel.size, 4**L, 3, 2)
        P = quad_points(sub.reshape(-1, 3, 2), TRI7_BARY).reshape(sel.size, -1, 2)
        W = np.broadcast_to(TRI7_W[None, :] * (area[sel] / 4**L)[:, None, None],
                            (sel.size, 4**L, 7)).reshape(sel.size, -1)
        for k, t in enumerate(sel):
            pts[t], wts[t] = P[k], W[k]
    return pts, wts


def _tri_data(mesh):
    V, A = mesh.verts, mesh.area
    b = mesh.tri_basis
    lab = np.where(b >= 0, mesh.edge_len[np.maximum(b, 0)], 0.0)
    cA = mesh.tri_sign * lab / (2 * A[:, None])
    size = np.max(np.linalg.norm(V - np.roll(V, 1, axis=1), axis=-1), axis=1)
    return V, A, b, cA, size


def _G_contract(ka, ks, X, Y, d):
    """sum_k sum_l X[i,k,:] . G(d_kl) . Y[l,j,:] for G = ka Rot - ks Q(phi).
    ka, ks (K, L) kernel values, d (K, L, 2) separations r - r',
    X (3, K, 2) weighted test halves, Y (L, 3, 2)... returns (3, L, 3) per source point."""
    dx, dy = d[..., 0], d[..., 1]
    r2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        c2 = np.where(r2 > 0, (dx * dx - dy * dy) / r2, 0.0)
        s2 = np.where(r2 > 0, 2 * dx * dy / r2, 0.0)
    # G = [[ks s2, ka - ks c2], [-ka - ks c2, -ks s2]]
    Gxx, Gxy = ks * s2, ka - ks * c2
    Gyx, Gyy = -ka - ks * c2, -ks * s2
    # a^T G b with a = X (test), b = Y (source): (3,K) x (K,L) -> (3,L) per component pair
    ax, ay = X[..., 0], X[..., 1]
    Px = ax @ Gxx + ay @ Gyx  # coefficient of b_x, (3, L)
    Py = ax @ Gxy + ay @ Gyy  # coefficient of b_y
    return Px, Py


class ApertureModel:
    """Aperture unknowns of a solver: the B (N_main x n_ap) and Y (n_ap x
    n_ap) blocks at each frequency (see the module docstring)."""

    NEAR_DZ = 5.0  # pairs closer than NEAR_DZ * dz (+ 1.5 sizes) use the fine rules
    FINE_LEVEL = 2  # at most 4^FINE_LEVEL sub-triangles per triangle in the fine rules
    CHUNK = 16  # test triangles per near-field chunk
    ALLOW_SHARED_THICK = False  # experimental (lo, hi) shared-M thick model, known to be wrong
    VERTICAL = True  # couple via segments / side walls to the aperture
    TEM_DYNAMIC = True  # full-span vias in closed sections: H1(2) TEM field (False: static 1/(2 pi rho))
    CUT_DZ = 25.0  # metal sources further than CUT_DZ * dz from the aperture are left out
    DROP = 1e-7  # coupling entries below DROP x the largest are dropped
    FINE_FRAC = 1.0  # fine-rule sub-triangles at most FINE_FRAC * dz across

    def __init__(self, solver):
        self.solver = solver
        m = solver.mesh
        st = solver.stack
        self.groups = []
        for g in m.ap_groups:
            lo, hi = g["planes"]
            for p in (lo, hi):
                if p not in st.planes:
                    raise ValueError(f"aperture plane {p!r} is not a plane of the stackup")
            if lo != hi and not self.ALLOW_SHARED_THICK:
                raise NotImplementedError(
                    f"thick aperture ({lo!r}, {hi!r}): one M shared by both faces collapses the slot "
                    "interior (E_z across the foil is lost: 18 um L2 gave Z0 4 ohm low vs 2-D FD). Needs "
                    "separate M per face + the interior section with wall currents; use one plane at the "
                    "foil's top face meanwhile (2-D FD: <= 0.3 ohm from the thickness)")
            if st.planes[hi] < st.planes[lo]:
                raise ValueError(f"aperture planes {lo!r}, {hi!r}: give (lower, upper)")
            between = [n for n, k in list(st.planes.items()) + list(st.metals.items())
                       if st.planes[lo] < k < st.planes[hi]]
            if between:
                raise ValueError(f"aperture planes {lo!r}, {hi!r}: levels {between} lie between them")
            b_hi = st.bounds.index(st.planes[hi])
            b_lo = st.bounds.index(st.planes[lo])
            sec = {"up": b_hi, "dn": b_lo - 1}
            levels = {s: [n for n in m.layer_names if st.section(n) == k] for s, k in sec.items()}
            am = g["mesh"]
            from .solver import AssemblyGeometry

            geom = AssemblyGeometry(am)
            self.groups.append(dict(lo=lo, hi=hi, mesh=am, off=g["off"], levels=levels, sec=sec, geom=geom,
                                    tri=_tri_data(am)))
        self.n = m.n_ap
        self.main = _tri_data(m)
        self._vert = self._vertical_sources()
        self._pairs = {}

    # -------------------------------------------------- vertical currents
    def _vertical_sources(self):
        """Vertical currents of the main mesh: list of (unknown index, xy (P,2)
        sub-filament points, weights (P,), z0, z1, direction +1 up / -1 down)."""
        sol, m, st = self.solver, self.solver.mesh, self.solver.stack
        out = []
        z = {n: st.level_z(n) for n in m.layer_names + list(st.planes)}
        z["gnd"] = 0.0
        for k, g in enumerate(getattr(m, "via_segments", [])):
            v = m.vias[g["via"]]
            out.append((m.n_basis + k, np.array([v.xy]), np.ones(1), z[g["bottom"]], z[g["top"]], -1.0))
        if getattr(m, "n_walls", 0):
            w = np.nonzero(m.wall)[0]
            zl = np.array([st.level_z(n) for n in m.layer_names])
            za, zb = zl[m.tri_layer[m.tp[w]]], zl[m.tri_layer[m.tm[w]]]
            p0, p1 = m.nodes[m.edge_nodes[w, 0]], m.nodes[m.edge_nodes[w, 1]]
            fr = (np.arange(4) + 0.5) / 4
            for i, e in enumerate(w):
                pts = p0[i] + fr[:, None] * (p1[i] - p0[i])
                out.append((e, pts, np.full(4, m.edge_len[e] / 4), min(za[i], zb[i]), max(za[i], zb[i]),
                            1.0 if zb[i] > za[i] else -1.0))
        return out

    @staticmethod
    def _hphi(rho, za, zb, z0, zlo, zhi):
        """H_phi per unit upward current of a vertical filament [za, zb] at
        horizontal distance rho, observed at height z0, with images in a
        PEC plane at zlo and (closed section) zhi."""
        def seg(a, b):
            return ((b - z0) / np.sqrt(rho**2 + (b - z0) ** 2) - (a - z0) / np.sqrt(rho**2 + (a - z0) ** 2))

        if zhi is None:
            tot = seg(za, zb) + seg(2 * zlo - zb, 2 * zlo - za)
        else:
            P = 2 * (zhi - zlo)
            tot = 0.0
            for n in range(-N_IMAGES, N_IMAGES + 1):
                tot = tot + seg(za + n * P, zb + n * P) + seg(2 * zlo - zb + n * P, 2 * zlo - za + n * P)
        return tot / (4 * np.pi * rho)

    # -------------------------------------------------------------- blocks
    def blocks(self, freq):
        """B (n_main x n_ap, scipy sparse), Y (n_ap x n_ap) at freq; n_main =
        all unknowns before the aperture ones."""
        from .solver import assemble

        sol, m = self.solver, self.solver.mesh
        import scipy.sparse as sp

        n_main = m.n_unknowns - self.n
        Bt = []  # per group: (na x n_main) sparse rows
        Y = np.zeros((self.n, self.n), complex)
        omega = 2 * np.pi * freq
        for g in self.groups:
            am = g["mesh"]
            na = am.n_basis
            cols = slice(g["off"] - n_main, g["off"] - n_main + na)
            K = ApertureKernels(freq, sol.stack, g["lo"], g["hi"], g["levels"], sol.rho_max)
            Zs = assemble(am, _SelfGreens(omega, K.self_pair), geom=g["geom"])
            Yg = -(EPS0 / MU0) * Zs
            if K.t_slot:
                ii, jj, gv = g["geom"].gram
                np.add.at(Yg, (ii, jj), -1j * omega * EPS0 * K.eps_slot * K.t_slot * gv)
            Y[cols, cols] += Yg
            Bt.append(self._coupling(g, K, "up") - self._coupling(g, K, "dn"))
        return sp.vstack(Bt).T.tocsr(), Y

    def _coupling(self, g, K, side):
        """D_side = <f_m^M, H_side(unit source n)> as a sparse (na x n_main)
        matrix. Sources further than CUT_DZ * dz from every aperture
        triangle (static coupling ~ (dz / R)^3) and entries below DROP x
        the largest are left out."""
        import scipy.sparse as sp

        sol, m, st = self.solver, self.solver.mesh, self.solver.stack
        am = g["mesh"]
        Va, Aa, ba, cAa, sza = g["tri"]
        Vm, Am, bm, cAm, szm = self.main
        n_main = m.n_unknowns - self.n
        S = K.sides[side]
        okA = ba >= 0
        tree_a = cKDTree(Va.mean(1))
        work = []
        for name, kern in S["kern"].items():
            li = m.layer_names.index(name)
            src = np.nonzero(m.tri_layer == li)[0]
            if src.size:
                dmin, _ = tree_a.query(m.centroid[src])
                src = src[dmin < self.CUT_DZ * kern.dz + 2 * sza.max()]
            if src.size:
                work.append((kern, src))
        # vertical currents (via segments, side walls) inside this section
        s = S["section"]
        zlo = float(st.z[st.bounds[s]])
        zhi = float(st.z[st.bounds[s + 1]]) if st.section_top_pec(s) else None
        z0 = zlo if side == "up" else zhi
        vert = []
        for v in (self._vert if self.VERTICAL else []):
            idx, xy, wv, za, zb, sg = v
            if za < zlo - 1e-12 or (zhi is not None and zb > zhi + 1e-12):
                continue
            if zhi is None:  # open section: decays like a dipole
                dmin, _ = tree_a.query(xy)
                if dmin.min() > self.CUT_DZ * (zb - zlo) + 2 * sza.max():
                    continue
            vert.append(v)
        cols = np.unique(np.concatenate([bm[src][bm[src] >= 0] for _, src in work] +
                                        [np.array([v[0] for v in vert], int)] + [np.zeros(0, int)]))
        colmap = -np.ones(n_main, int)
        colmap[cols] = np.arange(cols.size)
        D = np.zeros((am.n_basis, cols.size), complex)
        for kern, src in work:
            self._level_add(g, kern, src, D, colmap)
        if vert:
            P = quad_points(Va, TRI7_BARY)  # (nta,7,2)
            W = TRI7_W[None, :] * Aa[:, None]
            R = P[:, None, :, :] - Va[:, :, None, :]  # (nta,3,7,2) r_k - a_i
            sec_k = [k for zb_, zt_, k in st.via_sections(K.freq) if abs(zb_ - zlo) < 1e-12]
            k_tem = sec_k[0] if sec_k else None
            for idx, xy, wv, za, zb, sg in vert:
                full = zhi is not None and abs(za - zlo) < 1e-12 and abs(zb - zhi) < 1e-12
                Hv = 0.0
                for p, wp in zip(xy, wv):
                    d = P - p  # (nta,7,2)
                    rho = np.hypot(d[..., 0], d[..., 1])
                    if full:  # parallel-plate TEM: H = k I / (4j) H1(2)(k rho) (-> I / (2 pi rho))
                        h = (k_tem / 4j * hankel2(1, k_tem * rho) if (k_tem is not None and self.TEM_DYNAMIC)
                             else 1 / (2 * np.pi * rho))
                    else:
                        h = self._hphi(rho, za, zb, z0, zlo, zhi)
                    phi = np.stack([-d[..., 1], d[..., 0]], -1) / rho[..., None]  # z x rho_hat
                    Hv = Hv + (sg * wp * W * h)[..., None] * phi
                vals = cAa * np.einsum("tkd,tikd->ti", Hv, R)
                np.add.at(D[:, colmap[idx]], ba[okA], vals[okA])
        D[np.abs(D) < self.DROP * np.abs(D).max(initial=0.0)] = 0.0
        Dc = sp.csr_matrix(D)
        return sp.csr_matrix((Dc.data, cols[Dc.indices], Dc.indptr), shape=(am.n_basis, n_main))

    def _near_values(self, Va, Aa, sza, Vs, As, szs, kern, pt, ps):
        """(npairs, 3, 3) sum_k sum_l w_k w_l (r_k - a_i) . G . (r'_l - b_j)
        over the fine rules of the near pairs (pt[p], ps[p])."""
        from . import _fast

        dz = kern.dz
        ut, inv_t = np.unique(pt, return_inverse=True)
        us, inv_s = np.unique(ps, return_inverse=True)
        Pa, Wa = _fine_rule(Va[ut], Aa[ut], sza[ut], dz, self.FINE_LEVEL, self.FINE_FRAC)
        Ps, Ws = _fine_rule(Vs[us], As[us], szs[us], dz, self.FINE_LEVEL, self.FINE_FRAC)
        toff = np.r_[0, np.cumsum([len(p) for p in Pa])]
        soff = np.r_[0, np.cumsum([len(p) for p in Ps])]
        tp, tw = np.concatenate(Pa), np.concatenate(Wa)
        sp_, sw = np.concatenate(Ps), np.concatenate(Ws)
        out = np.zeros((pt.size, 3, 3), complex)
        if _fast.ENABLED:
            x, c = np.ascontiguousarray(kern._spl.x, float), np.ascontiguousarray(kern._spl.c, complex)
            _fast.aperture_near(inv_t.astype(np.int64), inv_s.astype(np.int64), toff, tp, tw, soff, sp_, sw,
                                np.ascontiguousarray(Va[ut]), np.ascontiguousarray(Vs[us]), x, c, out)
            return out
        for p in range(pt.size):  # plain numpy fallback
            a, b = inv_t[p], inv_s[p]
            pa, wa = tp[toff[a]:toff[a + 1]], tw[toff[a]:toff[a + 1]]
            pb, wb = sp_[soff[b]:soff[b + 1]], sw[soff[b]:soff[b + 1]]
            d = pa[:, None, :] - pb[None, :, :]
            ka, ks = kern(np.hypot(d[..., 0], d[..., 1]))
            X = wa[None, :, None] * (pa[None] - Va[ut[a]][:, None, :])  # (3,K,2)
            Px, Py = _G_contract(ka, ks, X, None, d)  # (3, L)
            Yb = wb[:, None, None] * (pb[:, None, :] - Vs[us[b]][None])  # (L,3,2)
            out[p] = Px @ Yb[:, :, 0] + Py @ Yb[:, :, 1]
        return out

    def _level_add(self, g, kern, src, D, colmap):
        """Add <f_i^M, G f_j^J> of all aperture triangles x the source
        triangles src (one metal level) to D (na x local columns, colmap:
        main basis -> local column)."""
        import scipy.sparse as sp

        Va, Aa, ba, cAa, sza = g["tri"]
        Vm, Am, bm, cAm, szm = self.main
        dz = kern.dz
        nta, ns = len(Va), src.size
        Vs, As, cAs, szs, bs = Vm[src], Am[src], cAm[src], szm[src], bm[src]
        bs = np.where(bs >= 0, colmap[np.maximum(bs, 0)], -1)
        nbm = D.shape[1]
        ok = bs >= 0
        # half-basis (s, j) -> main basis map, (3 ns) -> nbm
        PTs = sp.csr_matrix((np.ones(ok.sum()), (np.nonzero(ok.ravel())[0], bs[ok])), shape=(3 * ns, nbm))
        okA = ba >= 0
        cen_a, cen_s = Va.mean(1), Vs.mean(1)
        tree = cKDTree(cen_s)
        near = [None] * nta
        pt, pss = [], []
        for t, cs in enumerate(tree.query_ball_point(cen_a, self.NEAR_DZ * dz + 1.5 * (sza + szs.max()))):
            cs = np.array(cs, int)
            if cs.size:
                d = np.linalg.norm(cen_s[cs] - cen_a[t], axis=1)
                cs = cs[d < self.NEAR_DZ * dz + 1.5 * (sza[t] + szs[cs])]
            if cs.size:
                near[t] = cs
                pt.append(np.full(cs.size, t))
                pss.append(cs)
        if pt:
            pt, pss = np.concatenate(pt), np.concatenate(pss)
            vals = self._near_values(Va, Aa, sza, Vs, As, szs, kern, pt, pss)  # (np, 3, 3)
            vals *= cAa[pt][:, :, None] * cAs[pss][:, None, :]
            r = np.broadcast_to(ba[pt][:, :, None], vals.shape)
            cc = np.broadcast_to(bs[pss][:, None, :], vals.shape)
            ok2 = (r >= 0) & (cc >= 0)
            np.add.at(D, (r[ok2], cc[ok2]), vals[ok2])
        # far pairs: 3-point rules on both sides (near pairs masked)
        P3a = quad_points(Va, TRI3_BARY)
        W3a = TRI3_W[None, :] * Aa[:, None]
        P3s = quad_points(Vs, TRI3_BARY).reshape(-1, 2)
        W3s = (TRI3_W[None, :] * As[:, None]).ravel()
        Ys = (W3s[:, None, None] * (P3s[:, None, :] - np.repeat(Vs, 3, axis=0)))  # (3ns,3,2)
        chunk = max(1, int(1e6 // (3 * ns)))
        for t0 in range(0, nta, chunk):
            tt = np.arange(t0, min(nta, t0 + chunk))
            nt = tt.size
            pa = P3a[tt].reshape(-1, 2)
            d = pa[:, None, :] - P3s[None, :, :]
            ka, ks = kern(np.hypot(d[..., 0], d[..., 1]))
            mask = np.zeros((nt, ns), bool)
            for k, t in enumerate(tt):
                if near[t] is not None:
                    mask[k, near[t]] = True
            if mask.any():
                mask = np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1)
                ka = np.where(mask, 0.0, ka)
                ks = np.where(mask, 0.0, ks)
            Xa = (W3a[tt][:, None, :, None] * (P3a[tt][:, None, :, :] - Va[tt][:, :, None, :])
                  * cAa[tt][:, :, None, None])  # (nt,3i,3k,2)
            dx, dy = d[..., 0], d[..., 1]
            r2 = dx * dx + dy * dy
            with np.errstate(divide="ignore", invalid="ignore"):
                c2 = np.where(r2 > 0, (dx * dx - dy * dy) / r2, 0.0)
                s2 = np.where(r2 > 0, 2 * dx * dy / r2, 0.0)
            G = [(ks * s2), (ka - ks * c2), (-ka - ks * c2), (-ks * s2)]
            Gxx, Gxy, Gyx, Gyy = (x.reshape(nt, 3, -1) for x in G)
            ax, ay = Xa[..., 0], Xa[..., 1]  # (nt,3i,3k)
            Px = np.einsum("tik,tkl->til", ax, Gxx) + np.einsum("tik,tkl->til", ay, Gyx)
            Py = np.einsum("tik,tkl->til", ax, Gxy) + np.einsum("tik,tkl->til", ay, Gyy)
            # sum over source points with the per-vertex factors: (nt*3, 3ns) x (3ns -> half-bases)
            val = Px[..., None] * Ys[None, None, :, :, 0] + Py[..., None] * Ys[None, None, :, :, 1]
            val = val.reshape(nt, 3, ns, 3, 3).sum(3) * cAs[None, None, :, :]  # (nt,3,ns,3)
            HB = (PTs.T @ val.reshape(nt * 3, ns * 3).T).T  # (nt*3, nbm)
            rows = ba[tt].ravel()
            okr = rows >= 0
            np.add.at(D, rows[okr], HB[okr])
