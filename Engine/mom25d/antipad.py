"""Antipads: vias passing a solid inner plane through a clearance hole.

The hole (drill radius a, antipad radius b, b << wavelength) is closed with
PEC and replaced by equivalent magnetic currents carrying an axisymmetric
aperture field E_r(r) = -dU/dr, a < r < b. The potential profile U(r) is
expanded in `nr` piecewise-linear radial functions w_j (nodes geometrically
spaced from a to b; w_0 = 1 inside the barrel radius a, U(b) = 0 at the
plate edge) with unknown amplitudes V_j -- V_0 is the barrel-to-plane
voltage. M = E x n faces each section (+M above, -M below). The via is split
into two segments at the plane (no charge there). Testing the continuity of
tangential H across the aperture with each w_j gives, by Ampere-Maxwell
around the via axis (circulation of H at the plate = conduction current
through the loop + jw x induced plate charge inside it),

    sum_m C_mj I_m + jw sum_l C_jl V_l = 0,   C_mj = sum over the two sides of
        delta_j0 (I_m along n) + jw Int sigma_m w_j dA

with n the plate normal into the section and sigma_m the charge unknown m
induces on that plate surface. Only w_0 sees the via current (the tents
integrate dw/dr to zero). Charges couple through Phi_j(r) = Int sigma_1 w_j
dA (sigma_1: plate charge of a unit point charge on a metal level, full-wave
TM transfer of the section); vertical via currents in closed sections also
through their TEM / TM_n E_z at the plate. C_jl are the frequency-dependent
capacitances Int Y(k) w~_j w~_l J0(k d) k dk / 2 pi with the section's TM
input admittance Y = eps0 k^2 y_in; in a closed section they include the
radial TEM wave launched into the plate cavity (radiation).

With Z = -<f, E(J)> the current rows read Z I - C V = e; negating the
frill rows keeps the system symmetric: [[Z, -C], [-C^T, -jw C_ff]].

One ring function (the coax-like profile) is only right while b - a is
small next to the distance to other conductors; the default 4 rings let the
aperture field take its own shape. The plane has zero thickness (as in
the openEMS references): the gap capacitance then grows only like
ln(1/gap) as b -> a, so even a 2 um clearance does not short the via to
the plane (checked: |S21| 0.94 at 9 GHz vs 0 for connect=); a real plate
of thickness t adds ~2 pi eps t / ln(b/a), which is not modelled.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import hankel2, j0, j1, jv, kv, struve

from .constants import C0, EPS0, MU0
from .greens import _csqrt_re_pos, _gauss_panels
from .integrals import TRI7_BARY, TRI7_W, quad_points

_RQ, _WQ = np.polynomial.legendre.leggauss(24)


def ring_nodes(a, b, nr):
    return a * (b / a) ** (np.arange(nr + 1) / nr)


def ring_weight(j, r, nodes):
    """Radial basis function j (0: 1 for r < a, falling to 0 at nodes[1])."""
    r = np.asarray(r, float)
    rj, hi = nodes[j], nodes[j + 1]
    lo = nodes[j - 1] if j > 0 else 0.0
    up = (r - lo) / (rj - lo) if j > 0 else np.ones_like(r)
    w = np.where(r <= rj, up, (hi - r) / (hi - rj))
    return np.where((r >= lo) & (r <= hi), np.clip(w, 0, 1), 0.0)


def _int_j0(x):
    """Int_0^x J0(t) dt (real x >= 0)."""
    return x * j0(x) + 0.5 * np.pi * x * (j1(x) * _struve(0, x) - j0(x) * _struve(1, x))


def _struve(v, x):
    """scipy.special.struve with its isolated NaNs repaired: scipy 1.15 returns
    nan right at some zeros of H0 (e.g. x = 25.7653728), which made a whole
    frill block nan (16 GHz, 0.95 mm antipad). A nan is replaced by the mean
    of the values at x -+ 1e-4 (H_v is smooth there: error ~1e-9)."""
    s = np.asarray(struve(v, x), float)
    bad = ~np.isfinite(s)
    if bad.any():
        xb = np.broadcast_to(x, s.shape)[bad]
        s = s.copy()
        s[bad] = 0.5 * (struve(v, xb - 1e-4) + struve(v, xb + 1e-4))
    return s


def _lin_piece(k, r0, r1, al, be, bessel=None):
    """2 pi Int_r0^r1 (al + be r) J0(k r) r dr, exact for real k > 0.
    bessel: optional dict r -> (J0, J1, Int J0) at k r, shared between calls
    on the same k (the Struve functions dominate the cost)."""
    def prim(r):
        if bessel is not None and r in bessel:
            b0, b1, bi = bessel[r]
        else:
            x = k * r
            b0, b1, bi = j0(x), j1(x), _int_j0(x)
            if bessel is not None:
                bessel[r] = (b0, b1, bi)
        return (al * r * b1 / k
                + be * (r * r * b1 / k + r * b0 / k**2 - bi / k**3))
    return 2 * np.pi * (prim(r1) - prim(r0))


def ring_spectrum(j, k, nodes, bessel=None):
    """2 pi Int w_j(r) J0(k r) r dr for complex k (array). Gauss quadrature
    where k b is small (the complex part of the path), the exact closed form
    of the linear pieces on the real axis beyond (J0(kr) oscillates far too
    fast there for any fixed rule). bessel: a dict shared by the calls for
    all rings of one `nodes` on the same k (see _lin_piece)."""
    k = np.asarray(k, complex)
    out = np.zeros(k.shape, complex)
    segs = [(nodes[j], nodes[j + 1], nodes[j + 1] / (nodes[j + 1] - nodes[j]),
             -1.0 / (nodes[j + 1] - nodes[j]))]  # falling edge: w = (hi - r)/(hi - rj)
    if j > 0:
        lo, rj = nodes[j - 1], nodes[j]
        segs.insert(0, (lo, rj, -lo / (rj - lo), 1.0 / (rj - lo)))  # rising edge
    big = (np.abs(k.imag) < 1e-12 * np.abs(k)) & (np.abs(k) * nodes[-1] > 40)
    ks = k[~big]
    if ks.size:
        part = np.zeros(ks.shape, complex)
        if j == 0:
            a = nodes[0]
            kz = np.where(np.abs(ks) * a < 1e-6, 1e-6 / a, ks)
            part += 2 * np.pi * a * jv(1, kz * a) / kz
        cplx = ks.imag != 0  # complex Bessel only on the lifted part of the path (5x slower)
        for r0, r1, al, be in segs:
            r = r0 + 0.5 * (r1 - r0) * (_RQ + 1)
            wr = 0.5 * (r1 - r0) * _WQ * 2 * np.pi * r * (al + be * r)
            part[cplx] += jv(0, np.multiply.outer(ks[cplx], r)) @ wr
            part[~cplx] += j0(np.multiply.outer(ks[~cplx].real, r)) @ wr
        out[~big] = part
    if big.any():
        kb = k[big].real
        part = np.zeros(kb.shape)
        if j == 0:
            a = nodes[0]
            part += 2 * np.pi * a * j1(kb * a) / kb
        for r0, r1, al, be in segs:
            part += _lin_piece(kb, r0, r1, al, be, bessel)
        out[big] = part
    return out


def ring_area(j, nodes):
    """Int w_j dA."""
    return float(ring_spectrum(j, np.array([1e-9 / nodes[-1]]), nodes)[0].real)


class _SideTL:
    """TM transmission line of one stack section, seen from the plate.

    eps, t : layers from the plate outward; far: "pec" or "air"."""

    def __init__(self, eps, t, far, k0):
        self.eps, self.t, self.far, self.k0 = np.asarray(eps, complex), np.asarray(t, float), far, k0

    def admittances(self, kr):
        """y_in (looking into the section, far end terminated) and y_tr
        (current into the far plate per volt on the near plate; None for
        an open far end), normalised TM (y = eps_r / u)."""
        kr = np.asarray(kr, complex)
        kk = self.k0 * np.sqrt(self.eps)
        u = np.stack([_csqrt_re_pos(kr**2 - k**2) for k in kk])
        y = self.eps[:, None] / u
        x = u * self.t[:, None]
        e = np.exp(-2 * x)
        T = (1 - e) / (1 + e)
        S = 2 * np.exp(-x) / (1 + e)
        n = len(self.t)
        Y = [None] * (n + 1)  # input admittance looking outward before layer i
        if self.far == "pec":
            Y[n - 1] = y[n - 1] / T[n - 1]
        else:
            Y[n] = 1.0 / np.sqrt(kr**2 - self.k0**2)
            Y[n - 1] = y[n - 1] * (Y[n] + y[n - 1] * T[n - 1]) / (y[n - 1] + Y[n] * T[n - 1])
        for i in range(n - 2, -1, -1):
            Y[i] = y[i] * (Y[i + 1] + y[i] * T[i]) / (y[i] + Y[i + 1] * T[i])
        ytr = None
        if self.far == "pec":
            V = np.ones_like(kr)
            for i in range(n - 1):
                V = V * S[i] / (1 + Y[i + 1] / y[i] * T[i])
            ytr = y[n - 1] * V * S[n - 1] / T[n - 1]
        return Y[0], ytr


def _section_side(stack, plane, side):
    """(section index, layers from the plate outward, far termination,
    plate position in the section) above (side=+1) / below (-1) a plane."""
    k = stack.planes[plane]
    s = int(np.searchsorted(stack.bounds, k)) - (1 if side < 0 else 0)
    lo, hi = stack.bounds[s], stack.bounds[s + 1]
    lay = stack.layers[lo:hi]
    eps = [d.eps_r * (1 - 1j * d.tand) for d in lay]
    t = [d.t for d in lay]
    if side > 0:
        return s, eps, t, ("pec" if stack.section_top_pec(s) else "air"), "bottom"
    return s, eps[::-1], t[::-1], "pec", "top"


class FrillModel:
    """Frill (antipad) blocks of the impedance matrix."""

    def __init__(self, solver):
        self.solver = solver
        st, m = solver.stack, solver.mesh
        self.frills = list(getattr(m, "frills", []))
        self.offset = np.cumsum([0] + [fr["nr"] for fr in self.frills])  # first unknown of each
        self.nf = int(self.offset[-1])
        self.nodes = [ring_nodes(fr["a"], fr["b"], fr["nr"]) for fr in self.frills]
        self.sides = [[(sd, *_section_side(st, fr["plane"], sd)) for sd in (1, -1)]
                      for fr in self.frills]

    def _path(self, sub, eps, k0, extra_to, bmax):
        """Sommerfeld path (the section's LayeredGreens one if any), extended
        along the real axis to `extra_to` (frill spectra decay slowly)."""
        if sub is not None:
            kc, wc = sub._kr_c, sub._w_c
            kr, wr = sub._kr_r.astype(complex), sub._w_r
            k1 = float(sub._kr_r[-1])
        else:
            kmaxr = abs(k0 * np.sqrt(np.max(np.real(eps))))
            T = 1.3 * kmaxr + k0
            a = min(0.5 * k0, 1.0 / self.solver.rho_max)
            n1 = int(np.clip(np.ceil(2 * T / a), 100, 3000))
            tt, w = _gauss_panels(0.0, T, n1, 12)
            kc = tt + 1j * a * np.sin(np.pi * tt / T)
            wc = w * (1 + 1j * a * np.pi / T * np.cos(np.pi * tt / T))
            kr, wr, k1 = np.zeros(0, complex), np.zeros(0, complex), T
        ks, ws = [kc, kr], [wc, wr]
        if extra_to > k1:
            n = int(np.ceil((extra_to - k1) * bmax / 0.5))
            t2, w2 = _gauss_panels(k1, extra_to, max(n, 8), 8)
            ks.append(t2.astype(complex))
            ws.append(w2.astype(complex))
        return np.concatenate(ks), np.concatenate(ws)

    def blocks(self, greens, omega):
        """C (N_J x nf) and F = jw C_ff (nf x nf) at one frequency."""
        sol, st, m = self.solver, self.solver.stack, self.solver.mesh
        geom = sol.geometry()
        nb, nv, nf = m.n_basis, m.n_vias, self.nf
        k0 = omega / C0
        C = np.zeros((nb + nv, nf), complex)
        F = np.zeros((nf, nf), complex)
        bmax = max(fr["b"] for fr in self.frills)
        amin = min(fr["a"] for fr in self.frills)
        level_sec = {n: st.section(n) for n in m.layer_names}
        qp = quad_points(m.verts, TRI7_BARY)
        qw = TRI7_W[None, :] * m.area[:, None]
        cen = m.centroid
        cache = {}
        for k, fr in enumerate(self.frills):
            o, nr, nodes = self.offset[k], fr["nr"], self.nodes[k]
            xy = np.asarray(fr["xy"])
            for sd, s, eps, t, far, plate in self.sides[k]:
                sub = greens._sub.get(s) if hasattr(greens, "_sub") else None
                if (s, sd) not in cache:
                    kr, wk = self._path(sub, eps, k0, 400.0 / amin, bmax)
                    yin, ytr = _SideTL(eps, t, far, k0).admittances(kr)
                    cache[s, sd] = (kr, wk, yin, ytr)
                kr, wk, yin, ytr = cache[s, sd]

                def spectra(l, s=s, sd=sd, kr=kr):  # ring transforms of frill l on this side's path
                    if (l, s, sd) not in cache:
                        bes = {}
                        cache[l, s, sd] = np.stack([ring_spectrum(j, kr, self.nodes[l], bes)
                                                    for j in range(self.frills[l]["nr"])])
                    return cache[l, s, sd]
                wt = spectra(k)  # (nr, K)
                Ys = EPS0 * kr**2 * yin
                for l2, fr2 in enumerate(self.frills):
                    o2, nr2 = self.offset[l2], fr2["nr"]
                    d = float(np.hypot(*(np.asarray(fr2["xy"]) - xy)))
                    J = jv(0, kr * d) if d > 0 else 1.0
                    if fr2["plane"] == fr["plane"]:  # same plate, same side
                        wt2 = spectra(l2)
                        F[o:o + nr, o2:o2 + nr2] += 1j * omega * ((wt * (wk * Ys * J * kr)) @ wt2.T) / (2 * np.pi)
                    elif far == "pec" and any(s2 == s and sd2 == -sd for sd2, s2, *_ in self.sides[l2]):
                        wt2 = spectra(l2)
                        F[o:o + nr, o2:o2 + nr2] -= 1j * omega * ((wt * (wk * EPS0 * kr**2 * ytr * J * kr))
                                                                  @ wt2.T) / (2 * np.pi)
                if sd > 0:
                    C[nb + fr["seg_up"], o] += -1.0  # via current down, against n = +z
                else:
                    C[nb + fr["seg_low"], o] += 1.0  # via current down, along n = -z
                levels = [n for n in m.layer_names if level_sec[n] == s]
                if levels and sub is not None:
                    self._charges(C, o, nr, fr, sub, levels, plate, kr, wk, wt, qp, qw, cen, geom, m)
                if far == "pec":
                    self._vertical(C, o, nr, nodes, fr, s, sd, omega, geom, m, st)
        return C, F

    def _charges(self, C, o, nr, fr, sub, levels, plate, kr, wk, wt, qp, qw, cen, geom, m):
        """Charge columns: C[:, ring j] += -Int (div f) Phi_j dA."""
        nb = m.n_basis
        xy = np.asarray(fr["xy"])
        rcut = fr["b"] + 15 * float(np.sum(sub.t))
        Vh, Ve, u = sub._tl(kr)
        y = sub.epsc[:, None] / u
        x = u * sub.t[:, None]
        e = np.exp(-2 * x)
        Tt = (1 - e) / (1 + e)
        Ss = 2 * np.exp(-x) / (1 + e)
        rr = np.linspace(0, rcut, 160)
        nl = len(sub.t)
        for name in levels:
            li = sub.levels.index(name)
            if plate == "bottom":
                sig = -y[0] * Ve[li, 1] * Ss[0] / Tt[0]
            else:
                sig = -y[nl - 1] * Ve[li, nl - 1] * Ss[nl - 1] / Tt[nl - 1]
            # sig ~ exp(-k dz) (level to plate): most of the long real-axis
            # tail (kept for the ring self terms) contributes nothing here
            keep = np.abs(sig) > 1e-16 * np.abs(sig).max()
            ks, cw = kr[keep], (wt * (wk * sig * kr))[:, keep]
            cplx = ks.imag != 0
            phi = (jv(0, np.multiply.outer(rr, ks[cplx])) @ cw[:, cplx].T
                   + j0(np.multiply.outer(rr, ks[~cplx].real)) @ cw[:, ~cplx].T) / (2 * np.pi)  # (160, nr)
            spl = CubicSpline(rr, phi, axis=0)
            lidx = m.layer_names.index(name)
            on = np.nonzero((m.tri_layer == lidx) & (np.hypot(*(cen - xy).T) < rcut + 1e-3))[0]
            if not on.size:
                continue
            d = np.hypot(qp[on, :, 0] - xy[0], qp[on, :, 1] - xy[1])  # (n_on, 7)
            vals = np.where((d < rcut)[..., None], spl(np.minimum(d, rcut)), 0.0)  # (n_on,7,nr)
            iphi = np.einsum("tqj,tq->tj", vals, qw[on])  # (n_on, nr)
            bt, cv = geom.b[on], geom.cV[on]
            ok = bt >= 0
            for j in range(nr):
                val = -cv * iphi[:, j][:, None]
                np.add.at(C[:, o + j], bt[ok], val[ok])
            if geom.nv:
                pos = geom.pad_pos[on]
                sel = pos >= 0
                if sel.any():
                    C[nb:nb + geom.nv, o:o + nr] += -(geom.Dvia[pos[sel]].T @ iphi[sel])

    def _vertical(self, C, o, nr, nodes, fr, s, sd, omega, geom, m, st):
        """Vertical via currents in the closed section s: jw Int sigma^A w_j dA,
        sigma^A = eps E_z^A . n at the plate, E_z^A = -jw A_z (modal G_zz)."""
        if geom.via_spans is None:
            return
        nb = m.n_basis
        lo, hi = st.bounds[s], st.bounds[s + 1]
        zb, zt = float(st.z[lo]), float(st.z[hi])
        d = zt - zb
        lay = st.layers[lo:hi]
        eps_s = d / sum(L.t / (L.eps_r * (1 - 1j * L.tand)) for L in lay)
        kt = omega / C0 * np.sqrt(eps_s)
        zp = 0.0 if sd > 0 else d
        nz = 1.0 if sd > 0 else -1.0
        a = fr["a"]
        xy = np.asarray(fr["xy"])
        nmax = int(np.ceil(40 * d / (np.pi * a))) + 1
        n = np.arange(0, nmax + 1)
        kn = n * np.pi / d
        epsn = np.where(n == 0, 1.0, 2.0)
        edges = np.r_[0.0, nodes]
        rr = np.concatenate([r0 + 0.5 * (r1 - r0) * (_RQ + 1) for r0, r1 in zip(edges[:-1], edges[1:])])
        ww = np.concatenate([0.5 * (r1 - r0) * _WQ for r0, r1 in zip(edges[:-1], edges[1:])])
        wj = np.stack([ring_weight(j, rr, nodes) for j in range(nr)])  # (nr, R)
        area = np.array([ring_area(j, nodes) for j in range(nr)])
        gown = np.stack([_gn(nn, kt, d, rr) for nn in n])  # (n, R)
        Gown = (wj * (ww * 2 * np.pi * rr)) @ gown.T  # (nr, n)
        for mseg, (z0, z1) in enumerate(geom.via_spans):
            a1, a2 = np.clip(z0, zb, zt) - zb, np.clip(z1, zb, zt) - zb
            if a2 - a1 <= 1e-12 * d:
                continue
            v = geom.vias[mseg]
            dist = float(np.hypot(*(np.asarray(v.xy) - xy)))
            cn = np.where(n == 0, a2 - a1, (np.sin(kn * a2) - np.sin(kn * a1)) / np.where(n == 0, 1, kn))
            if dist < 1e-9:
                G = Gown
            else:
                G = np.outer(area, [_gn(nn, kt, d, np.array([dist]))[0] for nn in n])
            Az = -MU0 * (G @ (epsn / d * np.cos(kn * zp) * cn))  # (nr,) Int w_j A_z dA per unit current
            C[nb + mseg, o:o + nr] += 1j * omega * EPS0 * eps_s * (-1j * omega) * Az * nz


def _gn(n, kt, d, r):
    """Radial part of the n-th parallel-plate mode of G_zz."""
    r = np.maximum(np.asarray(r, float), 1e-12)
    if n == 0:
        return hankel2(0, kt * r) / 4j
    q = np.sqrt((n * np.pi / d) ** 2 - kt**2 + 0j)
    return kv(0, q * r) / (2 * np.pi)
