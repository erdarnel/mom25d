"""Multilayer stackups and their MPIE Green's functions.

Geometry (z up):
                    air half-space (or a PEC cover)          top="air" | "pec"
    z_N  --------   metal level (interface N; only with top="air")
         layer N    eps_N, tand_N
    z_k  --------   metal level k (any interface 1..N may carry metal)
    z_p  ========   optional solid PEC plane (`planes`, e.g. an inner GND)
         ...
         layer 1    eps_1, tand_1
    z_0 = 0 =====   PEC ground ("gnd")

Solid inner planes split the stack into sections that do not couple: each
section is a stack of its own (ground = the plane below, cover = the plane
above), `SectionedGreens` tabulates them separately and kernels between
levels of different sections are zero. Only vias pass through a plane
(through an antipad); the solver handles their vertical currents.

Metal levels are named; currents on them are horizontal (RWG). Vertical
currents (vias) are handled quasi-statically in the solver.

Green's functions follow Michalski & Zheng's transmission-line (TL) form:
for a unit shunt current source at level j and observation level i, with
V^h, V^e the TE / TM line voltages normalised so that the TE admittance of
layer k is u_k and the TM admittance eps_k / u_k (u_k = sqrt(k_rho^2 - k_k^2)),

    gA~(i|j) = V^h_i                               (= 1/D_TE for one slab)
    gV~(i|j) = (V^e_i + k0^2 V^h_i) / k_rho^2      (formulation-C K^phi / eps0)
    gW~(i|j) = sum_{k<=i} (V^e(z_k) - V^e(z_{k-1})) / u_k^2
              (line voltage -Int_0^{z_i} E_z dz from the source charge; it
               is the slab's gW = u0 tanh(u1 h)/(u1 D_TM) for one layer)

and g(rho) = (1/2pi) Int J0(k_rho rho) k_rho g~ dk_rho. As in greens.py the
leading term c e^{-jkR}/(4 pi R), R = sqrt(rho^2 + dz^2), is removed in
closed form (c from the two media at a level for dz = 0, from the spectral
asymptote for dz > 0) and the remainder is tabulated and spline-interpolated.
With one layer this reproduces GroundedSlabGreens (tested).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import j0, jv

from .constants import C0, MU0
from .greens import _csqrt_re_pos, _gauss_panels

GND = "gnd"


@dataclass
class Dielectric:
    t: float  # thickness [m]
    eps_r: float
    tand: float = 0.0


@dataclass
class Stackup:
    """Dielectric layers listed bottom (on the ground plane) to top, and
    the named metal levels.

    layers   : [Dielectric, ...] bottom -> top
    metals   : {name: interface index 1..N}; interface k is the top of layer
               k. Default {"top": N}. The ground plane is level "gnd" (0).
    top      : "air" (open) or "pec" (cover on top of layer N, e.g. stripline)
    sigma, t : conductor conductivity [S/m] (None: PEC) and thickness [m]
    ground_loss : include the ground plane's conductor loss (in every
               section the loss of the plane below it; covers are lossless)
    planes   : {name: interface index 1..N-1} solid PEC planes inside the
               stack (inner ground / power planes); vias may end on them
    """

    layers: list
    metals: dict = None
    top: str = "air"
    sigma: float = None
    t: float = 35e-6
    ground_loss: bool = True
    planes: dict = None

    def __post_init__(self):
        if not self.layers:
            raise ValueError("stackup needs at least one dielectric layer")
        self.layers = [d if isinstance(d, Dielectric) else Dielectric(*d) for d in self.layers]
        n = len(self.layers)
        if self.metals is None:
            self.metals = {"top": n}
        if self.top not in ("air", "pec"):
            raise ValueError("top must be 'air' or 'pec'")
        for name, k in self.metals.items():
            if name == GND:
                raise ValueError("'gnd' is reserved for the ground plane")
            if not 1 <= k <= n:
                raise ValueError(f"metal level {name}: interface {k} not in 1..{n}")
            if k == n and self.top == "pec":
                raise ValueError(f"metal level {name} would lie on the PEC cover")
        self.planes = dict(self.planes or {})
        for name, k in self.planes.items():
            if name == GND or name in self.metals:
                raise ValueError(f"plane name {name!r} is 'gnd' or a metal level")
            if not 1 <= k <= n - 1:
                raise ValueError(f"plane {name}: interface {k} not in 1..{n - 1} "
                                 "(use top='pec' for a cover)")
            if k in self.metals.values():
                raise ValueError(f"plane {name} lies on a metal level")
        if len(set(self.planes.values())) != len(self.planes):
            raise ValueError("two planes on the same interface")
        self.bounds = [0] + sorted(self.planes.values()) + [n]  # section boundaries
        self.z = np.concatenate([[0.0], np.cumsum([d.t for d in self.layers])])

    @classmethod
    def microstrip(cls, eps_r, h, tand=0.0, sigma=None, t=35e-6, ground_loss=True):
        return cls([Dielectric(h, eps_r, tand)], {"top": 1}, "air", sigma, t, ground_loss)

    @property
    def lossless(self):
        return not self.sigma and not any(d.tand for d in self.layers)

    def level_z(self, name):
        if name == GND:
            return 0.0
        return float(self.z[self.metals[name] if name in self.metals else self.planes[name]])

    def section(self, name):
        """Index of the section (between ground / planes) holding a metal level."""
        return int(np.searchsorted(self.bounds, self.metals[name])) - 1

    def section_top_pec(self, s):
        return s < len(self.bounds) - 2 or self.top == "pec"

    def level_h(self, name):
        """Height of a metal level above the plane (or ground) below it."""
        return self.level_z(name) - float(self.z[self.bounds[self.section(name)]])

    def eps_below(self, name):
        """Thickness-weighted eps_r between a metal level and the plane below."""
        k = self.metals[name]
        lo = self.bounds[self.section(name)]
        t = np.array([d.t for d in self.layers[lo:k]])
        e = np.array([d.eps_r for d in self.layers[lo:k]])
        return float(t @ e / t.sum())

    def covered(self, name):
        """True if there is dielectric (or a cover / plane) above the level."""
        s = self.section(name)
        return self.metals[name] < self.bounds[s + 1] or self.section_top_pec(s)

    def section_stack(self, s):
        """Section s as a Stackup of its own (z measured from its bottom)."""
        lo, hi = self.bounds[s], self.bounds[s + 1]
        metals = {n: k - lo for n, k in self.metals.items() if lo < k <= hi}
        return Stackup(self.layers[lo:hi], metals, "pec" if self.section_top_pec(s) else "air",
                       self.sigma, self.t, self.ground_loss)

    def via_sections(self, freq):
        """[(z_bottom, z_top, k_tem or None)] per section for via currents:
        k_tem is the parallel-plate TEM wavenumber (series-averaged eps) of a
        section closed by PEC on both sides, None for an open (air) top."""
        k0 = 2 * np.pi * freq / C0
        out = []
        for s in range(len(self.bounds) - 1):
            lo, hi = self.bounds[s], self.bounds[s + 1]
            k = None
            if self.section_top_pec(s):
                lay = self.layers[lo:hi]
                inv = sum(d.t / (d.eps_r * (1 - 1j * d.tand)) for d in lay) / sum(d.t for d in lay)
                k = k0 * np.sqrt(1 / inv)
            out.append((float(self.z[lo]), float(self.z[hi]), k))
        return out

    def surface_impedance(self, freq, t=None):
        """One-sided skin-effect impedance of a conductor slab of thickness t
        (default self.t), -> 1/(sigma t) at DC."""
        if not self.sigma:
            return 0.0
        w = 2 * np.pi * freq
        kc = np.sqrt(1j * w * MU0 * self.sigma)
        return kc / self.sigma / np.tanh(kc * (self.t if t is None else t))


class _Pair:
    """Kernels between two metal levels: g = c/(4 pi R) + S(rho),
    R = sqrt(rho^2 + dz^2)."""

    def __init__(self, dz, cA, cV, cW, spl, splw):
        self.dz, self.cA, self.cV, self.cW = dz, cA, cV, cW
        self.cv = cV  # GroundedSlabGreens naming
        self._spl, self._splw = spl, splw

    def smooth(self, rho):
        S = self._spl(rho)
        return S[..., 0], S[..., 1]

    def smooth_w(self, rho):
        return self._splw(rho)

    def _R(self, rho):
        return np.sqrt(np.asarray(rho) ** 2 + self.dz**2)

    def gA(self, rho):
        return self.cA / (4 * np.pi * self._R(rho)) + self.smooth(rho)[0]

    def gV(self, rho):
        return self.cV / (4 * np.pi * self._R(rho)) + self.smooth(rho)[1]

    def gW(self, rho):
        return self.cW / (4 * np.pi * self._R(rho)) + self.smooth_w(rho)


class LayeredGreens:
    """Tabulated spatial Green's functions between all metal levels of a
    Stackup at one frequency. `pair(i, j)` (level names or indices into
    `levels`) gives the kernels for observer level i and source level j."""

    def __init__(self, freq, stack, rho_max=0.1, gl_order=12, zs_ground=0.0, levels=None):
        if stack.planes:
            raise ValueError("stackup with inner planes: use SectionedGreens")
        self.freq = freq
        self.stack = stack
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / C0
        self.zs_ground = zs_ground
        self.levels = list(levels) if levels is not None else list(stack.metals)
        self.zl = np.array([stack.level_z(n) for n in self.levels])
        self.il = np.array([stack.metals[n] for n in self.levels])
        self.epsc = np.array([d.eps_r * (1 - 1j * d.tand) for d in stack.layers])
        self.kk = self.k0 * np.sqrt(self.epsc)
        self.t = np.array([d.t for d in stack.layers])
        self.h = float(stack.z[-1])
        self.k1 = self.kk[0]
        self.rho_max = rho_max
        self._build_path(gl_order)
        self._tabulate()

    # ------------------------------------------------------------- spectral
    def _tl(self, kr):
        """Normalised TE / TM voltages at every interface for a unit current
        source at each metal level: Vh, Ve of shape (L, N+1, K)."""
        N, K, L = len(self.t), kr.size, len(self.levels)
        u = np.stack([_csqrt_re_pos(kr**2 - k**2) for k in self.kk])  # (N,K)
        u0 = np.sqrt(kr**2 - self.k0**2)  # air: principal branch (proper on the path)
        x = u * self.t[:, None]
        e = np.exp(-2 * x)
        T = (1 - e) / (1 + e)  # tanh(u t)
        S = 2 * np.exp(-x) / (1 + e)  # sech(u t)
        out = []
        for y, yair in ((u, u0), (self.epsc[:, None] / u, 1.0 / u0)):
            # input admittances looking down / up at interfaces 1..N
            ydn = np.empty((N + 1, K), complex)
            ydn[1] = y[0] / T[0]  # layer 1 shorted by the ground
            for k in range(2, N + 1):
                yl, yk, tk = ydn[k - 1], y[k - 1], T[k - 1]
                ydn[k] = yk * (yl + yk * tk) / (yk + yl * tk)
            yup = np.empty((N + 1, K), complex)
            pec_top = self.stack.top == "pec"
            if not pec_top:
                yup[N] = yair
            for k in range(N - 1, 0, -1):
                yk, tk = y[k], T[k]  # layer k+1
                if k == N - 1 and pec_top:
                    yup[k] = yk / tk
                else:
                    yl = yup[k + 1]
                    yup[k] = yk * (yl + yk * tk) / (yk + yl * tk)
            V = np.zeros((L, N + 1, K), complex)
            for l, s in enumerate(self.il):
                V[l, s] = 1.0 / (ydn[s] + yup[s])
                for k in range(s, 1, -1):  # down to interface 1 (V_0 = 0)
                    V[l, k - 1] = V[l, k] * S[k - 1] / (1 + ydn[k - 1] / y[k - 1] * T[k - 1])
                for k in range(s, N):  # up through layer k+1
                    if k + 1 == N and pec_top:
                        break  # V_N = 0 on the cover
                    V[l, k + 1] = V[l, k] * S[k] / (1 + yup[k + 1] / y[k] * T[k])
            out.append(V)
        return out[0], out[1], u

    def spectral(self, kr):
        """gA~, gV~, gW~ of shape (L_obs, L_src, K)."""
        kr = np.asarray(kr, complex)
        Vh, Ve, u = self._tl(kr)
        il = self.il
        gA = Vh[:, il].transpose(1, 0, 2)
        gV = ((Ve[:, il] + self.k0**2 * Vh[:, il]) / kr**2).transpose(1, 0, 2)
        dV = np.diff(Ve, axis=1) / u[None] ** 2  # (L, N, K): layer k contribution
        cum = np.cumsum(dV, axis=1)  # sum over layers 1..k
        gW = cum[:, il - 1].transpose(1, 0, 2)
        return gA, gV, gW

    # ------------------------------------------------------ leading terms
    def _media(self, i):
        """(eps, k) above and below metal level index i."""
        n = self.il[i]
        eb, kb = self.epsc[n - 1], self.kk[n - 1]
        if n == len(self.t):
            ea, ka = 1.0, self.k0
        else:
            ea, ka = self.epsc[n], self.kk[n]
        return ea, ka, eb, kb

    def _leading(self):
        """Per pair: dz, (cA, cV, cW), (kA, kV) of c e^{-jkR}/(4 pi R)."""
        L = len(self.levels)
        self.dz = np.abs(self.zl[:, None] - self.zl[None, :])
        self.c = np.zeros((3, L, L), complex)
        self.kx = np.zeros((2, L, L), complex)
        tmin = self.t.min()
        Kbig = 15.0 / tmin
        big = None
        for i in range(L):
            for j in range(L):
                if i == j:
                    ea, ka, eb, kb = self._media(i)
                    cv = 2.0 / (ea + eb)
                    self.c[:, i, j] = (1.0, cv, cv)
                    self.kx[0, i, j] = np.sqrt(0.5 * (ka**2 + kb**2))
                    self.kx[1, i, j] = np.sqrt((eb * ka**2 + ea * kb**2) / (ea + eb))
                    continue
                if big is None:
                    big = [g[..., 0] for g in self.spectral(np.array([Kbig]))]
                dz = self.dz[i, j]
                ux = np.sqrt(Kbig**2 - self.k0**2 + 0j)
                f = 2 * ux * np.exp(ux * dz)
                cA, cV, cW = (f * g[i, j] for g in big)
                if self.zl[j] < self.zl[i]:
                    cW = 0.0  # source below the observer: -Int E_z dz is finite at rho = 0
                self.c[:, i, j] = (cA, cV, cW)
                self.kx[:, i, j] = self.k0
        # round tiny numerical noise of the static coefficients
        self.c = np.where(np.abs(self.c.imag) < 1e-9 * np.abs(self.c), self.c.real, self.c)

    def _remainder(self, kr):
        gA, gV, gW = self.spectral(kr)
        out = []
        for m, g in enumerate((gA, gV, gW)):
            kx = self.kx[min(m, 1)]
            ux = np.sqrt(kr[None, None] ** 2 - kx[..., None] ** 2)
            out.append(g - self.c[m][..., None] * np.exp(-ux * self.dz[..., None]) / (2 * ux))
        return out

    # ------------------------------------------------------------ integration
    def _build_path(self, order):
        k0 = self.k0
        kmaxr = np.abs(self.kk).max()
        T = 1.3 * kmaxr + k0
        self.lift = a = min(0.5 * k0, 1.0 / self.rho_max)
        n1 = int(np.clip(np.ceil(2 * T / a), 100, 3000))
        t, w = _gauss_panels(0.0, T, n1, order)
        kr1 = t + 1j * a * np.sin(np.pi * t / T)
        dk1 = w * (1 + 1j * a * np.pi / T * np.cos(np.pi * t / T))
        tmin = self.t.min()
        kmax = max(40 * kmaxr, 20.0 / tmin, T * 1.5)
        dk = min(np.pi / self.rho_max, 0.5 / tmin, kmaxr)
        n2 = int(np.ceil((kmax - T) / dk))
        t2, w2 = _gauss_panels(T, kmax, n2, 8)
        self._kr_c, self._w_c = kr1, dk1
        self._kr_r, self._w_r = t2, w2.astype(complex)

    @staticmethod
    def _sommerfeld(rho, C, kr, real_axis):
        out = np.empty((rho.size, C.shape[1]), complex)
        chunk = max(1, int(4e6 // kr.size))
        for s in range(0, rho.size, chunk):
            arg = np.outer(rho[s:s + chunk], kr)
            J = j0(arg.real) if real_axis else jv(0, arg)
            out[s:s + chunk] = J @ C
        return out

    def _residual(self, rho):
        L = len(self.levels)
        res = 0.0
        for kr, w, real in ((self._kr_c, self._w_c, False), (self._kr_r, self._w_r, True)):
            rem = np.stack(self._remainder(kr if not real else kr.astype(complex)))  # (3,L,L,K)
            C = (w * kr / (2 * np.pi))[None] * rem.reshape(-1, kr.size)
            res = res + self._sommerfeld(rho, C.T, kr.real if real else kr, real)
        return res.T.reshape(3, L, L, rho.size)

    def _tabulate(self):
        self._leading()
        L = len(self.levels)
        tmin, rmax = self.t.min(), self.rho_max
        lam = 2 * np.pi / np.abs(self.kk).max()
        r = [0.0]
        while r[-1] < rmax:
            r.append(r[-1] + min(lam / 80, max(tmin / 25, r[-1] / 25)))
        rho = np.array(r)
        RES = self._residual(rho)
        self.rho_tab = rho
        self._pairs = {}
        for i in range(L):
            for j in range(L):
                dz = self.dz[i, j]
                R = np.sqrt(rho**2 + dz**2)
                S = []
                for m in range(3):
                    k = self.kx[min(m, 1), i, j]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        f = np.where(R > 0, np.expm1(-1j * k * R) / (4 * np.pi * R), -1j * k / (4 * np.pi))
                    S.append(self.c[m, i, j] * f + RES[m, i, j])
                if self.zs_ground:
                    d = self.zl[i] + self.zl[j]
                    S[0] = S[0] + self.zs_ground / (1j * self.omega * MU0) * d / (2 * np.pi * (rho**2 + d**2) ** 1.5)
                spl = CubicSpline(rho, np.stack([S[0], S[1]], -1))
                self._pairs[i, j] = _Pair(dz, *self.c[:, i, j], spl, CubicSpline(rho, S[2]))

    # ----------------------------------------------------------------- public
    def index(self, level):
        return level if isinstance(level, (int, np.integer)) else self.levels.index(level)

    def pair(self, i, j):
        return self._pairs[self.index(i), self.index(j)]

    def via_sections(self):
        return self.stack.via_sections(self.freq)


class _ZeroPair:
    """Kernels between levels separated by a solid plane: identically zero."""

    cA = cV = cW = cv = 0.0
    zero = True

    def __init__(self, dz):
        self.dz = dz

    def smooth(self, rho):
        z = np.zeros(np.shape(rho), complex)
        return z, z

    def smooth_w(self, rho):
        return np.zeros(np.shape(rho), complex)

    gA = gV = gW = smooth_w


class SectionedGreens:
    """Green's functions of a Stackup with solid inner planes: one
    LayeredGreens per section (levels referenced to the plane below them),
    zero kernels between sections. Same interface as LayeredGreens."""

    def __init__(self, freq, stack, rho_max=0.1, gl_order=12, zs_ground=0.0, levels=None):
        self.freq = freq
        self.stack = stack
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / C0
        self.zs_ground = zs_ground
        self.levels = list(levels) if levels is not None else list(stack.metals)
        self.zl = np.array([stack.level_z(n) for n in self.levels])
        self.h = float(stack.z[-1])
        self.sec = [stack.section(n) for n in self.levels]
        self._sub = {}
        for s in sorted(set(self.sec)):
            names = [n for n, q in zip(self.levels, self.sec) if q == s]
            self._sub[s] = LayeredGreens(freq, stack.section_stack(s), rho_max, gl_order,
                                         zs_ground, names)

    def index(self, level):
        return level if isinstance(level, (int, np.integer)) else self.levels.index(level)

    def pair(self, i, j):
        i, j = self.index(i), self.index(j)
        if self.sec[i] != self.sec[j]:
            return _ZeroPair(abs(self.zl[i] - self.zl[j]))
        return self._sub[self.sec[i]].pair(self.levels[i], self.levels[j])

    def via_sections(self):
        return self.stack.via_sections(self.freq)


def make_greens(freq, stack, rho_max=0.1, gl_order=12, zs_ground=0.0, levels=None):
    """LayeredGreens, or SectionedGreens for a stack with inner planes."""
    cls = SectionedGreens if stack.planes else LayeredGreens
    return cls(freq, stack, rho_max, gl_order, zs_ground, levels)
