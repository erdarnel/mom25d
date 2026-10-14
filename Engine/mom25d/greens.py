"""Spatial-domain MPIE Green's functions for a grounded dielectric slab.

Geometry (z up):   air (eps0, mu0)          z > h
                   ---- conductors ----     z = h
                   substrate (eps_r)        0 < z < h
                   ==== PEC ground ====     z = 0

Only horizontal currents on the z = h interface are considered (the usual
microstrip / planar-circuit case). Then the MPIE needs just two kernels
(Mosig's formulation):

    A   = mu0      * gA  * J              (vector potential, xx = yy)
    V   = 1/eps0   * gV  * q              (scalar potential)

with spectral forms (u_i = sqrt(k_rho^2 - k_i^2), Re u_i > 0)

    gA~ = 1 / D_TE
    gV~ = (u0 + u1 tanh(u1 h)) / (D_TE * D_TM)
    D_TE = u0 + u1 coth(u1 h)
    D_TM = eps_r u0 + u1 tanh(u1 h)

The strip-to-ground line voltage  V = -Int_0^h E_z dz  of a horizontal
current sheet depends only on its charge q (TM part) through

    V   = 1/eps0   * gW  * q,     gW~ = u0 tanh(u1 h) / (u1 D_TM)

which equals gV in the static limit; it is used for port impedances.

Spatial forms follow from the Sommerfeld integral

    g(rho) = 1/(2 pi) * Int_0^inf J0(k_rho rho) k_rho g~(k_rho) dk_rho .

Numerical strategy
------------------
* The large-k_rho asymptote of each kernel is subtracted in closed form:
      gA~ ~ 1/(2 u_a)              ->  exp(-j k_a r)/(4 pi r)
      gV~ ~ c_v/(2 u_v)            ->  c_v exp(-j k_v r)/(4 pi r)
  with k_a, k_v, c_v chosen so the remainder decays like k_rho^-5.
* The remainder is integrated along a path lifted into the first quadrant
  (e^{jwt} convention) over [0, T] to step over the surface-wave poles and
  branch point, then along the real axis to K_max.
* The result is tabulated on a rho-grid; what the MoM needs is the smooth
  part S(rho) = g(rho) - c/(4 pi rho), which is interpolated.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv, j0

from .constants import C0, MU0


def _gauss_panels(a, b, n_panels, order):
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    t = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    wt = (half[:, None] * w[None, :]).ravel()
    return t, wt


def _csqrt_re_pos(z):
    s = np.sqrt(z)
    return np.where(s.real < 0, -s, s)


class GroundedSlabGreens:
    """Tabulated spatial Green's functions at one frequency.

    Parameters
    ----------
    freq : float           frequency [Hz]
    eps_r : float          substrate relative permittivity
    h : float              substrate thickness [m]
    tand : float           dielectric loss tangent
    rho_max : float        largest source-observer distance needed [m]
    zs_ground : complex    surface impedance of the ground plane [ohm]; 0 = PEC.
                           Its loss is folded into gA (see _ground_loss_kernel)
    """

    def __init__(self, freq, eps_r, h, tand=0.0, rho_max=0.1, gl_order=12, zs_ground=0.0):
        self.freq = freq
        self.h = h
        self.zs_ground = zs_ground
        self.omega = 2 * np.pi * freq
        self.k0 = self.omega / C0
        self.epsc = eps_r * (1 - 1j * tand)
        self.k1 = self.k0 * np.sqrt(self.epsc)
        k0, k1, epsc = self.k0, self.k1, self.epsc

        # closed-form asymptotic terms
        self.ka = np.sqrt(0.5 * (k0**2 + k1**2))
        self.cv = 2.0 / (epsc + 1.0)
        self.kv = np.sqrt((epsc * k0**2 + k1**2) / (epsc + 1.0))

        self.rho_max = rho_max
        self._build_path(gl_order)
        self._tabulate()
        # pair interface shared with layered.LayeredGreens (one level, dz = 0)
        self.cA, self.cV, self.cW, self.dz = 1.0, self.cv, self.cv, 0.0
        self.levels = ["top"]

    def pair(self, i=0, j=0):
        return self

    def gW(self, R):
        return self.cv / (4 * np.pi * R) + self.smooth_w(R)

    # ------------------------------------------------------------------ spectral
    def spectral(self, kr):
        """Return (gA~, gV~) at complex k_rho."""
        k0, k1, h, epsc = self.k0, self.k1, self.h, self.epsc
        u0 = np.sqrt(kr**2 - k0**2)  # principal branch is proper on our path
        u1 = _csqrt_re_pos(kr**2 - k1**2)  # kernels are even in u1
        x = u1 * h
        e = np.exp(-2 * x)
        small = np.abs(x) < 1e-4
        with np.errstate(divide="ignore", invalid="ignore"):
            u1coth = np.where(small, 1 / h + u1**2 * h / 3, u1 * (1 + e) / (1 - e))
            u1tanh = np.where(small, u1**2 * h, u1 * (1 - e) / (1 + e))
        dte = u0 + u1coth
        dtm = epsc * u0 + u1tanh
        gA = 1.0 / dte
        gV = (u0 + u1tanh) / (dte * dtm)
        with np.errstate(divide="ignore", invalid="ignore"):
            tanh_over_u1 = np.where(small, h, u1tanh / u1**2)
        gW = u0 * tanh_over_u1 / dtm
        return gA, gV, gW

    def _remainder(self, kr):
        gA, gV, gW = self.spectral(kr)
        ua = np.sqrt(kr**2 - self.ka**2)
        uv = np.sqrt(kr**2 - self.kv**2)
        return gA - 0.5 / ua, gV - 0.5 * self.cv / uv, gW - 0.5 * self.cv / uv

    # --------------------------------------------------------------- integration
    def _build_path(self, order):
        k0, k1r, h = self.k0, abs(self.k1), self.h
        T = 1.3 * k1r + k0
        # lift: large enough to clear poles, small enough that J0 growth
        # exp(a*rho_max) stays benign
        self.lift = a = min(0.5 * k0, 1.0 / self.rho_max)
        n1 = int(np.clip(np.ceil(2 * T / a), 100, 3000))
        t, w = _gauss_panels(0.0, T, n1, order)
        kr1 = t + 1j * a * np.sin(np.pi * t / T)
        dk1 = w * (1 + 1j * a * np.pi / T * np.cos(np.pi * t / T))

        kmax = max(40 * k1r, 20.0 / h, T * 1.5)
        dk = min(np.pi / self.rho_max, 0.5 / h, k1r)  # panel width
        n2 = int(np.ceil((kmax - T) / dk))
        t2, w2 = _gauss_panels(T, kmax, n2, 8)
        self._kr_c, self._w_c = kr1, dk1
        self._kr_r, self._w_r = t2, w2.astype(complex)
        self.kmax = kmax

    def _sommerfeld(self, rho, fs, kr, w, real_axis):
        """(1/2pi) Int J0(kr rho) kr f(kr) dkr for many rho (chunked)."""
        C = np.stack([w * kr * f / (2 * np.pi) for f in fs], 1)
        out = np.empty((rho.size, len(fs)), complex)
        chunk = max(1, int(4e6 // kr.size))
        for s in range(0, rho.size, chunk):
            arg = np.outer(rho[s:s + chunk], kr)
            J = j0(arg.real) if real_axis else jv(0, arg)
            out[s:s + chunk] = J @ C
        return out

    def residual(self, rho):
        """Spatial remainders (R_A, R_V, R_W) after removing closed-form terms."""
        rho = np.atleast_1d(np.asarray(rho, float))
        r1 = self._sommerfeld(rho, self._remainder(self._kr_c), self._kr_c, self._w_c, False)
        r2 = self._sommerfeld(rho, self._remainder(self._kr_r), self._kr_r.real, self._w_r, True)
        return tuple((r1 + r2).T)

    def _tabulate(self):
        h, rmax = self.h, self.rho_max
        lam = 2 * np.pi / abs(self.k1)
        # local spacing: fine near the source (image terms vary on scale h),
        # never coarser than lambda/80 (surface-wave oscillation)
        r = [0.0]
        while r[-1] < rmax:
            r.append(r[-1] + min(lam / 80, max(h / 25, r[-1] / 25)))
        rho = np.array(r)
        RA, RV, RW = self.residual(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            fa = np.where(rho > 0, np.expm1(-1j * self.ka * rho) / (4 * np.pi * rho),
                          -1j * self.ka / (4 * np.pi))
            fv = np.where(rho > 0, np.expm1(-1j * self.kv * rho) / (4 * np.pi * rho),
                          -1j * self.kv / (4 * np.pi))
        self.rho_tab = rho
        self.SA_tab = fa + RA + self._ground_loss_kernel(rho)
        self.SV_tab = self.cv * fv + RV
        self.SW_tab = self.cv * fv + RW
        self._spl = CubicSpline(rho, np.stack([self.SA_tab, self.SV_tab], axis=-1))
        self._splw = CubicSpline(rho, self.SW_tab)

    def _ground_loss_kernel(self, rho):
        """Ground-plane conductor loss as an addition to gA.

        Quasi-statically the ground carries J_g = -P_h * K (P_h: Poisson
        kernel of height h, the image-doubled Biot-Savart field), so its
        dissipation Zs Int |J_g|^2 = Zs <K, P_h * P_h * K> = Zs <K, P_2h * K>,
        which has the form of the vector-potential term:
            jw mu0 gA  ->  jw mu0 gA + Zs P_2h(rho),
            P_2h(rho) = 2h / (2 pi (rho^2 + 4 h^2)^1.5).
        P_2h is smooth on the scale 2h, so it lives in the tabulated part.
        """
        if not self.zs_ground:
            return 0.0
        h2 = 2 * self.h
        p2h = h2 / (2 * np.pi * (rho**2 + h2**2) ** 1.5)
        return self.zs_ground / (1j * self.omega * MU0) * p2h

    # ------------------------------------------------------------------ public
    def smooth(self, R):
        """Smooth parts S_A(R), S_V(R); g = c/(4 pi R) + S with c_A = 1, c_V = cv."""
        S = self._spl(R)
        return S[..., 0], S[..., 1]

    def smooth_w(self, R):
        """Smooth part of the line-voltage kernel; gW = cv/(4 pi R) + S_W."""
        return self._splw(R)

    def gA(self, R):
        return 1 / (4 * np.pi * R) + self.smooth(R)[0]

    def gV(self, R):
        return self.cv / (4 * np.pi * R) + self.smooth(R)[1]
