"""Fields derived from a MoM solution, on regular display grids.

plane_currents : surface current in the solid planes (ground / inner planes)
efield         : electric field in the volume (stack and air), quasi-static
                 in the charge sense, see below

Both work on a grid of cell centres  x = x0 + (i + 1/2) dx,  y = y0 + (j + 1/2) dx
(`Grid`), from the solution vector x of `Solver.solve_currents` (or one column
of `SweepResult.currents` / `wave_currents`).

Plane currents (per face, summed to the plane's net current):
- face toward a section holding meshed copper (or open to air): the
  return current of the copper's horizontal currents, the retarded image
  J = -T * K, T(k) = exp(-h sqrt(k^2 - k1^2)) (homogeneous medium of the
  dielectric between copper and face; static limit: the Poisson kernel of
  height h), plus the current a via segment drives into the
  face, I / (2 pi rho) h / sqrt(rho^2 + h^2) radial (h = its length in the
  section);
- face of a closed section without copper (between two planes): the TEM
  radial wave of the via currents in it, K = -j k I / 4 H1(2)(k rho) radial
  (exact beyond ~ one section height from a via; the evanescent TM_n near
  fields are left out).

Electric field: E = -grad(phi) from the triangle charges, phi with the
solver's own scalar-potential kernel (gV, formulation C) extended to any
height inside the layers (TL voltages interpolated within a layer); the
-jw A part is left out (error ~ (k1 h)^2 next to the copper, ~6 % at
k1 h = 0.25), as are the antipad aperture fields. Closed sections without
copper get the TEM field of their via currents, E_z = -(w mu0 / 4) I H0(2)(k rho).
Charges are deposited on a grid of dx / 2 and convolved (FFT) with the
kernels, so points closer than ~dx to charged copper are smoothed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve
from scipy.special import hankel2, j0, j1, jv

from .constants import EPS0, MU0
from .integrals import TRI7_BARY, TRI7_W, quad_points


@dataclass
class Grid:
    """Display grid: nx x ny cells of size dx from (x0, y0) [m]."""

    x0: float
    y0: float
    dx: float
    nx: int
    ny: int

    @classmethod
    def around(cls, mesh, dx, margin=0.0, bounds=None):
        """Grid covering the mesh (or bounds (x0, y0, x1, y1)) plus margin."""
        x0, y0 = mesh.nodes.min(0) - margin if bounds is None else np.asarray(bounds[:2]) - margin
        x1, y1 = mesh.nodes.max(0) + margin if bounds is None else np.asarray(bounds[2:]) + margin
        return cls(float(x0), float(y0), dx, int(np.ceil((x1 - x0) / dx)), int(np.ceil((y1 - y0) / dx)))

    @property
    def xc(self):
        return self.x0 + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def yc(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.dx

    def fine(self):
        """The half-size deposition grid (2 nx x 2 ny)."""
        return Grid(self.x0, self.y0, self.dx / 2, 2 * self.nx, 2 * self.ny)


def _deposit(g, pts, w):
    """Sum w (..., ncomp) of points pts (N, 2) into the cells of grid g:
    (ncomp, ny, nx)."""
    w = w.reshape(len(pts), -1)
    ix = np.floor((pts[:, 0] - g.x0) / g.dx).astype(int)
    iy = np.floor((pts[:, 1] - g.y0) / g.dx).astype(int)
    ok = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
    out = np.zeros((w.shape[1], g.ny, g.nx), w.dtype)
    for c in range(w.shape[1]):
        np.add.at(out[c], (iy[ok], ix[ok]), w[ok, c])
    return out


def _extended(grid, mesh):
    """Fine (dx / 2) deposition grid covering the display grid and the whole
    mesh, aligned with the display cells, and the slices that cut the
    display region out of its coarsened result."""
    dx = grid.dx
    (mx0, my0), (mx1, my1) = mesh.nodes.min(0), mesh.nodes.max(0)
    i0 = min(int(np.floor((mx0 - grid.x0) / dx)), 0)
    j0 = min(int(np.floor((my0 - grid.y0) / dx)), 0)
    i1 = max(int(np.ceil((mx1 - grid.x0) / dx)), grid.nx)
    j1 = max(int(np.ceil((my1 - grid.y0) / dx)), grid.ny)
    eg = Grid(grid.x0 + i0 * dx, grid.y0 + j0 * dx, dx, i1 - i0, j1 - j0)
    return eg.fine(), (slice(-j0, -j0 + grid.ny), slice(-i0, -i0 + grid.nx))


def _coarsen(a):
    """(..., 2 ny, 2 nx) -> (..., ny, nx) cell means."""
    s = a.shape
    return a.reshape(s[:-2] + (s[-2] // 2, 2, s[-1] // 2, 2)).mean(axis=(-3, -1))


def _offsets(g):
    """Kernel offsets (dx, dy) of a (2 ny - 1, 2 nx - 1) kernel for grid g."""
    ox = (np.arange(2 * g.nx - 1) - (g.nx - 1)) * g.dx
    oy = (np.arange(2 * g.ny - 1) - (g.ny - 1)) * g.dx
    return np.meshgrid(ox, oy)


def _conv(a, k):
    return fftconvolve(a.real, k, "same") + 1j * fftconvolve(a.imag, k, "same") if np.iscomplexobj(a) \
        else fftconvolve(a, k, "same")


def _cconv(a, k):
    """Convolution of complex data with a complex kernel."""
    return (fftconvolve(a.real, k.real, "same") - fftconvolve(a.imag, k.imag, "same")
            + 1j * (fftconvolve(a.real, k.imag, "same") + fftconvolve(a.imag, k.real, "same")))


def current_at_quad(mesh, x):
    """Surface current density K (Nt, 7, 2) [A/m] at the 7-point quadrature
    points of every triangle, and those points (Nt, 7, 2) and weights (Nt, 7)."""
    b = mesh.tri_basis
    cA = mesh.tri_sign * np.where(b >= 0, mesh.edge_len[np.maximum(b, 0)], 0.0) / (2 * mesh.area[:, None])
    qp = quad_points(mesh.verts, TRI7_BARY)
    xb = np.where(b >= 0, x[np.maximum(b, 0)], 0)
    rv = qp[:, :, None, :] - mesh.verts[:, None, :, :]
    K = np.einsum("ti,tqid,ti->tqd", cA, rv, xb)
    return K, qp, TRI7_W[None, :] * mesh.area[:, None]


def _segments(solver):
    """Via segments: (xy (n, 2), radius (n,), z_bottom, z_top) [m]."""
    m, st = solver.mesh, solver.stack
    z = {n: st.level_z(n) for n in list(m.layer_names) + list(getattr(m, "planes", []))}
    z["gnd"] = 0.0
    segs = m.via_segments
    xy = np.array([m.vias[g["via"]].xy for g in segs]).reshape(-1, 2)
    r = np.array([m.vias[g["via"]].drill / 2 for g in segs])
    lo = np.array([z[g["bottom"]] for g in segs])
    hi = np.array([z[g["top"]] for g in segs])
    return xy, r, lo, hi


def antipads(mesh):
    """[(plane, (x, y), outer radius b), ...] of the modelled antipads."""
    return [(fr["plane"], fr["xy"], fr["b"]) for fr in getattr(mesh, "frills", [])]


# ------------------------------------------------------------------ planes
def solver_k(st, freq, za, zb):
    """Wavenumber of the dielectric between heights za < zb (thickness-
    weighted complex eps)."""
    from .constants import C0

    w = eps = 0.0
    for k, d in enumerate(st.layers):
        o = max(0.0, min(zb, st.z[k + 1]) - max(za, st.z[k]))
        w, eps = w + o, eps + o * d.eps_r * (1 - 1j * d.tand)
    return 2 * np.pi * freq / C0 * np.sqrt(eps / w if w > 0 else 1.0)


def _image(dens, d, h, k1, pad):
    """Return current of a PEC plane under the sheet current density dens
    (2, ny, nx) at height h in a medium of wavenumber k1: -T * K with
    T(k) = exp(-u h), u = sqrt(k^2 - k1^2) (Re u >= 0) -- the retarded image
    (homogeneous medium); its static limit is the Poisson kernel P_h, but
    unlike P_h it does not smooth a wave travelling at ~k1 (a TEM line's
    ground current equals the strip current; P_h lost exp(-beta h), 9 % on a
    0.5 mm substrate at 5 GHz). Zero padding by `pad` cells."""
    a = np.pad(dens, ((0, 0), (pad, pad), (pad, pad)))
    ny, nx = a.shape[1:]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d)
    kr2 = kx[None, :] ** 2 + ky[:, None] ** 2
    u = np.sqrt(kr2 - k1**2 + 0j)
    u = np.where(u.real < 0, -u, u)
    T = np.exp(-u * h)
    out = np.fft.ifft2(np.fft.fft2(a) * T)
    return out[:, pad:pad + dens.shape[1], pad:pad + dens.shape[2]]


def plane_currents(solver, x, freq, grid, planes=None, kernel_radius=5e-3, per_face=False):
    """Net surface current of each plane (vector sum of its two faces) on
    `grid`: {plane: (2, ny, nx) complex [A/m]} (with per_face: {plane:
    {"below" / "above": ...}} in solver z). planes: names (default all inner
    planes of the stack; "gnd" may be added). See the module notes."""
    m, st = solver.mesh, solver.stack
    planes = list(st.planes) if planes is None else list(planes)
    fg, sl = _extended(grid, m)
    K, qp, qw = current_at_quad(m, x)
    nb, nv = m.n_basis, m.n_vias
    Iv = x[nb:nb + nv]
    sxy, sr, slo, shi = _segments(solver)
    GX, GY = np.meshgrid(grid.xc, grid.yc)
    ox, oy = _offsets(fg)
    nr = int(np.ceil(kernel_radius / fg.dx))
    cy, cx = (np.array(ox.shape) - 1) // 2
    ox, oy = ox[cy - min(nr, cy):cy + min(nr, cy) + 1, cx - min(nr, cx):cx + min(nr, cx) + 1], \
        oy[cy - min(nr, cy):cy + min(nr, cy) + 1, cx - min(nr, cx):cx + min(nr, cx) + 1]
    secs = st.via_sections(freq)
    lev_sec = {n: st.section(n) for n in m.layer_names}
    lev_z = {n: st.level_z(n) for n in m.layer_names}
    out = {}
    for name in planes:
        kp = 0 if name == "gnd" else st.planes[name]
        zp = float(st.z[kp])
        faces = {}
        for side in ("below", "above"):
            if side == "below" and kp == 0:
                continue
            s = int(np.searchsorted(st.bounds, kp)) - (1 if side == "below" else 0)
            zb, zt, ktem = secs[s]
            levels = [n for n in m.layer_names if lev_sec[n] == s]
            Kf = np.zeros((2,) + GX.shape, complex)
            sgn = -1.0 if side == "below" else 1.0  # current toward the face = sgn * (current flowing down)
            for n in levels:  # image return current of the copper
                h = abs(lev_z[n] - zp)
                tri = np.nonzero(m.tri_layer == m.layer_names.index(n))[0]
                dep = _deposit(fg, qp[tri].reshape(-1, 2), (qw[tri][..., None] * K[tri]).reshape(-1, 2))
                k1 = solver_k(st, freq, min(zp, lev_z[n]), max(zp, lev_z[n]))
                Kf -= _coarsen(_image(dep / fg.dx**2, fg.dx, h, k1, nr))[:, sl[0], sl[1]]
            # via segments in this section that touch the face
            lo_s, hi_s = np.maximum(slo, zb), np.minimum(shi, zt)
            inside = hi_s - lo_s > 1e-12 * max(zt - zb, 1e-9)
            touch = inside & ((np.isclose(lo_s, zp) if side == "above" else np.isclose(hi_s, zp)))
            closed_empty = not levels and ktem is not None
            for q in np.nonzero(touch)[0]:
                i_in = sgn * Iv[q]
                dx, dy = GX - sxy[q, 0], GY - sxy[q, 1]
                rho = np.maximum(np.hypot(dx, dy), sr[q])
                if closed_empty:
                    kr = -1j * ktem / 4 * hankel2(1, ktem * rho) * i_in
                else:
                    hseg = hi_s[q] - lo_s[q]
                    kr = i_in / (2 * np.pi * rho) * hseg / np.hypot(rho, hseg)
                Kf += kr * np.array([dx, dy]) / rho
            faces[side] = Kf
        out[name] = faces if per_face else sum(faces.values())
    return out


# ------------------------------------------------------------------ E field
class _SectionKernels:
    """Scalar-potential kernels (and their rho / z derivatives) from the
    metal levels of one stack section to observation heights zo (absolute)."""

    def __init__(self, lg, z_off, zo):
        self.lg = lg
        k0 = lg.k0
        z_int = lg.stack.z  # local interface heights 0..N
        N = len(lg.t)
        zl = np.asarray(zo, float) - z_off  # local heights
        top_open = lg.stack.top == "air"
        rho = lg.rho_tab
        nz, L = zl.size, len(lg.levels)
        Phi = np.zeros((rho.size, nz, L), complex)
        Phr = np.zeros_like(Phi)
        Phz = np.zeros_like(Phi)
        c_lead = np.array([2.0 / (lg._media(j)[0] + lg._media(j)[2]) for j in range(L)])
        dz = zl[:, None] - lg.zl[None, :]  # (nz, L)
        for kr, w, real in ((lg._kr_c, lg._w_c, False), (lg._kr_r, lg._w_r, True)):
            krc = kr.astype(complex)
            Vh, Ve, u = lg._tl(krc)  # (L, N+1, K), u (N, K)
            u0 = np.sqrt(krc**2 - k0**2)
            g = np.zeros((nz, L, kr.size), complex)
            gz = np.zeros_like(g)
            for iz, z in enumerate(zl):
                k = int(np.searchsorted(z_int, z))  # layer index (z_int[k-1] < z <= z_int[k])
                for V, store in ((Ve, 0), (Vh, 1)):
                    if k > N:  # open top
                        Vz = V[:, N] * np.exp(-u0 * (z - z_int[N]))
                        dVz = -u0 * Vz
                    else:
                        k = max(k, 1)
                        a, b = V[:, k - 1], V[:, k]
                        uk, t = u[k - 1], lg.t[k - 1]
                        s = z - z_int[k - 1]
                        # sinh(u x) / sinh(u t) in overflow-safe exponentials
                        e = np.exp(-2 * uk * t)
                        def sr(xx):  # noqa: E306
                            return np.exp(uk * (xx - t)) * (1 - np.exp(-2 * uk * xx)) / (1 - e)

                        def cr(xx):
                            return np.exp(uk * (xx - t)) * (1 + np.exp(-2 * uk * xx)) / (1 - e)
                        Vz = a * sr(t - s) + b * sr(s)
                        dVz = uk * (-a * cr(t - s) + b * cr(s))
                    if store == 0:
                        ge, gez = Vz, dVz
                    else:
                        gh, ghz = Vz, dVz
                g[iz] = (ge + k0**2 * gh) / krc**2
                gz[iz] = (gez + k0**2 * ghz) / krc**2
            ad = np.abs(dz)[..., None]
            lead = c_lead[None, :, None] * np.exp(-krc * ad) / (2 * krc)
            g -= lead
            gz -= -c_lead[None, :, None] * np.sign(dz)[..., None] * np.exp(-krc * ad) / 2
            Cw = (w * kr / (2 * np.pi))
            G = (Cw * g).reshape(-1, kr.size).T  # (K, nz*L)
            Gr = (Cw * kr * g).reshape(-1, kr.size).T
            Gz = (Cw * gz).reshape(-1, kr.size).T
            for r0 in range(0, rho.size, max(1, int(2e6 // kr.size))):
                arg = np.outer(rho[r0:r0 + max(1, int(2e6 // kr.size))], kr)
                J0 = j0(arg.real) if real else jv(0, arg)
                J1 = j1(arg.real) if real else jv(1, arg)
                sl = slice(r0, r0 + arg.shape[0])
                Phi[sl] += (J0 @ G).reshape(-1, nz, L)
                Phr[sl] -= (J1 @ Gr).reshape(-1, nz, L)
                Phz[sl] += (J0 @ Gz).reshape(-1, nz, L)
        R = np.sqrt(rho[:, None, None] ** 2 + dz[None] ** 2)
        Phi += c_lead / (4 * np.pi * R)
        Phr += -c_lead * rho[:, None, None] / (4 * np.pi * R**3)
        Phz += -c_lead * dz[None] / (4 * np.pi * R**3)
        self.rho = rho
        A = np.stack([Phi, Phr, Phz], -1)  # (rho, nz, L, 3)
        self.spl = [CubicSpline(rho, A[:, i].reshape(rho.size, -1)) for i in range(nz)]
        self.nz, self.L = nz, L

    def __call__(self, i, r):
        """(..., L, 3) [Phi, dPhi/drho, dPhi/dz] for observation height i at
        rho r (clipped to the table)."""
        return self.spl[i](np.minimum(r, self.rho[-1])).reshape(r.shape + (self.L, 3))


def efield(solver, x, freq, grid, z, greens=None):
    """Complex E (3, nz, ny, nx) [V/m] at heights z (solver coordinates,
    inside dielectric or air, not on a plane) on `grid`. See the module notes."""
    m, st = solver.mesh, solver.stack
    g = greens or solver.greens(freq)
    if not hasattr(g, "_sub") and not hasattr(g, "_tl"):  # single slab: tabulate the layered form
        from .layered import make_greens

        g = make_greens(freq, st, solver.rho_max, levels=m.layer_names)
    z = np.atleast_1d(np.asarray(z, float))
    fg, sl = _extended(grid, m)
    E = np.zeros((3, z.size, grid.ny, grid.nx), complex)
    q = solver.charges(x[:, None], g.omega)[:, 0]  # surface charge density per triangle
    qp = quad_points(m.verts, TRI7_BARY)
    qw = TRI7_W[None, :] * m.area[:, None]
    ox, oy = _offsets(fg)
    rr = np.hypot(ox, oy)
    cosx, cosy = np.where(rr > 0, ox / np.where(rr > 0, rr, 1), 0), np.where(rr > 0, oy / np.where(rr > 0, rr, 1), 0)
    sec_of = np.array([int(np.searchsorted(st.bounds, 0)) for _ in z])
    zint = st.z
    for i, zi in enumerate(z):
        k = int(np.searchsorted(zint, zi))
        sec_of[i] = min(max(int(np.searchsorted(st.bounds, k)) - 1, 0), len(st.bounds) - 2)
        if np.any(np.isclose(zi, zint[st.bounds])) or zi <= 0:
            raise ValueError(f"efield: z = {zi * 1e3:.4g} mm lies on a plane / the ground")
    lev_sec = {n: st.section(n) for n in m.layer_names}
    secs = st.via_sections(freq)
    sxy, sr, slo, shi = _segments(solver)
    Iv = x[m.n_basis:m.n_basis + m.n_vias]
    GX, GY = np.meshgrid(grid.xc, grid.yc)
    for s in sorted(set(sec_of)):
        iz = np.nonzero(sec_of == s)[0]
        levels = [n for n in m.layer_names if lev_sec[n] == s]
        if levels:
            lg = g._sub[s] if hasattr(g, "_sub") else g
            ker = _SectionKernels(lg, float(zint[st.bounds[s]]), z[iz])
            deps = []
            for n in lg.levels:
                tri = np.nonzero(m.tri_layer == m.layer_names.index(n))[0]
                deps.append(_deposit(fg, qp[tri].reshape(-1, 2), (q[tri][:, None] * qw[tri]).reshape(-1, 1))[0])
            for a, zz in enumerate(iz):
                K = ker(a, rr)  # (ky, kx, L, 3)
                for jl, dep in enumerate(deps):
                    if not np.any(dep):
                        continue
                    Kr, Kz = K[:, :, jl, 1], K[:, :, jl, 2]
                    E[0, zz] -= _coarsen(_cconv(dep, Kr * cosx))[sl] / EPS0
                    E[1, zz] -= _coarsen(_cconv(dep, Kr * cosy))[sl] / EPS0
                    E[2, zz] -= _coarsen(_cconv(dep, Kz))[sl] / EPS0
        else:
            zb, zt, ktem = secs[s]
            if ktem is None:
                continue
            span = (slo <= zb + 1e-12) & (shi >= zt - 1e-12)
            for qv in np.nonzero(span)[0]:
                rho = np.maximum(np.hypot(GX - sxy[qv, 0], GY - sxy[qv, 1]), sr[qv])
                Ez = -(g.omega * MU0 / 4) * hankel2(0, ktem * rho) * (-Iv[qv])  # upward current
                E[2, iz] += Ez[None]
    return E
