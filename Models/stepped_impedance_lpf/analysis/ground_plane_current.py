"""Return current on the ground plane of the stepped-impedance LPF.

The ground is a PEC at z = 0 and mu = mu0 everywhere, so in the quasi-static
limit (k1 h << 1; here k1 h ~ 0.18 at 9 GHz) the tangential H at z = 0+ is
twice the Biot-Savart field of the trace current K (image doubling), and
J_gnd = z x H gives

    J_gnd(r) = -(h / 2 pi) * Int K(r') / (|r - r'|^2 + h^2)^(3/2) dA'

The kernel integrates to 1: the total return current equals the trace current,
spread laterally over a width ~ h. K is evaluated from the RWG coefficients at
7 quadrature points per triangle (it is linear inside each triangle).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")

from lpf import H as h, layout, length, substrate  # noqa: E402
from mom25d import GHz, Solver, mm  # noqa: E402
from mom25d.integrals import TRI7_BARY, TRI7_W, quad_points  # noqa: E402

x_end = length()
mesh = layout().mesh(max_cell=0.6 * mm, edge_cell=0.06 * mm)

sol = Solver(mesh, substrate())
freqs = np.array([1.5, 6.5]) * GHz
res = sol.sweep(freqs, keep_currents=True)
S = res.renormalize(50)

# RWG current density at quadrature points: K(r) = sum_i cA_i x_i (r - v_i)
b = mesh.tri_basis
lab = np.where(b >= 0, mesh.edge_len[np.maximum(b, 0)], 0.0)
cA = mesh.tri_sign * lab / (2 * mesh.area[:, None])  # (Nt,3)
qp = quad_points(mesh.verts, TRI7_BARY)  # (Nt,7,2)
qw = (TRI7_W[None, :] * mesh.area[:, None]).ravel()  # (Nt*7,)
rv = qp[:, :, None, :] - mesh.verts[:, None, :, :]  # (Nt,7,3,2)


def trace_K(xrwg):
    xb = np.where(b >= 0, xrwg[np.maximum(b, 0)], 0)  # (Nt,3)
    return np.einsum("ti,tkid,ti->tkd", cA, rv, xb).reshape(-1, 2)  # (Nt*7,2)


# observation grid on the ground plane (physical region: filter + part of feeds)
xo = np.arange(-6 * mm, x_end + 6 * mm, 0.08 * mm)
yo = np.arange(-4 * mm, 4 * mm + 1e-9, 0.04 * mm)
XO, YO = np.meshgrid(xo, yo)
obs = np.c_[XO.ravel(), YO.ravel()]
src = qp.reshape(-1, 2)


def ground_J(K, chunk=400):
    Kw = K * qw[:, None]
    out = np.empty((len(obs), 2), complex)
    for i in range(0, len(obs), chunk):
        d = obs[i:i + chunk, None, :] - src[None, :, :]
        ker = h / (2 * np.pi) / (d[..., 0] ** 2 + d[..., 1] ** 2 + h * h) ** 1.5
        out[i:i + chunk] = -ker @ Kw
    return out.reshape(*XO.shape, 2)


tri = Triangulation(mesh.nodes[:, 0] / mm, mesh.nodes[:, 1] / mm, mesh.tris)
wA = np.repeat(mesh.area, 3)


def nodal(v):
    num = np.bincount(mesh.tris.ravel(), weights=np.repeat(v, 3) * wA, minlength=len(mesh.nodes))
    den = np.bincount(mesh.tris.ravel(), weights=wA, minlength=len(mesh.nodes))
    return num / den


fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
axes = axes.reshape(2, 2)
for fi, f in enumerate(freqs):
    xm = res.matched_currents(fi, 0)
    Kc = sol.current_density(xm[:, None])[:, :, 0]
    kmag = np.linalg.norm(np.abs(Kc), axis=1)
    Jg = ground_J(trace_K(xm))
    gmag = np.sqrt(np.sum(np.abs(Jg) ** 2, axis=-1))
    vmax = max(kmag.max(), gmag.max())
    norm = LogNorm(vmax * 3e-3, vmax)
    s21 = 20 * np.log10(abs(S[fi, 1, 0]))

    # check: x-current averaged over a 2 mm slab in the middle of the long
    # narrow section; the ground carries minus the trace current
    x0, x1 = x_end / 2 - 1 * mm, x_end / 2 + 1 * mm
    slab = (src[:, 0] > x0) & (src[:, 0] < x1)
    It = np.sum(trace_K(xm)[slab, 0] * qw[slab]) / (x1 - x0)
    cols = (xo > x0) & (xo < x1)
    Ig = np.trapezoid(Jg[:, cols, 0], yo, axis=0).mean()
    print(f"{f / GHz:.2f} GHz: slab-averaged I_trace = {abs(It):.4f} A, "
          f"I_ground = {abs(Ig):.4f} A, I_ground/I_trace = {Ig / It:.3f} "
          f"(expected -{2 / np.pi * np.arctan(yo[-1] / h):.3f}: the 1/y^2 tail beyond |y| = "
          f"{yo[-1] / mm:.0f} mm is outside the window)")

    ax = axes[fi, 0]
    pc = ax.tripcolor(tri, nodal(kmag), shading="gouraud", cmap="inferno", norm=norm)
    ax.set_title(f"Trace |K| at {f / GHz:.2f} GHz (S21 = {s21:.1f} dB)")
    ax = axes[fi, 1]
    ax.pcolormesh(XO / mm, YO / mm, gmag, shading="gouraud", cmap="inferno", norm=norm)
    sl = (slice(None, None, 12), slice(None, None, 25))
    Jr = Jg[sl].real
    n = np.linalg.norm(Jr, axis=-1) + 1e-30
    keep = np.sqrt(np.sum(np.abs(Jg[sl]) ** 2, axis=-1)) > vmax * 0.02
    ax.quiver((XO / mm)[sl][keep], (YO / mm)[sl][keep], (Jr[..., 0] / n)[keep],
              (Jr[..., 1] / n)[keep], color="cyan", scale=60, width=0.0015, alpha=0.8)
    ax.set_title(f"Ground-plane |J| at {f / GHz:.2f} GHz (arrows: Re J direction)")
    for a in axes[fi]:
        a.set_facecolor("#1a1a2e")
        a.set_aspect("equal")
        a.set(xlim=(xo[0] / mm, xo[-1] / mm), ylim=(-4, 4), xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(pc, ax=axes[fi], label="|J|  [A/m per 1 A incident]", shrink=0.9)
fig.suptitle("Stepped-impedance LPF: trace current (left) and quasi-static ground return "
             "current (right). Wave incident on port 1, port 2 matched.", fontsize=10)
fig.savefig(os.path.join(RESULTS, "ground_plane_current.png"), dpi=130)
print("saved ground_plane_current.png, N =", mesh.n_basis)
