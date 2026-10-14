import os

import numpy as np
import pytest

from mom25d import GHz, GroundedSlabGreens, Layout, Solver, Substrate, analytic, mm
from mom25d.integrals import potential_integrals

DATA = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------- kernels
@pytest.mark.parametrize("f", [1e9, 10e9, 30e9])
def test_greens_matches_images_for_air_substrate(f):
    """eps_r = 1 over ground: exact solution is the source plus a negative image."""
    h = 0.5e-3
    g = GroundedSlabGreens(f, 1.0, h, rho_max=0.06)
    k = g.k0
    R = np.array([1e-5, 1e-4, 5e-4, 2e-3, 1e-2, 0.03, 0.06])
    R2 = np.sqrt(R**2 + 4 * h**2)
    exact = (np.exp(-1j * k * R) / R - np.exp(-1j * k * R2) / R2) / (4 * np.pi)
    scale = 1 / (4 * np.pi * R)  # error relative to the direct term
    assert np.max(np.abs(g.gA(R) - exact) / scale) < 1e-6
    assert np.max(np.abs(g.gV(R) - exact) / scale) < 1e-6


def test_greens_path_independence():
    """Dielectric case: result must not depend on the integration-path lift."""
    g1 = GroundedSlabGreens(10e9, 4.4, 0.8e-3, rho_max=0.05)
    g2 = GroundedSlabGreens(10e9, 4.4, 0.8e-3, rho_max=0.02)  # different lift / panels
    R = np.array([1e-4, 1e-3, 5e-3, 1.5e-2])
    for a, b in ((g1.gA(R), g2.gA(R)), (g1.gV(R), g2.gV(R))):
        assert np.max(np.abs(a - b) * 4 * np.pi * R) < 1e-6


def test_potential_integrals_vs_brute_force():
    V = np.array([[[0, 0], [1.0, 0.2], [0.3, 0.9]]])
    r = np.array([[0.4, 0.3], [2.0, -0.5], [0.5, 0.1]])
    I0, I1 = potential_integrals(V, r)
    n = 1200
    u, v = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    m = u + v < 1
    P = V[0, 0] + u[m, None] * (V[0, 1] - V[0, 0]) + v[m, None] * (V[0, 2] - V[0, 0])
    e1, e2 = V[0, 1] - V[0, 0], V[0, 2] - V[0, 0]
    w = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0]) / m.sum()
    for k in range(len(r)):
        d = P - r[k]
        R = np.hypot(d[:, 0], d[:, 1])
        assert I0[0, k] == pytest.approx((w / R).sum(), rel=2e-3)
        np.testing.assert_allclose(I1[0, k], (w * d / R[:, None]).sum(0), atol=1e-3)


# ------------------------------------------------------------ full solver
ER, H = 3.55, 0.508 * mm


def test_through_line():
    W = analytic.width_for_z0(50, H, ER)
    lay = Layout()
    lay.add_rect(0, -W / 2, 10 * mm, W / 2)
    lay.add_port("P1", (0, 0), "-x", W, 20 * mm)
    lay.add_port("P2", (10 * mm, 0), "+x", W, 20 * mm)
    f = np.array([2, 10]) * GHz
    r = Solver(lay.mesh(0.6 * mm, 0.08 * mm), Substrate(ER, H), verbose=False).sweep(f)
    np.testing.assert_allclose(r.eeff[:, 0], analytic.eeff(W, H, ER, f), rtol=5e-3)
    np.testing.assert_allclose(r.z0_line[:, 0].real, 50, rtol=0.03)
    assert r.z0_line[1, 0].real > r.z0_line[0, 0].real  # Z0 rises with frequency
    assert np.all(np.abs(r.S[:, 0, 0]) < 5e-3)
    np.testing.assert_allclose(np.abs(r.S[:, 1, 0]), 1, atol=5e-3)
    phase = np.angle(r.S[:, 1, 0] * np.exp(1j * analytic.beta(W, H, ER, f) * 10 * mm))
    assert np.all(np.abs(phase) < np.deg2rad(0.5))


def test_width_step():
    W1 = analytic.width_for_z0(50, H, ER)
    W2 = analytic.width_for_z0(25, H, ER)
    lay = Layout()
    lay.add_port("P1", (0, 0), "-x", W1, 16 * mm)
    lay.add_port("P2", (0, 0), "+x", W2, 20 * mm)
    f = np.array([1, 6]) * GHz
    r = Solver(lay.mesh(0.5 * mm, 0.08 * mm), Substrate(ER, H), verbose=False).sweep(f)
    z1, z2 = r.z0[0]
    # low frequency: quasi-static impedance-step reflection
    assert r.S[0, 0, 0].real == pytest.approx((z2 - z1) / (z2 + z1), abs=5e-3)
    assert z2 / z1 == pytest.approx(0.5, rel=0.02)
    for S in r.S:
        assert abs(S[0, 1] - S[1, 0]) < 5e-3  # reciprocity
        assert np.abs(S.conj().T @ S - np.eye(2)).max() < 5e-3  # lossless


def test_notch_filter_vs_openems():
    """Open-stub notch filter from the openEMS tutorial, compared with FDTD."""
    ref = np.loadtxt(os.path.join(DATA, "notch_openems.txt"))
    er, h, W, Ls = 3.66, 0.254 * mm, 0.6 * mm, 12 * mm
    lay = Layout()
    lay.add_rect(-W / 2 - 1 * mm, -W / 2, W / 2 + 1 * mm, W / 2)
    lay.add_rect(-W / 2, W / 2, W / 2, W / 2 + Ls)
    lay.add_port("P1", (-W / 2 - 1 * mm, 0), "-x", W, 12 * mm)
    lay.add_port("P2", (W / 2 + 1 * mm, 0), "+x", W, 12 * mm)
    f = np.array([2.0, 3.0, 3.55, 3.6, 3.65, 3.7, 3.75, 4.5, 6.0]) * GHz
    r = Solver(lay.mesh(0.5 * mm, 0.05 * mm), Substrate(er, h), verbose=False).sweep(f)
    s21 = r.db(1, 0, r.renormalize(50))
    ref21 = 20 * np.log10(np.interp(f, ref[:, 0], ref[:, 2]))
    band = np.abs(ref21) < 15  # away from the steep null
    assert np.max(np.abs(s21[band] - ref21[band])) < 0.3
    # notch frequency (parabola through the three lowest points)
    i = np.argmin(s21)
    p = np.polyfit(f[i - 1:i + 2] / GHz, s21[i - 1:i + 2], 2)
    assert -p[1] / (2 * p[0]) == pytest.approx(3.671, rel=3e-3)


def test_ground_plane_loss_matches_direct_integral():
    """The P_2h term in gA must dissipate exactly (1/2) Re(Zs) Int |J_gnd|^2,
    with J_gnd = -P_h * K the quasi-static ground return current."""
    from mom25d import assemble
    from mom25d.integrals import TRI7_BARY, TRI7_W, quad_points

    W = analytic.width_for_z0(50, H, ER)
    lay = Layout()
    lay.add_rect(0, -W / 2, 4 * mm, W / 2)
    lay.add_port("P1", (0, 0), "-x", W, 8 * mm)
    lay.add_port("P2", (4 * mm, 0), "+x", W, 8 * mm)
    mesh = lay.mesh(0.6 * mm, 0.08 * mm)
    f = 5 * GHz
    on = Solver(mesh, Substrate(ER, H, sigma=5.8e7), verbose=False)
    off = Solver(mesh, Substrate(ER, H, sigma=5.8e7, ground_loss=False), verbose=False)
    zs = on.sub.surface_impedance(f)
    Zg = assemble(mesh, on.greens(f), zs) - assemble(mesh, off.greens(f), zs)
    x = off.solve_currents(f)[:, 0]
    p_z = 0.5 * np.real(np.conj(x) @ Zg @ x)

    b = mesh.tri_basis
    lab = np.where(b >= 0, mesh.edge_len[np.maximum(b, 0)], 0.0)
    cA = mesh.tri_sign * lab / (2 * mesh.area[:, None])
    qp = quad_points(mesh.verts, TRI7_BARY)
    xb = np.where(b >= 0, x[np.maximum(b, 0)], 0)
    K = np.einsum("ti,tkid,ti->tkd", cA, qp[:, :, None, :] - mesh.verts[:, None], xb).reshape(-1, 2)
    Kw = K * (TRI7_W[None] * mesh.area[:, None]).reshape(-1, 1)
    src = qp.reshape(-1, 2)
    dx = 0.08 * mm
    X, Y = np.meshgrid(np.arange(-14 * mm, 18 * mm, dx), np.arange(-8 * mm, 8 * mm, dx))
    obs = np.c_[X.ravel(), Y.ravel()]
    Jg = np.empty((len(obs), 2), complex)
    for i in range(0, len(obs), 4000):
        d = obs[i:i + 4000, None] - src[None]
        Jg[i:i + 4000] = -(H / (2 * np.pi) / (d[..., 0] ** 2 + d[..., 1] ** 2 + H * H) ** 1.5) @ Kw
    p_direct = 0.5 * zs.real * np.sum(np.abs(Jg) ** 2) * dx * dx
    assert p_z == pytest.approx(p_direct, rel=2e-3)
