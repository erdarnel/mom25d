"""Derived fields (fields.py) and the HTML viewers (view3d.py)."""

import numpy as np
import pytest

from mom25d import Dielectric, LayeredGreens, Layout, Solver, Stackup, mm, view3d
from mom25d.fields import Grid, _SectionKernels, efield, plane_currents

H, ER, W, LINE, FEED = 0.508, 3.55, 1.1, 10.0, 12.0  # [mm]


def test_kernels_vs_images():
    """Charge at 0.5 mm in air over PEC (two eps=1 layers): Phi, dPhi/drho,
    dPhi/dz at heights below / above the source = the image solution."""
    st = Stackup([Dielectric(0.3 * mm, 1.0), Dielectric(0.2 * mm, 1.0)], {"top": 2})
    lg = LayeredGreens(1e8, st, rho_max=0.02)
    zo = np.array([0.1, 0.45, 0.55, 0.9]) * mm
    K = _SectionKernels(lg, 0.0, zo)
    r = np.array([0.05, 0.3, 1.0, 3.0]) * mm
    h = 0.5 * mm
    for i, z in enumerate(zo):
        v = K(i, r)[:, 0, :].real
        R1, R2 = np.hypot(r, z - h), np.hypot(r, z + h)
        ref = np.stack([1 / R1 - 1 / R2, -r / R1**3 + r / R2**3, -(z - h) / R1**3 + (z + h) / R2**3], 1) / (4 * np.pi)
        assert np.allclose(v, ref, rtol=2e-3)


def test_kernels_interface_conditions():
    """phi and D_z continuous across a dielectric interface without charge."""
    st = Stackup([Dielectric(0.3 * mm, 4.0), Dielectric(0.2 * mm, 2.0)], {"top": 2})
    lg = LayeredGreens(1e8, st, rho_max=0.02)
    d = 1e-7
    K = _SectionKernels(lg, 0.0, np.array([0.3 * mm - d, 0.3 * mm + d]))
    r = np.array([0.5, 2.0]) * mm
    a, b = K(0, r)[:, 0, :], K(1, r)[:, 0, :]
    assert np.allclose(a[:, 0], b[:, 0], rtol=3e-3)
    assert np.allclose(4 * a[:, 2], 2 * b[:, 2], rtol=3e-3)


def _line():
    lay = Layout()
    lay.add_rect(0, -W / 2 * mm, LINE * mm, W / 2 * mm)
    lay.add_port("P1", (0, 0), "-x", W * mm, FEED * mm)
    lay.add_port("P2", (LINE * mm, 0), "+x", W * mm, FEED * mm)
    st = Stackup([Dielectric(H * mm, ER)], {"top": 1}, planes=None)
    m = lay.mesh(0.4 * mm, edge_cell=0.1 * mm)
    return Solver(m, st, verbose=False)


@pytest.fixture(scope="module")
def line():
    sol = _line()
    f = 5e9
    x = sol.solve_currents(f)[:, 0]
    return sol, f, x


def test_efield_integrates_to_line_voltage(line):
    """-Int E_z dz from the ground to the strip = the line voltage (gW)."""
    sol, f, x = line
    g = sol.greens(f)
    grid = Grid(4.0 * mm, -0.05 * mm, 0.1 * mm, 2, 1)  # cells at x = 4.05 / 4.15 mm, y = 0
    z = np.linspace(0.02, H - 0.02, 25) * mm
    E = efield(sol, x, f, grid, z, greens=g)
    vint = -np.trapezoid(E[2, :, 0, :], z, axis=0).mean()
    vint += -E[2, 0, 0, :].mean() * 0.02 * mm - E[2, -1, 0, :].mean() * 0.02 * mm  # ends
    q = sol.charges(x[:, None], g.omega)
    vl = sol.line_voltage(g, q, np.array([[4.1 * mm, 0.0]]), "top")[0, 0]
    assert abs(vint - vl) / abs(vl) < 0.08


def test_ground_return_current(line):
    """The image return current under the strip carries the strip current
    back: Int K_x dy across the ground = -I at every cross-section (plane
    'gnd', face above). The retarded image keeps it within ~3 % (window
    +-25 mm); the static Poisson kernel lost exp(-beta h) ~ 9 % here."""
    from mom25d.mesh import Port

    sol, f, x = line
    m = sol.mesh
    grid = Grid(0.0, -25 * mm, 0.1 * mm, 100, 500)
    K = plane_currents(sol, x, f, grid, planes=["gnd"], kernel_radius=30e-3)["gnd"]
    ig = K[0].sum(0) * grid.dx
    cut = Port("c", (0, 0), "+x", W * mm, LINE * mm)
    r = []
    for xc in m.xg[(m.xg > 1 * mm) & (m.xg < 9 * mm)]:
        idx, sg = m.cut_edges(cut, xc)
        r.append((-(sg * m.edge_len[idx]) @ x[idx], ig[int((xc - grid.x0) / grid.dx)]))
    r = np.array(r)
    big = np.abs(r[:, 0]) > 0.5 * np.abs(r[:, 0]).max()
    ratio = -r[big, 1] / r[big, 0]
    assert np.all(np.abs(np.abs(ratio) - 1) < 0.04)
    assert np.all(np.abs(np.angle(ratio)) < np.radians(4))  # homogeneous-medium phase error ~h sqrt(k1^2 - beta^2)


def test_viewers_write(tmp_path, line):
    sol, f, x = line
    cases = [dict(f=f, port="P1", x=x)]
    for fn, args in ((view3d.model_view, ()), (view3d.current_view, (cases,)), (view3d.field_view, (cases,))):
        kw = dict(dx=0.25 * mm, dz=0.1 * mm, air=0.3 * mm) if fn is view3d.field_view else {}
        out = fn(sol, *args, str(tmp_path / f"{fn.__name__}.html"), verbose=False, **kw) if fn is not view3d.model_view \
            else fn(sol, str(tmp_path / "model.html"))
        s = open(out).read()
        assert "__DATA__" not in s and "const D = {" in s


def test_sparams_view(tmp_path):
    f = np.linspace(1e9, 3e9, 5)
    S = np.zeros((5, 3, 3), complex) + 0.1
    out = view3d.sparams_view([dict(name="a", f=f, S=S), dict(name="b", f=f, S=0.5 * S, kind="ref")],
                              str(tmp_path / "s.html"), band=(1.5e9, 2.5e9))
    assert '"n":3' in open(out).read()
