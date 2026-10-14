"""Plane apertures (aperture.py): kernels, dense == H-matrix, and a GCPW over a slotted
inner plane (air-backed, the slot carries the field) against a 2-D quasi-static FD
solution of the same cross-section (via walls from L1 to the plane)."""
import numpy as np
from shapely.geometry import box

from mom25d import CoplanarGround, Dielectric, GHz, Layout, Solver, Stackup, mm
from mom25d.aperture import ApertureKernels


def test_current_over_plane_is_twice_biot_savart():
    d = 0.1e-3
    st = Stackup([Dielectric(5e-3, 1.0), Dielectric(d, 1.0), Dielectric(5e-3, 1.0)], metals={"L1": 2},
                 top="air", planes={"L2": 1})
    K = ApertureKernels(1e9, st, "L2", "L2", {"up": ["L1"], "dn": []}, 0.02)
    rho = np.array([0.0, 0.05e-3, 0.1e-3, 0.3e-3, 1e-3])
    ha, hs = K.sides["up"]["kern"]["L1"](rho)
    ref = -d / (2 * np.pi * (rho**2 + d**2) ** 1.5)  # x-current at height d: H_y = 2 * Biot-Savart
    assert np.allclose(ha.real, ref, rtol=1e-3)
    assert np.abs(hs).max() < 1e-6 * np.abs(ref).max()  # homogeneous: no TE/TM split


def _gcpw(a, airback=True, L=2.0, feed=8.0):
    W, GAP, YG, P = 0.30, 0.25, 1.30, 0.40
    s = mm
    st = Stackup([Dielectric((2.0 if airback else 0.079) * s, 1.0 if airback else 3.09), Dielectric(0.076 * s, 3.11)],
                 metals={"L1": 2}, top="air", planes={"L2": 1})
    lay = Layout()
    lay.add_rect(-L / 2 * s, -W / 2 * s, L / 2 * s, W / 2 * s, layer="L1")
    vb, con = ("L2", ()) if airback else ("gnd", ("L2",))
    for sy in (-1, 1):
        y0, y1 = sorted((sy * (W / 2 + GAP), sy * YG))
        lay.add_rect(-L / 2 * s, y0 * s, L / 2 * s, y1 * s, layer="L1")
        for y, x0 in ((0.55, -L / 2 + 0.2), (0.90, -L / 2 + 0.4)):
            x = x0
            while x <= L / 2 - 0.15 + 1e-9:
                lay.add_via(x * s, sy * y * s, drill=0.15 * s, pad=0.3 * s, top="L1", bottom=vb, connect=con)
                x += P
    g = CoplanarGround(gap=GAP * s, width=(YG - W / 2 - GAP) * s, via_pitch=P * s, via_drill=0.15 * s,
                       via_pad=0.3 * s, via_to=vb, via_connect=con)
    lay.add_port("P1", (-L / 2 * s, 0), "-x", W * s, feed * s, layer="L1", ground=g)
    lay.add_port("P2", (L / 2 * s, 0), "+x", W * s, feed * s, layer="L1", ground=g)
    lay.add_gap_port("C", (0, 0.05 * s), (0, 0.50 * s), 0.1 * s, layer="L1")  # anchors Z0 (reciprocity)
    X = L / 2 + feed + 1.0
    lay.add_aperture("L2", box(-X * s, -a * s, X * s, a * s))
    return lay, st


def test_gcpw_over_slotted_plane_vs_2d_fd():
    # 2-D FD (quasi-static, same cross-section; CLAUDE.md decision 33): solid plane 35.61 ohm, slot half-width 0.2 mm 71.6 ohm.
    # The MoM sits ~1 ohm above FD for the solid plane (via rows vs a via wall), same here.
    lay, st = _gcpw(0.2)
    m = lay.mesh(0.3 * mm, edge_cell=0.04 * mm, kind="tri")
    r = Solver(m, st, verbose=False, method="dense").sweep([20 * GHz])
    assert abs(r.z0[0, 0] - 71.6) < 2.5
    assert r.fit_residual[0].max() < 5e-3


def test_dense_equals_hmatrix():
    lay, st = _gcpw(0.3, airback=False, L=1.0, feed=4.0)
    m = lay.mesh(0.4 * mm, edge_cell=0.06 * mm, kind="tri")
    rd = Solver(m, st, verbose=False, method="dense").sweep([20 * GHz])
    rh = Solver(m, st, verbose=False, method="hmatrix").sweep([20 * GHz])
    assert np.abs(rd.renormalize() - rh.renormalize()).max() < 1e-3


def test_view_planes_show_apertures():
    """view3d draws plane apertures as openings (display mesh area = outline minus aperture)."""
    from shapely.geometry import box as sbox
    from mom25d.view3d import _tri_poly
    import numpy as np

    V, T = _tri_poly(sbox(0, 0, 10, 4), cut=sbox(2, 1.5, 8, 2.5))
    a = V[T]
    area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                        - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])).sum()
    assert abs(area - (40 - 6)) < 1e-6
