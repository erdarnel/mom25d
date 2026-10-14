"""Port options for feeds whose line voltage is not valid (Port z0_from,
Solver v_weight), and the scipy struve nan repair in the antipad kernels."""

import types

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

from mom25d import Dielectric, GHz, Layout, Solver, Stackup, mm
from mom25d.antipad import _int_j0
from mom25d.mesh import Port
from mom25d.solver import VOLTAGE_WEIGHT


def test_struve_nan_repaired():
    # scipy 1.15: struve(0, 25.7653728) is nan (a zero of H0)
    x = np.array([25.7653728, 19.3, 34.4])
    v = _int_j0(x)
    assert np.isfinite(v).all()
    ref = [quad(j0, 0, xi, limit=200)[0] for xi in x]
    assert np.allclose(v, ref, atol=1e-7)


def test_reciprocal_z0_ignores_bad_voltage():
    """Two port groups, true Z0 50 / 70 ohm; the second port's voltage
    estimate is wrong (20 ohm). With z0_from='reciprocity' its Z0 comes from
    reciprocity alone and the level from the first port."""
    z = np.array([50.0, 70.0])
    S = np.array([[0.2, 0.8j], [0.8j, -0.1]])  # reciprocal power-wave S
    D = np.sqrt(z)
    SI = S * D[None, :] / D[:, None]  # S = D SI D^-1
    zphi = np.array([50.0, 20.0])
    ports = [Port("a", (0, 0), "-x", 1e-3, 1e-2), Port("b", (0, 0), "+x", 1e-3, 1e-2, z0_from="reciprocity")]
    ns = types.SimpleNamespace(groups={"a": [0], "b": [1]}, ports=ports, _voltage_z0=Solver._voltage_z0)
    assert np.allclose(Solver._reciprocal_z0(ns, SI, zphi), z, rtol=1e-6)
    ports[1].z0_from = "voltage"  # default behaviour: the bad estimate pulls both
    assert not np.allclose(Solver._reciprocal_z0(ns, SI, zphi), z, rtol=1e-2)
    ports[0].z0_from = ports[1].z0_from = "reciprocity"
    with pytest.raises(ValueError):
        Solver._reciprocal_z0(ns, SI, zphi)
    with pytest.raises(ValueError):
        Port("c", (0, 0), "+x", 1e-3, 1e-2, z0_from="x")


def test_z0_recip_min_drops_weak_couplings():
    """Three ports: 1-3 strongly coupled, 2 weakly coupled to both with a
    non-reciprocal (x2) isolation term. Port 3's voltage is invalid. With
    z0_recip_min = 0.1 the weak terms are ignored: ports 1, 2 keep their line
    voltage Z0 and port 3 gets its Z0 from port 1."""
    z = np.array([55.0, 79.0, 54.5])
    S = np.array([[0.1, 0.02, 0.9j], [0.01, 0.3, 0.015], [0.9j, 0.015, 0.1]])
    D = np.sqrt(z)
    SI = S * D[None, :] / D[:, None]
    zphi = np.array([55.0, 79.0, 30.0])
    ports = [Port(n, (0, 0), "-x", 1e-3, 1e-2) for n in "abc"]
    ports[2].z0_from = "reciprocity"
    ns = types.SimpleNamespace(groups={"a": [0], "b": [1], "c": [2]}, ports=ports,
                               _voltage_z0=Solver._voltage_z0, z0_recip_min=0.1)
    assert np.allclose(Solver._reciprocal_z0(ns, SI, zphi), z, rtol=1e-9)
    ns.z0_recip_min = None  # every coupling: the x2 non-reciprocity moves port 2
    assert abs(Solver._reciprocal_z0(ns, SI, zphi)[1] / 79.0 - 1) > 0.1


def test_current_only_fit_on_inverted_line():
    """A strip under a plane with 2 mm of air and a dummy ground below it is
    the mirror image of the open microstrip on top: with v_weight = 0 the
    fitted eps_eff of both agree (the voltage-based fit of the inverted line
    does not)."""
    W, L, h = 0.37 * mm, 12 * mm, 0.185 * mm
    st = Stackup([Dielectric(2 * mm, 1.0), Dielectric(h, 3.54), Dielectric(h, 3.54)],
                 {"low": 1, "top": 3}, planes={"P": 2})

    def line(layer):
        lay = Layout()
        lay.add_plane("P")
        lay.add_rect(-L / 2, -W / 2, L / 2, W / 2, layer=layer)
        lay.add_port("P1", (-L / 2, 0), "-x", W, 6 * mm, layer=layer, margin=1 * mm)
        lay.add_port("P2", (L / 2, 0), "+x", W, 6 * mm, layer=layer, margin=1 * mm)
        return lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm, kind="tri")

    f = np.array([30.0]) * GHz
    top = Solver(line("top"), st, verbose=False).sweep(f, interp=None)
    low0 = Solver(line("low"), st, verbose=False, v_weight=0.0).sweep(f, interp=None)
    low1 = Solver(line("low"), st, verbose=False).sweep(f, interp=None)
    assert VOLTAGE_WEIGHT == 1.0
    assert low0.eeff[0, 0] == pytest.approx(top.eeff[0, 0], rel=5e-3)
    assert abs(low1.eeff[0, 0] / top.eeff[0, 0] - 1) > abs(low0.eeff[0, 0] / top.eeff[0, 0] - 1)
