"""DXF import on a small synthetic drawing (mm, header declaring cm as in a
real HFSS export)."""

import numpy as np
import pytest

ezdxf = pytest.importorskip("ezdxf")
pytest.importorskip("shapely")

from mom25d.dxf import header_units, outline, read_layers, split_by_vias  # noqa: E402

MM = 1e-3


@pytest.fixture
def dxf_path(tmp_path):
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 5  # cm, but the coordinates are mm
    msp = doc.modelspace()
    # trace 10 x 0.4 mm with a round end (bulge 1 = half circle of the 0.4 mm chord)
    msp.add_lwpolyline([(0, -0.2, 0), (10, -0.2, 1), (10, 0.2, 0), (0, 0.2, 0)],
                       format="xyb", close=True, dxfattribs={"layer": "L1"})
    # pour 0..10 x 1..3 mm with a 0.5 mm keyhole hole around (5, 2): the
    # hole is joined to the outer ring by a zero-width slit at x = 5
    hole = [(5 + 0.5 * np.cos(a), 2 + 0.5 * np.sin(a)) for a in np.linspace(-np.pi / 2, 1.5 * np.pi, 33)]
    ring = [(0, 1), (5, 1), *hole, (5, 1), (10, 1), (10, 3), (0, 3)]
    msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": "L1"})
    msp.add_circle((2, 2), 0.15, dxfattribs={"layer": "vias"})
    # board outline in two halves along x = 5 that miss each other by 1 nm
    # (as arc-flattened exports do): their plain union is two polygons
    e = 1e-6
    msp.add_lwpolyline([(-1, -1), (5, -1), (5, 4), (-1, 4)], close=True, dxfattribs={"layer": "sub"})
    msp.add_lwpolyline([(5 + e, -1), (11, -1), (11, 4), (5 + e, 4), (5 + e, 1.5)], close=True,
                       dxfattribs={"layer": "sub"})
    p = str(tmp_path / "board.dxf")
    doc.saveas(p)
    return p


def test_read_and_classify(dxf_path):
    assert header_units(dxf_path) == "cm"
    d = read_layers(dxf_path, MM)
    assert set(d) == {"L1", "vias", "sub"}
    vias = d["vias"]["circles"]
    assert vias.shape == (1, 3) and np.allclose(vias[0], [2 * MM, 2 * MM, 0.15 * MM])
    sig, gnd = split_by_vias(d["L1"]["polygons"], vias[:, :2])
    # trace: rectangle + half disc of the bulge
    assert sig.area == pytest.approx(10 * 0.4 * MM**2 + np.pi * 0.2**2 / 2 * MM**2, rel=1e-4)
    # pour: the keyhole became a real hole (one interior ring, area removed)
    assert gnd.geom_type == "Polygon" and len(gnd.interiors) == 1
    assert gnd.area == pytest.approx((20 - np.pi * 0.25) * MM**2, rel=5e-3)


def test_outline_seam(dxf_path):
    d = read_layers(dxf_path, MM)
    import shapely

    raw = shapely.union_all(d["sub"]["polygons"])
    o = outline(d["sub"]["polygons"])
    assert raw.geom_type == "MultiPolygon" and o.geom_type == "Polygon"
    assert o.area == pytest.approx(12 * 5 * MM**2, rel=1e-9)
    assert raw.area == pytest.approx(o.area, rel=1e-6)
