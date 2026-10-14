"""ODB++ import on a tiny synthetic export (inch units, Altium-style)."""

import os

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")

from mom25d import GHz, Solver, Substrate, mm  # noqa: E402
from mom25d.odb import ODB, find_ports, layout_from_copper  # noqa: E402

IN = 25.4


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


@pytest.fixture
def odb_dir(tmp_path):
    """top_layer with
      feature 0: RF trace, 20 mil wide, x = 0..1 inch at y = 0.5 (net RF)
      feature 1: GND pour 0..1 x 0..1 inch whose clearance hole around the
                 trace is listed *after* a second, unrelated island (net GND)
      feature 2: stray pad (net OTHER)
    """
    root = str(tmp_path / "odb")
    feats = """#
$0 r20
$1 r40
#
L 0 0.5 1.0 0.5 0 P 0
S P 0
OB 0 0 I
OS 1 0
OS 1 1
OS 0 1
OS 0 0
OE
OB 2 2 I
OS 2.1 2
OS 2.1 2.1
OS 2 2.1
OS 2 2
OE
OB 0 0.45 H
OS 1 0.45
OS 1 0.55
OS 0 0.55
OS 0 0.45
OE
SE
P 0.5 0.8 1 P 0 0
"""
    _write(root, "steps/pcb/layers/top_layer/features", feats)
    eda = """HDR test
LYR drill top_layer
NET RF
SNT TOP T 0 0
FID C 1 0
NET GND
SNT PLN S C 0
FID C 1 1
NET OTHER
SNT TOP T 0 1
FID C 1 2
"""
    _write(root, "steps/pcb/eda/data", eda)
    return root


def test_parse_nets_and_surface_holes(odb_dir):
    odb = ODB(odb_dir)
    feats = odb.layer_features("top_layer")
    assert len(feats) == 3
    rf = odb.copper("top_layer", nets=["RF"])
    assert rf.area == pytest.approx(IN * 0.02 * IN * 1.0 + np.pi * (0.01 * IN) ** 2, rel=0.02)
    gnd = odb.copper("top_layer", nets=["GND"])
    # the hole (0.1 inch tall band) belongs to the first island even though
    # it is listed after the second one
    assert gnd.area == pytest.approx((1.0 - 0.1 + 0.01) * IN**2, rel=1e-6)
    assert rf.distance(gnd) == pytest.approx(0.04 * IN, rel=1e-3)
    assert set(odb.nets_in_window("top_layer", (0, 30, 0, 30))) == {"RF", "GND", "OTHER"}


def test_ports_and_solve(odb_dir):
    odb = ODB(odb_dir)
    window = (5.0, 20.0, 5.0, 20.0)  # mm, cuts the RF trace on both sides
    cu = odb.copper("top_layer", window, nets=["RF"])
    ports = find_ports(cu, window)
    assert [(d, round(w, 4)) for _, d, w in ports] == [("-x", 0.508), ("+x", 0.508)]
    lay = layout_from_copper(cu, window, feed_length=6 * mm, ports=ports)
    mesh = lay.mesh(max_cell=0.6 * mm, edge_cell=0.08 * mm)
    r = Solver(mesh, Substrate(3.66, 0.254 * mm), verbose=False).sweep([5 * GHz])
    # straight 15 mm line between the reference planes
    assert abs(r.S[0, 0, 0]) < 0.01
    assert abs(r.S[0, 1, 0]) == pytest.approx(1, abs=0.01)
    bl = r.gamma[0, 0].imag * 15 * mm
    assert np.angle(r.S[0, 1, 0] * np.exp(1j * bl)) == pytest.approx(0, abs=0.01)


@pytest.fixture
def odb_stack(tmp_path):
    """Four copper layers, a blind top->layer3 via on net RF and a blind
    top->layer2 via on net GND, both at (0.5, 0.5) inch / (0.6, 0.5)."""
    root = str(tmp_path / "odb")
    rows = [("SIGNAL", "TOP_LAYER", "", ""), ("SIGNAL", "LAYER2", "", ""),
            ("SIGNAL", "LAYER3", "", ""), ("SIGNAL", "LAYER4", "", ""),
            ("DRILL", "DRILL_A", "TOP_LAYER", "LAYER3"), ("DRILL", "DRILL_B", "TOP_LAYER", "LAYER2"),
            ("DRILL", "DRILL_NON-PLATED_X", "TOP_LAYER", "LAYER4")]
    _write(root, "matrix/matrix", "".join(
        f"LAYER {{\n    ROW={i + 1}\n    TYPE={t}\n    NAME={n}\n    START_NAME={s}\n    END_NAME={e}\n}}\n\n"
        for i, (t, n, s, e) in enumerate(rows)))
    for n in ("top_layer", "layer2", "layer3", "layer4"):
        _write(root, f"steps/pcb/layers/{n}/features", "#\n$0 r20\n#\nP 0.5 0.5 0 P 0 0\n")
    _write(root, "steps/pcb/layers/drill_a/features", "#\n$0 r10\n#\nP 0.5 0.5 0 P 0 0\n")
    _write(root, "steps/pcb/layers/drill_b/features", "#\n$0 r10\n#\nP 0.6 0.5 0 P 0 0\n")
    _write(root, "steps/pcb/layers/drill_non-plated_x/features", "#\n$0 r10\n#\nP 0.7 0.5 0 P 0 0\n")
    _write(root, "steps/pcb/eda/data", "HDR test\nLYR drill_a drill_b top_layer\n"
                                        "NET RF\nFID C 0 0\nNET GND\nFID C 1 0\n")
    return root


def test_drill_spans_and_via_layers(odb_stack):
    odb = ODB(odb_stack)
    assert list(odb.signal_rows()) == ["top_layer", "layer2", "layer3", "layer4"]
    assert odb.drill_spans() == {"drill_a": ("top_layer", "layer3"), "drill_b": ("top_layer", "layer2")}
    assert odb.ground_layer("top_layer") == "layer2"
    v = {h["net"]: h for h in odb.all_vias()}
    assert v["RF"]["drill"] == pytest.approx(0.01 * IN, rel=1e-3)
    # signal via: connects the modelled layers it spans
    assert odb.via_connection(v["RF"], ["top_layer", "layer3"], "layer4", {"GND"}) == ("top_layer", "layer3")
    assert odb.via_connection(v["RF"], ["top_layer"], "layer2", {"GND"}) is None  # leads elsewhere
    # ground via: to the ground plane if it reaches it
    assert odb.via_connection(v["GND"], ["top_layer"], "layer2", {"GND"}) == ("top_layer", "gnd")
    assert odb.via_connection(v["GND"], ["top_layer", "layer3"], "layer4", {"GND"}) is None


@pytest.fixture
def odb_stackup(tmp_path):
    """Masks + four copper layers with Altium-style stackup attributes:
    .layer_dielectric (inch) = dielectric below each copper layer (the
    bottom one: the bottom mask again), mask thickness on the mask layers,
    .copper_weight (oz)."""
    root = str(tmp_path / "odb")
    rows = [("SOLDER_MASK", "TOP_SOLDER"), ("SIGNAL", "TOP_LAYER"), ("SIGNAL", "LAYER2"),
            ("SIGNAL", "LAYER3"), ("SIGNAL", "LAYER4"), ("SOLDER_MASK", "BOTTOM_SOLDER")]
    _write(root, "matrix/matrix", "".join(f"LAYER {{\n    ROW={i + 1}\n    TYPE={t}\n    NAME={n}\n}}\n\n"
                                          for i, (t, n) in enumerate(rows)))
    attrs = {"top_solder": (0.001, None, "Mask XY"), "top_layer": (0.008, 1.5, ""),
             "layer2": (0.010, 1.0, ""), "layer3": (0.012, 0.5, ""), "layer4": (0.001, 1.5, ""),
             "bottom_solder": (0.001, None, "")}
    for n, (d, w, c) in attrs.items():
        txt = f".layer_dielectric = {d}\n" + (f".copper_weight = {w}\n" if w else "") + f".comment = {c}\n"
        _write(root, f"steps/pcb/layers/{n}/attrlist", txt)
        _write(root, f"steps/pcb/layers/{n}/features", "#\n#\n")
    _write(root, "misc/attrlist", ".board_thickness = 0.04\n_board_material = laminate-x\n")
    return root


def test_stack_from_export(odb_stackup):
    from mom25d.odb import OZ_MM, build_stackup

    odb = ODB(odb_stackup)
    rows = odb.stack()
    assert [r["type"] for r in rows] == ["mask", "copper", "copper", "copper", "copper", "mask"]
    assert rows[1]["below_mm"] == pytest.approx(0.008 * IN) and rows[1]["copper_mm"] == pytest.approx(1.5 * OZ_MM)
    # top -> layer3 spans two dielectrics and layer2's copper
    assert odb.dielectric_between("top_layer", "layer3") == pytest.approx((0.008 + 0.010) * IN + 1.0 * OZ_MM)
    # one level over layer2: thickness, copper and solder mask from the export
    st, diel, rep = build_stackup(odb, ["top_layer"], "layer2", er=3.66, tand=0.003)
    assert diel == [(pytest.approx(0.008 * IN), 3.66, 0.003)]
    assert len(st.layers) == 2 and st.layers[1].t == pytest.approx(0.001 * IN * 1e-3)  # mask on top
    assert st.covered("top_layer") and st.t == pytest.approx(1.5 * OZ_MM * 1e-3)
    assert any("Mask XY" in ln for ln in rep)
    # an inner level gets no mask; a given thickness overrides the export
    st2, diel2, _ = build_stackup(odb, ["layer2"], "layer3", dielectrics=[(0.2, 3.5, None)], tand=0.01)
    assert len(st2.layers) == 1 and diel2 == [(0.2, 3.5, 0.01)]
    st3, _, _ = build_stackup(odb, ["top_layer"], "layer2", er=3.66, mask=False)
    assert len(st3.layers) == 1
    # eps_r is never in the export
    with pytest.raises(ValueError, match="laminate-x"):
        build_stackup(odb, ["top_layer"], "layer2")


@pytest.fixture
def odb_standard(tmp_path):
    """Standard ODB++ stack: DIELECTRIC layers with thickness (UNITS=MM),
    eps_r and tan d; a mask with the placeholder eps_r = 1."""
    root = str(tmp_path / "odb")
    rows = [("SOLDER_MASK", "SM_TOP"), ("SIGNAL", "S1"), ("DIELECTRIC", "D1"), ("SIGNAL", "S2"),
            ("DIELECTRIC", "D2"), ("POWER_GROUND", "P3")]
    _write(root, "matrix/matrix", "".join(f"LAYER {{\n    ROW={i + 1}\n    TYPE={t}\n    NAME={n}\n}}\n\n"
                                          for i, (t, n) in enumerate(rows)))
    attrs = {"sm_top": "UNITS=MM\n.layer_dielectric=0.02\n.dielectric_constant=1\n.loss_tangent=0.025\n",
             "s1": "UNITS=MM\n.copper_weight=1\n", "s2": "UNITS=MM\n.copper_weight=0.5\n",
             "p3": "UNITS=MM\n.copper_weight=1\n",
             "d1": "UNITS=MM\n.layer_dielectric=0.1\n.dielectric_constant=3.4\n.loss_tangent=0.02\n",
             "d2": "UNITS=MM\n.layer_dielectric=0.2\n.dielectric_constant=4.4\n.loss_tangent=0.015\n"}
    for n, txt in attrs.items():
        _write(root, f"steps/pcb/layers/{n}/attrlist", txt)
        _write(root, f"steps/pcb/layers/{n}/features", "#\n#\n")
    return root


def test_standard_stack(odb_standard):
    from mom25d.odb import OZ_MM, build_stackup

    odb = ODB(odb_standard)
    assert odb.standard_stack()
    assert odb.gap("s1", "p3") == [(pytest.approx(0.1 + 0.5 * OZ_MM), 3.4, 0.02, "d1+s2"), (0.2, 4.4, 0.015, "d2")]
    # eps_r / tan d from the export, one model layer per dielectric, mask on top
    st, diel, rep = build_stackup(odb, ["s1"], "p3")
    eps = [d.eps_r for d in st.layers]
    assert eps == [4.4, 3.4, 3.5]  # bottom first; mask eps_r 1 = placeholder -> mask_er
    assert st.layers[2].tand == 0.025 and st.layers[2].t == pytest.approx(0.02e-3)
    assert st.metals == {"s1": 2} and st.t == pytest.approx(OZ_MM * 1e-3)
    assert diel[0][1] == pytest.approx((0.1 + 0.5 * OZ_MM) * 3.4 / (0.3 + 0.5 * OZ_MM) + 0.2 * 4.4 / (0.3 + 0.5 * OZ_MM))
    # an explicit eps_r overrides the export (one homogeneous layer)
    st2, _, _ = build_stackup(odb, ["s1"], "p3", er=3.66, mask=False)
    assert [d.eps_r for d in st2.layers] == [3.66]


SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "Models", "odb_rigidflex_sample", "geometry",
                      "designodb_rigidflex.tgz")


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="ODB++ sample design not downloaded")
def test_sample_design_stack(tmp_path):
    """The public ODB++ sample (designodb_rigidflex, standard stack with
    DIELECTRIC layers): the importer reads its layers, eps_r and tan d."""
    import tarfile

    from mom25d.odb import build_stackup

    with tarfile.open(SAMPLE) as tf:
        tf.extractall(tmp_path, filter="data")
    odb = ODB(str(tmp_path / "designodb_rigidflex"))
    assert odb.standard_stack()
    st, diel, rep = build_stackup(odb, ["signal_1", "signal_3"], "plane_4")
    assert st.metals == {"signal_3": 1, "signal_1": 3}
    assert [round(d.eps_r, 2) for d in st.layers] == [4.4, 3.4, 3.4, 3.5]  # last: mask (eps_r assumed)
    assert diel[1][0] == pytest.approx(0.25, rel=1e-3)
    assert odb.dielectric_between("signal_1", "plane_4") == pytest.approx(0.065 * 2 + 2 * 0.0305 + 0.25, rel=2e-3)


@pytest.fixture
def odb_component(tmp_path):
    """Two 22 mil traces (nets RF_A, RF_B) ending in round pads 1.5 mm
    apart, bridged by a 2-pin part C1 (VALUE '1pF') on comp_+_top."""
    root = str(tmp_path / "odb")
    _write(root, "matrix/matrix", "LAYER {\n    ROW=1\n    TYPE=SIGNAL\n    NAME=TOP_LAYER\n}\n\n"
                                  "LAYER {\n    ROW=2\n    TYPE=SIGNAL\n    NAME=LAYER2\n}\n")
    feats = "#\n$0 r22\n$1 r30\n#\nL 0 0.5 0.47 0.5 0 P 0\nL 0.53 0.5 1.0 0.5 0 P 0\nP 0.47 0.5 1 P 0 0\nP 0.53 0.5 1 P 0 0\n"
    _write(root, "steps/pcb/layers/top_layer/features", feats)
    _write(root, "steps/pcb/layers/layer2/features", "#\n#\n")
    _write(root, "steps/pcb/eda/data", "HDR test\nLYR top_layer\nNET RF_A\nFID C 0 0\nFID C 0 2\n"
                                        "NET RF_B\nFID C 0 1\nFID C 0 3\n")
    _write(root, "steps/pcb/layers/comp_+_top/components",
           "UNITS=INCH\n# CMP 0\nCMP 0 0.5 0.5 0.0 N C1 CAP ;\nPRP VALUE '1pF'\nPRP TYPE 'Capacitor'\n"
           "TOP 0 0.47 0.5 0.0 N 0 0 1\nTOP 1 0.53 0.5 0.0 N 1 0 2\n"
           "# CMP 1\nCMP 0 0.9 0.9 0.0 N R9 RES ;\nPRP VALUE '10Kohms'\n"
           "TOP 0 0.88 0.9 0.0 N 0 0 1\nTOP 1 0.92 0.9 0.0 N 1 0 2\n")
    return root


def test_components_placed_and_solved(odb_component):
    """The component from the export is parsed, placed as a lumped C across
    its pads and solved: |S21| close to the ideal series 1 pF."""
    from mom25d.odb import component_value, components, layout_from_copper, place_components

    odb = ODB(odb_component)
    cs = components(odb, "top")
    assert [c["refdes"] for c in cs] == ["C1", "R9"]
    c1 = cs[0]
    assert [p["net"] for p in c1["pins"]] == ["RF_A", "RF_B"] and c1["pins"][0]["x"] == pytest.approx(0.47 * IN)
    assert repr(component_value(c1)) == "1 pF"
    window = (5.0, 20.0, 5.0, 20.0)
    cu = odb.copper("top_layer", window, nets=["RF_A", "RF_B"])
    lay = layout_from_copper(cu, window, 8 * mm)
    placed, skipped = place_components(lay, cs, cu, "top", window)
    assert [r for r, _ in placed] == ["C1"] and skipped == []  # R9 lies outside the window
    lp = lay.lumped[0]
    assert lp.length == pytest.approx(0.06 * IN * mm)
    assert lp.width == pytest.approx(0.8 * 0.030 * IN * mm, rel=1e-3)  # 80 % of the 30 mil pads
    f = np.array([2e9])
    r = Solver(lay.mesh(0.4 * mm, 0.05 * mm, kind="tri"), Substrate(3.66, 0.254 * mm), verbose=False).sweep(f)
    z0 = r.z0_line[0, 0].real
    ideal = abs(2 * z0 / (2 * z0 + 1 / (2j * np.pi * f[0] * 1e-12)))
    assert abs(r.S[0, 1, 0]) == pytest.approx(ideal, abs=0.04)
