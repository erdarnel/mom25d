"""DXF import: copper polygons, via circles and board outlines (needs ezdxf
and shapely).

Board exports (e.g. from HFSS) put each copper layer on its own DXF layer as
closed LWPOLYLINEs (arcs as bulges) and vias / pads as CIRCLEs. Traps this
module handles:
  - holes in planes are often cut as "keyholes": the hole outline joined to
    the outer outline by a zero-width slit -> an invalid polygon. make_valid +
    buffer(0) turns the slit into a proper interior ring.
  - a board outline drawn as several polygons that share an edge does not
    union into one polygon when the shared edge has different vertices (arc
    flattening): `outline` closes such seams.
  - the header unit ($INSUNITS) can be wrong (an HFSS export declared cm for
    mm coordinates): the caller gives the scale; `header_units` only reports.
Lengths returned are in metres (drawing units x `scale`).
"""

import numpy as np

INSUNITS = {0: None, 1: "inch", 2: "foot", 4: "mm", 5: "cm", 6: "m", 8: "microinch", 9: "mil", 13: "um"}


def header_units(path):
    """Unit name declared in the DXF header ($INSUNITS), or None."""
    import ezdxf

    return INSUNITS.get(ezdxf.readfile(path).header.get("$INSUNITS", 0))


def _clean(coords):
    """Valid polygon(s) from a closed ring (keyhole slits become holes)."""
    import shapely
    from shapely.geometry import Polygon

    g = shapely.make_valid(Polygon(coords)).buffer(0)
    return g


def read_layers(path, scale, layers=None, flatten=None):
    """{layer: dict(polygons=[shapely Polygon / MultiPolygon], circles=(n, 3)
    array of x, y, r)} for the modelspace of a DXF file, in metres.

    scale   : metres per drawing unit (1e-3 for mm)
    layers  : layer names to read (default: all)
    flatten : max deviation when arcs (bulges) are flattened, in drawing
              units (default 2 um)
    """
    import shapely.affinity
    from ezdxf import path as dxfpath
    import ezdxf

    flatten = flatten if flatten is not None else 2e-6 / scale
    msp = ezdxf.readfile(path).modelspace()
    out = {}
    for e in msp:
        lay = e.dxf.layer
        if layers is not None and lay not in layers:
            continue
        d = out.setdefault(lay, dict(polygons=[], circles=[]))
        kind = e.dxftype()
        if kind == "CIRCLE":
            c = e.dxf.center
            d["circles"].append((c.x * scale, c.y * scale, e.dxf.radius * scale))
        elif kind == "LWPOLYLINE" and e.closed or kind == "POLYLINE" and e.is_closed:
            v = [(q.x, q.y) for q in dxfpath.make_path(e).flattening(flatten)]
            if len(v) < 3:
                continue
            g = _clean(v)
            if g.area > 0:
                d["polygons"].append(shapely.affinity.scale(g, scale, scale, origin=(0, 0)))
    for d in out.values():
        d["circles"] = np.array(d["circles"], float).reshape(-1, 3)
    return out


def outline(polygons, seam=1e-5):
    """Union of outline polygons with seams narrower than `seam` [m] closed
    (mitred close: corners stay sharp)."""
    import shapely

    u = shapely.union_all(polygons)
    return u.buffer(seam, join_style="mitre").buffer(-seam, join_style="mitre")


def split_by_vias(polygons, via_xy):
    """(signal, ground) unions: polygons containing a via centre are ground
    (stitched pours, planes), the others signal copper."""
    import shapely
    from shapely.geometry import Polygon

    via_xy = np.asarray(via_xy, float).reshape(-1, 2)
    sig, gnd = [], []
    for g in polygons:
        hit = via_xy.size and shapely.contains_xy(g, via_xy[:, 0], via_xy[:, 1]).any()
        (gnd if hit else sig).append(g)
    empty = Polygon()
    return (shapely.union_all(sig) if sig else empty), (shapely.union_all(gnd) if gnd else empty)
