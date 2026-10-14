"""ODB++ import: copper of selected nets in a window -> mom25d Layout.

What maps onto the solver (one signal layer over a solid ground plane):
  * copper of one signal layer (normally the outer layer, e.g. top_layer),
    optionally restricted to named nets via the ODB++ EDA data
  * clipped to a rectangular window; every place a selected trace crosses the
    window boundary becomes a port (feed line extending outward)
  * the substrate (eps_r, h) is the dielectric between that layer and the
    adjacent ground layer -- ODB++ exports usually lack the stackup, so it is
    supplied by the user

  * optionally, copper pours of ground nets on the same layer (clipped to a
    band around the signal copper) and the plated vias that tie them to the
    ground layer (`ODB.vias`, from the drill layers spanning signal->ground).
    Ports whose trace has pour beside it at the window edge get GCPW feeds
    (`port_grounds`): the measured gap / ground width continue outside the
    window, with fence vias.

Not representable: components (series parts leave gaps), other layers, vias
to anything but the adjacent ground plane (signal vias to inner layers).
Non-Manhattan copper is meshed as a staircase on the tensor grid.

Written for Altium ODB++ exports (inch units).
Requires shapely >= 2.
"""

import math
import os
import re

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box

from .mesh import _DIRS, CoplanarGround, Layout

ARC_SEG_DEG = 10.0
MM = 1e-3
OZ_MM = 0.0348  # copper thickness per oz/ft^2 [mm]
CACHE_VERSION = 2  # bump whenever parsing changes, so stale layer caches are ignored


def _lines(path):
    with open(path, encoding="latin-1") as f:
        for ln in f:
            yield ln.rstrip("\r\n")


def _arc_points(xs, ys, xe, ye, xc, yc, cw):
    r = math.hypot(xs - xc, ys - yc)
    a0 = math.atan2(ys - yc, xs - xc)
    a1 = math.atan2(ye - yc, xe - xc)
    if cw:
        if a1 >= a0:
            a1 -= 2 * math.pi
    elif a1 <= a0:
        a1 += 2 * math.pi
    if abs(xs - xe) < 1e-12 and abs(ys - ye) < 1e-12:  # full circle
        a1 = a0 + (-2 * math.pi if cw else 2 * math.pi)
    n = max(2, int(abs(a1 - a0) / math.radians(ARC_SEG_DEG)) + 1)
    a = np.linspace(a0, a1, n + 1)
    return list(zip(xc + r * np.cos(a), yc + r * np.sin(a)))


def _attrlist(path):
    out = {}
    if os.path.exists(path):
        for ln in _lines(path):
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _units(path):
    """ODB++ default is inch; 'UNITS=MM' or 'U MM' switches to metric."""
    for ln in _lines(path):
        u = ln.replace(" ", "").upper()
        if u.startswith("UNITS=") or (u.startswith("U") and u[1:] in ("MM", "INCH")):
            return "MM" if u.endswith("MM") else "INCH"
        if ln and ln[0] not in "#$@&U":
            break
    return "INCH"


class ODB:
    """Reader for an extracted ODB++ directory (the one containing matrix/, steps/)."""

    def __init__(self, root, step=None, cache_dir=None):
        """cache_dir: where parsed layers are pickled (parsing a large layer
        takes tens of seconds); None disables caching."""
        self.root = root
        self.cache_dir = cache_dir
        steps = sorted(os.listdir(os.path.join(root, "steps")))
        self.step_name = step or steps[0]
        self.step = os.path.join(root, "steps", self.step_name)
        self._sym_cache = {}

    def layers(self):
        return sorted(os.listdir(os.path.join(self.step, "layers")))

    def matrix(self):
        """Layer records of matrix/matrix in stack order: dicts with lower-case
        name, type, start, end (start/end only for drill/rout layers)."""
        out, cur = [], None
        for ln in _lines(os.path.join(self.root, "matrix", "matrix")):
            t = ln.strip()
            if t.startswith("LAYER"):
                cur = {}
            elif t == "}" and cur is not None:
                if "NAME" in cur:
                    out.append(dict(row=int(cur.get("ROW", 0)), name=cur["NAME"].lower(),
                                    type=cur.get("TYPE", "").upper(),
                                    start=cur.get("START_NAME", "").lower(),
                                    end=cur.get("END_NAME", "").lower()))
                cur = None
            elif cur is not None and "=" in t:
                k, v = t.split("=", 1)
                cur[k.strip()] = v.strip()
        return sorted(out, key=lambda r: r["row"])

    def ground_layer(self, layer):
        """The signal layer next to `layer` in the stack (inward)."""
        sig = [r["name"] for r in self.matrix() if r["type"] in ("SIGNAL", "POWER_GROUND", "MIXED")]
        i = sig.index(layer.lower())
        return sig[i + 1] if i + 1 < len(sig) else sig[i - 1]

    def drill_layers(self, layer, ground):
        """Plated drill layers whose span covers both `layer` and `ground`."""
        rows = self.matrix()
        sig = {r["name"]: r["row"] for r in rows if r["type"] not in ("DRILL", "ROUT")}
        a, b = sig[layer.lower()], sig[ground.lower()]
        out = []
        for r in rows:
            if r["type"] != "DRILL" or "non-plated" in r["name"] or "npth" in r["name"]:
                continue
            if r["start"] not in sig or r["end"] not in sig:
                continue
            lo, hi = sorted((sig[r["start"]], sig[r["end"]]))
            if lo <= min(a, b) and hi >= max(a, b):
                out.append(r["name"])
        return [n for n in out if n in self.layers()]

    # ------------------------------------------------------------- stackup
    def layer_attrs(self, layer):
        """A layer's attrlist as {attribute: string} ({} if absent)."""
        return _attrlist(os.path.join(self.step, "layers", layer.lower(), "attrlist"))

    def board_attrs(self):
        """misc/attrlist ({} if absent), e.g. .board_thickness, _board_material."""
        return _attrlist(os.path.join(self.root, "misc", "attrlist"))

    def stack(self):
        """Stack data exported by the CAD tool, top to bottom: dicts with
        name, type ("copper" / "mask" / "dielectric"), thickness_mm (mask /
        dielectric layer), below_mm (vendor convention, see below),
        copper_mm, eps, tand -- None where the export lacks a value.

        Standard ODB++: DIELECTRIC layers in the matrix carry
        .layer_dielectric (thickness, in the attrlist's UNITS), and
        .dielectric_constant / .loss_tangent. Altium exports instead put
        .layer_dielectric on the copper layers, meaning the dielectric
        *below* each one (checked against .board_thickness to 0.1 %; for
        the bottom layer it repeats the bottom mask) and give no eps_r.
        eps_r <= 1 counts as missing (placeholder on masks)."""
        out = []
        for r in self.matrix():
            t = r["type"]
            kind = ("copper" if t in ("SIGNAL", "POWER_GROUND", "MIXED") else "mask" if t == "SOLDER_MASK"
                    else "dielectric" if t == "DIELECTRIC" else None)
            if kind is None:
                continue
            at = self.layer_attrs(r["name"])
            u = 1.0 if at.get("UNITS", "INCH").upper().startswith("MM") else 25.4

            def num(key):
                v = at.get(key)
                try:
                    return float(v) if v not in (None, "") else None
                except ValueError:
                    return None
            d, w, e, tg = num(".layer_dielectric"), num(".copper_weight"), num(".dielectric_constant"), \
                num(".loss_tangent")
            rec = dict(name=r["name"], type=kind, thickness_mm=None, below_mm=None, copper_mm=None,
                       eps=None, tand=None, comment=at.get(".comment", ""))
            if kind == "copper":
                rec["below_mm"] = d * u if d is not None else None
                rec["copper_mm"] = w * OZ_MM if w is not None else None
            else:
                rec["thickness_mm"] = d * u if d is not None else None
                rec["eps"] = e if e is not None and e > 1.0 else None
                rec["tand"] = tg
            out.append(rec)
        return out

    def standard_stack(self):
        """True if the export has DIELECTRIC layers (standard ODB++)."""
        return any(r["type"] == "dielectric" for r in self.stack())

    def gap(self, upper, lower):
        """Dielectric pieces from copper layer `upper` down to `lower`:
        [(thickness_mm, eps or None, tand or None, name), ...] top to
        bottom; copper layers in between count as dielectric of the piece
        above them. None if the export lacks a thickness."""
        rows = self.stack()
        cu = [i for i, r in enumerate(rows) if r["type"] == "copper"]
        names = [rows[i]["name"] for i in cu]
        i, j = cu[names.index(upper.lower())], cu[names.index(lower.lower())]
        if j <= i:
            raise ValueError(f"{lower} is not below {upper}")
        pieces = []
        if self.standard_stack():
            for r in rows[i + 1:j]:
                if r["type"] == "dielectric":
                    if r["thickness_mm"] is None:
                        return None
                    pieces.append((r["thickness_mm"], r["eps"], r["tand"], r["name"]))
                elif r["type"] == "copper":
                    if r["copper_mm"] is None or not pieces:
                        return None
                    tk, e, tg, n = pieces[-1]
                    pieces[-1] = (tk + r["copper_mm"], e, tg, n + "+" + r["name"])
            return pieces or None
        for k in cu[cu.index(i):cu.index(j)]:  # vendor (Altium) convention
            r = rows[k]
            if r["below_mm"] is None or (k != i and r["copper_mm"] is None):
                return None
            pieces.append((r["below_mm"] + (r["copper_mm"] if k != i else 0.0), None, None,
                           "below " + r["name"]))
        return pieces

    def dielectric_between(self, upper, lower):
        """Total thickness [mm] from copper layer `upper` down to `lower`."""
        g = self.gap(upper, lower)
        return None if g is None else sum(p[0] for p in g)

    def signal_rows(self):
        """{layer name: stack position} of the copper layers (top = 0)."""
        rows = [r for r in self.matrix() if r["type"] in ("SIGNAL", "POWER_GROUND", "MIXED")]
        return {r["name"]: i for i, r in enumerate(rows)}

    def drill_spans(self):
        """{plated drill layer: (start layer, end layer)}."""
        out = {}
        for r in self.matrix():
            if r["type"] != "DRILL" or "non-plated" in r["name"] or "npth" in r["name"]:
                continue
            if r["name"] in self.layers() and r["start"] and r["end"]:
                out[r["name"]] = (r["start"], r["end"])
        return out

    def all_vias(self, window=None, nets=None, drill_layers=None):
        """Every plated hole: [dict(x, y, drill, net, start, end, drill_layer)]
        in mm, optionally only `nets` / `drill_layers`, inside window."""
        out = []
        for dl, (st, en) in self.drill_spans().items():
            if drill_layers is not None and dl not in drill_layers:
                continue
            feats = self.layer_features(dl)
            netof = {i: n for n, idx in self.net_features(dl).items() for i in idx}
            for i, g in enumerate(feats):
                if g is None:
                    continue
                net = netof.get(i)
                if nets is not None and net not in nets:
                    continue
                x0, y0, x1, y1 = g.bounds
                x, y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                if window is not None and not (window[0] <= x <= window[1] and window[2] <= y <= window[3]):
                    continue
                out.append(dict(x=x, y=y, drill=max(x1 - x0, y1 - y0), net=net, start=st, end=en,
                                drill_layer=dl))
        return out

    def via_connection(self, via, levels, ground, ground_nets=()):
        """Which modelled layers a hole connects: (top, bottom) with bottom =
        "gnd" for holes of `ground_nets` that reach the ground layer, or
        None if it touches fewer than two of them. `levels` are the modelled
        copper layers; intermediate ones are connected by the solver wherever
        they have copper at the hole."""
        rows = self.signal_rows()
        lo, hi = sorted((rows[via["start"]], rows[via["end"]]))
        on = sorted((l for l in levels if lo <= rows[l] <= hi), key=lambda l: rows[l])
        if via["net"] in ground_nets and lo <= rows[ground] <= hi and on:
            return on[0], "gnd"
        if len(on) >= 2:
            return on[0], on[-1]
        return None

    def vias(self, layer, window=None, nets=None, ground=None):
        """Plated holes connecting `layer` to its ground layer:
        [(x_mm, y_mm, drill_mm, net), ...], optionally only `nets`, in window."""
        ground = ground or self.ground_layer(layer)
        out = []
        for dl in self.drill_layers(layer, ground):
            feats = self.layer_features(dl)
            netof = {i: n for n, idx in self.net_features(dl).items() for i in idx}
            for i, g in enumerate(feats):
                if g is None:
                    continue
                net = netof.get(i)
                if nets is not None and net not in nets:
                    continue
                x0, y0, x1, y1 = g.bounds
                x, y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                if window is not None and not (window[0] <= x <= window[1] and window[2] <= y <= window[3]):
                    continue
                out.append((x, y, max(x1 - x0, y1 - y0), net))
        return out

    # ------------------------------------------------------------ symbols
    def symbol(self, name, units="INCH"):
        """Standard or user symbol outline (shapely, mm) centred at origin."""
        key = (name, units)
        if key in self._sym_cache:
            return self._sym_cache[key]
        s = 25.4e-3 if units == "INCH" else 1e-3  # symbol sizes: mil or micron -> mm
        g = None
        if m := re.fullmatch(r"r([\d.]+)", name):
            g = Point(0, 0).buffer(float(m[1]) * s / 2, quad_segs=8)
        elif m := re.fullmatch(r"s([\d.]+)", name):
            a = float(m[1]) * s / 2
            g = box(-a, -a, a, a)
        elif m := re.fullmatch(r"rect([\d.]+)x([\d.]+)(?:xr([\d.]+))?(?:x(\d+))?", name):
            w, h = float(m[1]) * s, float(m[2]) * s
            g = box(-w / 2, -h / 2, w / 2, h / 2)
            if m[3]:
                r = min(float(m[3]) * s, w / 2, h / 2)
                g = box(-w / 2 + r, -h / 2 + r, w / 2 - r, h / 2 - r).buffer(r, quad_segs=4)
        elif m := re.fullmatch(r"oval([\d.]+)x([\d.]+)", name):
            w, h = float(m[1]) * s, float(m[2]) * s
            r = min(w, h) / 2
            g = (LineString([(-(w / 2 - r), -(h / 2 - r)), (w / 2 - r, h / 2 - r)]).buffer(r, quad_segs=8)
                 if w != h else Point(0, 0).buffer(r, quad_segs=8))
        elif m := re.fullmatch(r"donut_r([\d.]+)x([\d.]+)", name):
            g = Point(0, 0).buffer(float(m[1]) * s / 2).difference(Point(0, 0).buffer(float(m[2]) * s / 2))
        else:
            p = os.path.join(self.root, "symbols", name, "features")
            if os.path.exists(p):
                parts = [x for x in self.parse_features(p) if x is not None]
                g = shapely.union_all(parts) if parts else None
            else:
                print("WARNING: unknown ODB++ symbol", name)
        self._sym_cache[key] = g
        return g

    # ----------------------------------------------------------- features
    def parse_features(self, path):
        """One shapely geometry (mm) per feature record, in file order
        (None for records that produce no copper) -- the index matches the
        feature numbers used by the EDA data (FID records)."""
        units = _units(path)
        sc = 25.4 if units == "INCH" else 1.0
        syms, geoms = {}, []
        surf, contour, contours = False, None, []

        def close_contour():
            nonlocal contour
            if contour and len(contour[1]) >= 3:
                contours.append(contour)
            contour = None

        for ln in _lines(path):
            if not ln or ln[0] in "#@&":
                continue
            if ln[0] == "$":
                i, n = ln[1:].split(None, 1)
                syms[int(i)] = n.split()[0]
                continue
            rec = ln.split(";")[0].split()
            if not rec:
                continue
            t = rec[0]
            if surf:
                if t == "OB":
                    close_contour()
                    contour = [rec[3] == "I", [(float(rec[1]) * sc, float(rec[2]) * sc)]]
                elif t == "OS":
                    contour[1].append((float(rec[1]) * sc, float(rec[2]) * sc))
                elif t == "OC":
                    xs, ys = contour[1][-1]
                    xe, ye, xc, yc = (float(v) * sc for v in rec[1:5])
                    contour[1].extend(_arc_points(xs, ys, xe, ye, xc, yc, rec[5] == "Y")[1:])
                elif t == "OE":
                    close_contour()
                elif t == "SE":
                    close_contour()
                    geoms[-1] = self._surface(contours)
                    surf, contours = False, []
                continue
            if t in ("U", "UNITS", "ID", "F"):
                continue
            if t == "S":
                surf, contours = True, []
                geoms.append(None)  # filled at SE
            elif t == "L":
                xs, ys, xe, ye = (float(v) * sc for v in rec[1:5])
                geoms.append(self._stroke([(xs, ys), (xe, ye)], syms[int(rec[5])], units))
            elif t == "A":
                xs, ys, xe, ye, xc, yc = (float(v) * sc for v in rec[1:7])
                pts = _arc_points(xs, ys, xe, ye, xc, yc, rec[10] == "Y")
                geoms.append(self._stroke(pts, syms[int(rec[7])], units))
            elif t == "P":
                x, y = float(rec[1]) * sc, float(rec[2]) * sc
                sym = int(rec[3])
                g = self.symbol(syms[sym], units) if sym >= 0 else None
                if g is not None:
                    o = rec[6]
                    if o in ("8", "9"):
                        ang, mir = float(rec[7]), o == "9"
                    else:
                        ang, mir = (int(o) % 4) * 90.0, int(o) >= 4
                    if mir:
                        g = affinity.scale(g, -1, 1, origin=(0, 0))
                    if ang:
                        g = affinity.rotate(g, -ang, origin=(0, 0))  # ODB rotation is clockwise
                    g = affinity.translate(g, x, y)
                geoms.append(g)
            else:  # text, barcodes, ... keep numbering aligned
                geoms.append(None)
        # a feature record's polarity (P/N) is ignored: negative features on
        # signal layers are rare in Altium output
        return [g if g is not None and not g.is_empty else None for g in geoms]

    def _stroke(self, pts, symname, units):
        s = 25.4e-3 if units == "INCH" else 1e-3
        if m := re.fullmatch(r"r([\d.]+)", symname):
            w = float(m[1]) * s
            if w <= 0:
                return None
            return (LineString(pts).buffer(w / 2, quad_segs=4) if len(set(pts)) > 1
                    else Point(pts[0]).buffer(w / 2))
        if m := re.fullmatch(r"s([\d.]+)", symname):
            w = float(m[1]) * s
            return (LineString(pts).buffer(w / 2, cap_style="square", join_style="mitre")
                    if len(set(pts)) > 1 else Point(pts[0]).buffer(w / 2, cap_style="square"))
        g = self.symbol(symname, units)  # other brushes: bounding circle
        r = max(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1]) / 2
        return LineString(pts).buffer(r, quad_segs=4)

    @staticmethod
    def _surface(contours):
        """Surface from island (I) / hole (H) contours. Exporters (Altium) do
        not always list a hole right after its island, so each hole is
        assigned to the smallest island that contains it."""

        def poly(pts):
            p = Polygon(pts)
            return p if p.is_valid else shapely.make_valid(p)

        islands = [p for p in (poly(pts) for isl, pts in contours if isl) if not p.is_empty]
        if not islands:
            return None
        holes = [poly(pts) for isl, pts in contours if not isl]
        area = np.array([p.area for p in islands])
        tree = shapely.STRtree(islands)
        per = [[] for _ in islands]
        for h in holes:
            if h.is_empty:
                continue
            rp = h.representative_point()
            cand = [i for i in tree.query(rp) if islands[i].contains(rp)]
            if cand:
                per[min(cand, key=lambda i: area[i])].append(h)
        out = [isl.difference(shapely.union_all(hs)) if hs else isl for isl, hs in zip(islands, per)]
        return shapely.union_all(out)

    # --------------------------------------------------------------- nets
    def net_features(self, layer):
        """{net name: [feature index, ...]} for one layer, from steps/<step>/eda/data."""
        path = os.path.join(self.step, "eda", "data")
        out, net, li = {}, None, None
        for ln in _lines(path):
            if ln.startswith("LYR "):
                names = ln.split()[1:]
                if layer not in names:
                    raise ValueError(f"layer {layer} not in EDA layer list")
                li = names.index(layer)
            elif ln.startswith("NET "):
                net = ln.split()[1]
            elif ln.startswith("FID ") and net is not None:
                p = ln.split()
                if int(p[2]) == li:
                    out.setdefault(net, []).append(int(p[3]))
        return out

    def layer_features(self, layer):
        """parse_features() of a board layer, cached."""
        path = os.path.join(self.step, "layers", layer, "features")
        if self.cache_dir:
            import pickle

            cp = os.path.join(self.cache_dir, f"{self.step_name}_{layer}_v{CACHE_VERSION}.pkl")
            if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(path):
                with open(cp, "rb") as f:
                    return pickle.load(f)
        feats = self.parse_features(path)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cp, "wb") as f:
                pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        return feats

    def nets_in_window(self, layer, window):
        """{net: copper area in window [mm^2]} on a layer."""
        feats = self.layer_features(layer)
        wb = box(window[0], window[2], window[1], window[3])
        out = {}
        for net, idx in self.net_features(layer).items():
            a = sum(feats[i].intersection(wb).area for i in idx
                    if i < len(feats) and feats[i] is not None and feats[i].intersects(wb))
            if a > 0:
                out[net] = a
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def copper(self, layer, window=None, nets=None):
        """Union of the copper of `layer` (mm), optionally only `nets`, clipped
        to window = (x0, x1, y0, y1) in mm."""
        feats = self.layer_features(layer)
        if nets is not None:
            nf = self.net_features(layer)
            missing = [n for n in nets if n not in nf]
            if missing:
                raise ValueError(f"nets not found on {layer}: {missing}")
            idx = sorted({i for n in nets for i in nf[n]})
            feats = [feats[i] for i in idx if i < len(feats)]
        geoms = [g for g in feats if g is not None]
        if window is not None:
            wb = box(window[0], window[2], window[1], window[3])
            geoms = [g for g in geoms if g.intersects(wb)]
        cu = shapely.union_all(geoms) if geoms else Polygon()
        if window is not None:
            cu = cu.intersection(wb)
        # clipping can leave line/point slivers in a GeometryCollection
        polys = [p for p in getattr(cu, "geoms", [cu]) if p.geom_type in ("Polygon", "MultiPolygon")]
        return shapely.union_all(polys) if polys else Polygon()

    def profile(self):
        g = [x for x in self.parse_features(os.path.join(self.step, "profile")) if x is not None]
        return shapely.union_all(g)


# ------------------------------------------------------------------ layout
def find_ports(copper, window, min_width=0.02):
    """Where copper touches the window boundary: list of
    (ref_xy_mm, direction, width_mm)."""
    x0, x1, y0, y1 = window
    eps = 1e-4 * max(x1 - x0, y1 - y0)
    sides = [  # (edge band, direction, axis of the edge line)
        (box(x0, y0, x0 + eps, y1), "-x", 1, x0),
        (box(x1 - eps, y0, x1, y1), "+x", 1, x1),
        (box(x0, y0, x1, y0 + eps), "-y", 0, y0),
        (box(x0, y1 - eps, x1, y1), "+y", 0, y1),
    ]
    ports = []
    for band, d, ax, c in sides:
        hit = copper.intersection(band)
        for part in getattr(hit, "geoms", [hit]):
            if part.is_empty or part.area == 0:
                continue
            b = part.bounds
            lo, hi = (b[1], b[3]) if ax == 1 else (b[0], b[2])
            if hi - lo < min_width:
                continue
            ref = (c, 0.5 * (lo + hi)) if ax == 1 else (0.5 * (lo + hi), c)
            ports.append((ref, d, hi - lo))
    return ports


def port_grounds(pour, window, ports, tol=1e-4):
    """Coplanar ground of each port, measured where the pour meets the
    window edge: [(gaps, widths) or None, ...] for ports [(ref, dir, width)]
    (mm). gaps / widths are (left, right) per CoplanarGround (left = +t, the
    feed direction rotated +90 deg); a side without pour is None / 0."""
    x0, x1, y0, y1 = window
    eps = 1e-4 * max(x1 - x0, y1 - y0)
    edge = {"-x": box(x0, y0, x0 + eps, y1), "+x": box(x1 - eps, y0, x1, y1),
            "-y": box(x0, y0, x1, y0 + eps), "+y": box(x0, y1 - eps, x1, y1)}
    out = []
    for ref, d, w in ports:
        u = np.array(_DIRS[d], float)
        n = np.array([-u[1], u[0]])
        ax = 1 if d[1] == "x" else 0  # coordinate along the window edge
        hit = pour.intersection(edge[d]) if pour is not None and not pour.is_empty else Polygon()
        iv = []
        for part in getattr(hit, "geoms", [hit]):
            if part.is_empty or part.area == 0:
                continue
            b = part.bounds
            lo, hi = (b[1], b[3]) if ax == 1 else (b[0], b[2])
            iv.append(tuple(sorted(((lo - ref[ax]) * n[ax], (hi - ref[ax]) * n[ax]))))
        left = [t for t in iv if t[0] >= w / 2 - tol]
        right = [t for t in iv if t[1] <= -w / 2 + tol]
        gl = min(left, key=lambda t: t[0]) if left else None
        gr = max(right, key=lambda t: t[1]) if right else None
        if gl is None and gr is None:
            out.append(None)
            continue
        gaps = (gl[0] - w / 2 if gl else None, -w / 2 - gr[1] if gr else None)
        widths = (gl[1] - gl[0] if gl else 0.0, gr[1] - gr[0] if gr else 0.0)
        out.append((gaps, widths))
    return out


def layout_from_copper(copper, window, feed_length, ports=None, min_port_width=0.02,
                       pour=None, vias=()):
    """Single-layer Layout (SI units) from copper in mm. Ports default to every boundary
    crossing; each gets a straight feed line of feed_length [m] outside the
    window, de-embedded to the window edge. `pour` (mm) is extra copper that
    gets no ports (e.g. a GND pour), `vias` [(x_mm, y_mm, drill_mm, ...)] are
    grounding vias (pad = the via's cell / pour copper)."""
    lay = Layout()
    lay.add_polygon(affinity.scale(copper, MM, MM, origin=(0, 0)))
    if pour is not None and not pour.is_empty:
        lay.add_polygon(affinity.scale(pour, MM, MM, origin=(0, 0)))
    for v in vias:
        lay.add_via(v[0] * MM, v[1] * MM, drill=v[2] * MM, pad=v[2] * MM)
    if ports is None:
        ports = find_ports(copper, window, min_port_width)
    for k, (ref, d, w) in enumerate(ports):
        lay.add_port(f"P{k + 1}", (ref[0] * MM, ref[1] * MM), d, w * MM, feed_length)
    return lay


def layout_from_layers(copper, window, feed_length, ports, pours=None, vias=(),
                       feed_vias=None):
    """Multilayer Layout (SI units) from {layer: copper (mm)}.

    ports : {layer: [(ref, dir, width[, ground]), ...]} -> feed lines of
            feed_length [m] outside the window on that layer, de-embedded to
            the window edge; ground = (gaps, widths) in mm from port_grounds
            makes it a GCPW feed
    feed_vias : dict(pitch=, drill=, pad=) in mm for the GCPW fence vias
    pours : {layer: pour copper (mm)} (no ports)
    vias  : [dict(x, y, drill, top, bottom), ...] in mm; top / bottom are
            layer names or "gnd". Pads are drill-sized (the pad copper comes
            from the layers themselves)."""
    lay = Layout()
    for L, cu in copper.items():
        if not cu.is_empty:
            lay.add_polygon(affinity.scale(cu, MM, MM, origin=(0, 0)), layer=L)
    for L, pg in (pours or {}).items():
        if pg is not None and not pg.is_empty:
            lay.add_polygon(affinity.scale(pg, MM, MM, origin=(0, 0)), layer=L)
    for v in vias:
        lay.add_via(v["x"] * MM, v["y"] * MM, drill=v["drill"] * MM, pad=v["drill"] * MM,
                    top=v["top"], bottom=v["bottom"])
    k = 0
    fv = feed_vias or {}
    for L, pl in ports.items():
        for ref, d, w, *g in pl:
            k += 1
            gnd = None
            if g and g[0] is not None:
                (gl, gr), (wl, wr) = g[0]
                mm_ = lambda x: None if x is None else x * MM  # noqa: E731
                gnd = CoplanarGround((mm_(gl), mm_(gr)), (wl * MM, wr * MM),
                                     via_pitch=fv["pitch"] * MM if fv.get("pitch") else None,
                                     via_drill=fv.get("drill", 0.3) * MM,
                                     via_pad=fv["pad"] * MM if fv.get("pad") else None)
            lay.add_port(f"P{k}", (ref[0] * MM, ref[1] * MM), d, w * MM, feed_length, layer=L,
                         ground=gnd)
    return lay


def stitched_pour(odb, layer, window, copper, nets=("GND",), band=1.0, ground=None):
    """Ground-pour copper of `nets` on `layer` within `band` mm of `copper`
    (clipped to the window), and the grounding vias inside it. Pour islands
    without a via are dropped (they would float).

    Returns (pour, vias, n_dropped) with vias [(x, y, drill, net), ...] in mm."""
    wb = box(window[0], window[2], window[1], window[3])
    pour = odb.copper(layer, window, list(nets))
    pour = pour.difference(copper).intersection(copper.buffer(band, join_style="mitre")).intersection(wb)
    parts = [p for p in getattr(pour, "geoms", [pour]) if p.geom_type == "Polygon" and p.area > 0]
    vias = [v for v in odb.vias(layer, window, set(nets), ground)]
    keep, kv = [], []
    for p in parts:
        inner = p.buffer(-1e-3)
        mine = [v for v in vias if inner.contains(Point(v[0], v[1]))]
        if mine:
            keep.append(p)
            kv += mine
    return (shapely.union_all(keep) if keep else Polygon()), kv, len(parts) - len(keep)


def build_stackup(odb, levels, ground, er=None, tand=None, dielectrics=None, mask=True,
                  mask_er=3.5, mask_tand=0.02, sigma=None, t=None):
    """Stackup for modelled copper `levels` (top to bottom) over `ground`.

    Per gap (below each level, top to bottom) the dielectric comes from the
    export (`ODB.gap`): a standard export gives every dielectric layer with
    its thickness, eps_r and tan d (each becomes a layer of the model);
    an Altium-style export gives only the total thickness. `er` / `tand`
    override the export's eps_r / tan d for every gap (one homogeneous
    layer per gap); `dielectrics` = [(t_mm or None, er, tand or None), ...]
    overrides per gap (None thickness: the export's). An eps_r found
    nowhere is an error. A top solder mask (export thickness, eps_r if
    given there, else mask_er) is added over the top level when that is
    the outer copper layer and mask=True. Copper thickness t [m] defaults
    to the export's weight of the top level.

    Returns (Stackup, diel [(t_mm, eps_eff, tand) per gap, top to bottom;
    eps_eff thickness-weighted], report lines naming every value's source)."""
    from .layered import Dielectric, Stackup

    rep = []
    lower = list(levels[1:]) + [ground]
    gaps = []  # per gap: [(t_mm, eps, tand)] top to bottom
    for k, (L, M) in enumerate(zip(levels, lower)):
        given = dielectrics[k] if dielectrics else None
        pieces = odb.gap(L, M)
        if given and given[0] is not None:
            total, src = given[0], "given"
        elif pieces:
            total, src = sum(p[0] for p in pieces), "ODB++"
        else:
            raise ValueError(f"dielectric {L} -> {M}: not in the export, give its thickness")
        g_er = given[1] if given else er
        g_td = given[2] if given and len(given) > 2 and given[2] is not None else tand
        if g_er is not None:  # one homogeneous layer with the given eps_r
            gaps.append([(total, g_er, g_td if g_td is not None else 0.0)])
            rep.append(f"{L} -> {M}: {total:.4f} mm ({src}), eps_r {g_er} (given), "
                       f"tan d {gaps[-1][0][2]}")
            continue
        if not pieces or any(p[1] is None for p in pieces):
            mat = odb.board_attrs().get("_board_material", "") or "unknown"
            raise ValueError(f"eps_r of the dielectric {L} -> {M} is not in the export "
                             f"(board material: {mat}): give it")
        scale = total / sum(p[0] for p in pieces)  # a given total thickness rescales the pieces
        gap = [(p[0] * scale, p[1], (p[2] or 0.0) if g_td is None else g_td) for p in pieces]
        gaps.append(gap)
        for (tk, e, td), p in zip(gap, pieces):
            rep.append(f"{L} -> {M}: {p[3]} {tk:.4f} mm ({src}), eps_r {e}, tan d {td} (ODB++)")
    diel = []
    for gap in gaps:
        tt = sum(p[0] for p in gap)
        diel.append((tt, sum(p[0] * p[1] for p in gap) / tt, sum(p[0] * p[2] for p in gap) / tt))
    layers, metals, n = [], {}, 0
    for i in range(len(levels) - 1, -1, -1):  # bottom (on the ground) first
        for tk, e, td in reversed(gaps[i]):
            layers.append(Dielectric(tk * MM, e, td))
        n = len(layers)
        metals[levels[i]] = n
    rows = odb.stack()
    cu = [r for r in rows if r["type"] == "copper"]
    if t is None:
        top = next(r for r in cu if r["name"] == levels[0].lower())
        t = top["copper_mm"] * MM if top["copper_mm"] else 35e-6
        rep.append(f"copper {t / MM * 1e3:.1f} um ({'ODB++ weight' if top['copper_mm'] else 'default'})")
    if mask and cu and cu[0]["name"] == levels[0].lower():
        above = [r for r in rows[:rows.index(cu[0])] if r["type"] == "mask"]
        if above and above[-1]["thickness_mm"]:
            m = above[-1]
            me = m["eps"] or mask_er
            md = m["tand"] if m["tand"] is not None else mask_tand
            layers.append(Dielectric(m["thickness_mm"] * MM, me, md))
            rep.append(f"solder mask over {levels[0]}: {m['thickness_mm'] * 1e3:.1f} um (ODB++"
                       + (f", {m['comment']}" if m["comment"] else "") + f"), eps_r {me} "
                       + ("(ODB++)" if m["eps"] else "(assumed, --mask-er)") + f", tan d {md} (planar layer)")
    return Stackup(layers, metals, sigma=sigma, t=t), diel, rep


# ---------------------------------------------------------------- components
def _net_names(odb):
    """Net names in eda/data order (TOP records reference them by index)."""
    out = []
    path = os.path.join(odb.step, "eda", "data")
    if os.path.exists(path):
        for ln in _lines(path):
            if ln.startswith("NET "):
                out.append(ln.split()[1].split(";")[0])
    return out


def components(odb, side="top"):
    """Components of one board side from layers/comp_+_top (or _bot):
    [dict(refdes, part, x, y [mm], rot, value, type, pins=[dict(name, x, y,
    net)]), ...]."""
    path = os.path.join(odb.step, "layers", f"comp_+_{side[:3]}", "components")
    if not os.path.exists(path):
        return []
    nets = _net_names(odb)
    u = 25.4
    out, cur = [], None
    for ln in _lines(path):
        t = ln.strip()
        if t.upper().startswith("UNITS="):
            u = 1.0 if t.split("=", 1)[1].strip().upper().startswith("MM") else 25.4
        elif t.startswith("CMP "):
            f = t.split(";")[0].split()
            cur = dict(refdes=f[6], part=f[7] if len(f) > 7 else "", x=float(f[2]) * u,
                       y=float(f[3]) * u, rot=float(f[4]), value="", type="", pins=[])
            out.append(cur)
        elif t.startswith("PRP ") and cur is not None:
            name = t.split()[1].upper()
            val = t.split("'")[1] if t.count("'") >= 2 else ""
            if name in ("VALUE", "TYPE"):
                cur[name.lower()] = val
        elif t.startswith("TOP ") and cur is not None:
            f = t.split(";")[0].split()
            k = int(f[6]) if len(f) > 6 else -1
            cur["pins"].append(dict(name=f[8] if len(f) > 8 else f[1], x=float(f[2]) * u,
                                    y=float(f[3]) * u, net=nets[k] if 0 <= k < len(nets) else ""))
    return out


_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "\u00b5": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3,
       "meg": 1e6, "M": 1e6, "G": 1e9}


def component_value(comp):
    """lumped Value from a component's VALUE text ('3.3Kohms', '100nF',
    '2.2nH', '0R', '10pF') and its kind (refdes prefix R / C / L or TYPE);
    None if it cannot be read."""
    from .lumped import C, L, R

    txt = comp.get("value", "").strip().replace(" ", "").replace("\u03a9", "ohm")
    kind = (comp.get("type", "") or comp["refdes"]).strip().upper()[:1]
    m = re.match(r"^([0-9]*\.?[0-9]+)(meg|[fpnumkKMG\u00b5]?)(R?)([0-9]*)(ohms?|f|h)?$", txt, re.I)
    if not m:
        return None
    num = float(m.group(1)) * _SI.get(m.group(2), 1.0)
    if m.group(3):  # 4R7 style
        num = float(m.group(1) + "." + (m.group(4) or "0"))
    unit = (m.group(5) or "").lower()
    kind = {"o": "R", "f": "C", "h": "L"}.get(unit[:1], kind)
    if not unit and not m.group(2) and not m.group(3) and kind != "R":
        return None  # a bare number is only unambiguous for a resistor
    return {"R": R, "C": C, "L": L}.get(kind, lambda v: None)(num)


def place_components(layout, comps, copper, layer, window=None, values=None, width_frac=0.8):
    """Add a lumped bridge (Layout.add_lumped) for every 2-pin component whose
    pins both land on `copper` (mm) inside `window`. values: {refdes: value
    (lumped Value, 'C 1p'-style text or a path to a series .s2p)} overriding
    the export's VALUE. Returns (placed [(refdes, value)], skipped
    [(refdes, reason)])."""
    from shapely.geometry import LineString

    from .lumped import series_from_s2p

    placed, skipped = [], []
    wbox = box(window[0], window[2], window[1], window[3]) if window else None
    for c in comps:
        pins = c["pins"]
        if len(pins) != 2:
            continue
        pts = [Point(p["x"], p["y"]) for p in pins]
        if wbox is not None and not all(wbox.contains(p) for p in pts):
            continue
        if not all(copper.buffer(1e-3).contains(p) for p in pts):
            skipped.append((c["refdes"], "a pad is not on the modelled copper (net not selected?)"))
            continue
        v = (values or {}).get(c["refdes"])
        if isinstance(v, str) and v.lower().endswith(".s2p"):
            v = series_from_s2p(v)
        elif v is None:
            v = component_value(c)
            if v is None:
                skipped.append((c["refdes"], f"no usable value ({c.get('value') or 'none'})"))
                continue
        a, b = np.array([pins[0]["x"], pins[0]["y"]]), np.array([pins[1]["x"], pins[1]["y"]])
        d = (b - a) / np.hypot(*(b - a))
        n = np.array([-d[1], d[0]])
        widths = []
        for p in (a, b):  # pad width across the bridge
            cut = copper.intersection(LineString([p - 2 * n, p + 2 * n]))
            seg = [g for g in getattr(cut, "geoms", [cut]) if g.distance(Point(*p)) < 1e-6]
            widths.append(seg[0].length if seg else 0.0)
        w = width_frac * min(widths)
        if w <= 0:
            skipped.append((c["refdes"], "pad width not found"))
            continue
        layout.add_lumped(c["refdes"], tuple(a * MM), tuple(b * MM), w * MM, v, layer=layer)
        placed.append((c["refdes"], v))
    return placed, skipped
