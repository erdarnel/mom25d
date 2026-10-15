"""Manhattan layouts on one or more metal layers, ports, vias, copper pours
and structured triangular meshing with RWG edges.

Every feature lives on a named metal layer (default "top"; the names are
the metal levels of the Stackup, "gnd" is the ground plane). Layouts are
unions of axis-aligned rectangles (plus shapely polygons, via pads and
copper pours). One tensor-product grid is shared by all layers; it is built
from all rectangle / port coordinates (plus optional edge refinement), every
grid cell whose centre lies inside the layout is split into two triangles,
and every interior edge becomes one RWG basis function.

Keeping transverse cuts of feed lines on grid lines makes the total current
through a cut simply the sum of the RWG coefficients (times edge lengths) of
the edges lying on it, which is what port de-embedding uses.
"""

from dataclasses import dataclass, field

import numpy as np

_DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
FRILL_RINGS = 4  # radial potential basis functions per antipad (antipad.py)


@dataclass
class CoplanarGround:
    """Side grounds of a grounded coplanar (GCPW) port feed.

    gap, width : gap between the line and the ground strip, and the strip's
                 width [m]; a number for both sides or (left, right), where
                 left is +t (the feed's normal n = direction rotated +90 deg).
                 None (or a 0 width) leaves that side without ground.
    via_pitch  : fence vias along each strip centre (None: no vias -- the
                 strips must then be grounded by other copper)
    via_drill, via_pad : fence via drill / pad diameter [m] (pad default
                 min(2 drill, 0.9 width))
    via_to     : layer the fence vias go to (default the ground plane)
    via_connect : inner planes the fence vias connect to on the way (as
                 Via.connect, e.g. an inner ground under the feed grounds)
    """

    gap: object
    width: object
    via_pitch: float = None
    via_drill: float = 0.3e-3
    via_pad: float = None
    via_to: str = "gnd"
    via_connect: tuple = ()

    def sides(self):
        """[(sign of t, gap, width), ...] for the sides that have ground."""
        g = self.gap if isinstance(self.gap, (tuple, list)) else (self.gap, self.gap)
        w = self.width if isinstance(self.width, (tuple, list)) else (self.width, self.width)
        return [(sg, float(gi), float(wi)) for sg, gi, wi in zip((1, -1), g, w)
                if gi is not None and wi]

    def key(self, scale):
        """Hashable description, lengths divided by `scale` (port grouping)."""
        return tuple((sg, round(gi / scale, 6), round(wi / scale, 6)) for sg, gi, wi in self.sides())


@dataclass
class Port:
    """A microstrip feed line with a delta-gap source, de-embedded to `ref`.

    ref        : (x, y) centre of the line at the reference plane [m]
    direction  : '+x', '-x', '+y' or '-y' -- points AWAY from the circuit,
                 i.e. along the feed line from the reference plane to its end;
                 or any direction as an angle [deg from +x] or a vector
                 (oblique feeds need the triangular mesh, kind="tri")
    width      : line width [m]
    length     : feed line length from reference plane to open end [m]
    gap_from_end : position of the delta-gap measured from the open end
    margin     : distance kept free of near fields when fitting standing waves
                 (from the reference plane and from the gap)
    layer      : metal layer of the feed line
    ground     : CoplanarGround -> a GCPW feed: side ground strips (and fence
                 vias) along the whole feed, from the reference plane on
    z0_from    : "voltage" (default): the port's line impedance comes from its
                 sampled line voltage; "reciprocity": only from reciprocity
                 with the "voltage" ports (Solver._reciprocal_z0) -- for feeds
                 whose line voltage (-Int E_z down to the plane below) is not
                 the voltage of the line, e.g. a strip whose return plane is
                 ABOVE it (an inverted outer layer over a dummy ground)
    """

    name: str
    ref: tuple
    direction: object
    width: float
    length: float
    gap_from_end: float = None
    margin: float = None
    layer: str = "top"
    ground: CoplanarGround = None
    z0_from: str = "voltage"

    def __post_init__(self):
        if self.z0_from not in ("voltage", "reciprocity"):
            raise ValueError("z0_from must be 'voltage' or 'reciprocity'")
        d = self.direction
        if isinstance(d, str):
            if d not in _DIRS:
                raise ValueError(f"direction must be one of {list(_DIRS)}, an angle or a vector")
            u = np.array(_DIRS[d], float)
        else:
            u = np.atleast_1d(np.asarray(d, float))
            if u.size == 1:
                a = np.radians(u[0])
                u = np.array([np.cos(a), np.sin(a)])
            u = u / np.linalg.norm(u)
            for k, v in _DIRS.items():  # axis-aligned after all: use the name
                if np.allclose(u, v, atol=1e-12):
                    self.direction, u = k, np.array(v, float)
        self.u = u  # outward unit vector along the feed
        self.n = np.array([-u[1], u[0]])  # unit normal (t axis)
        self.ref = (float(self.ref[0]), float(self.ref[1]))
        if self.gap_from_end is None:
            self.gap_from_end = self.width
        self.aligned = isinstance(self.direction, str)
        if self.aligned:
            self.axis = 0 if self.direction[1] == "x" else 1
            self.sgn = _DIRS[self.direction][self.axis]
        else:
            self.axis = self.sgn = None

    @property
    def s_gap(self):
        return self.length - self.gap_from_end

    def coord(self, s):
        """Grid coordinate (along the port axis) at outward distance s."""
        return self.ref[self.axis] + self.sgn * s

    def point(self, s, t=0.0):
        """(x, y) at outward distance s along the feed, t across it."""
        s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
        return np.asarray(self.ref) + s[..., None] * self.u + t[..., None] * self.n

    def local(self, xy):
        """(s, t) feed coordinates of points (..., 2)."""
        d = np.asarray(xy, float) - np.asarray(self.ref)
        return d @ self.u, d @ self.n

    def corners(self):
        """Feed rectangle corners (any direction)."""
        W2 = self.width / 2
        return [tuple(self.point(s, t)) for s, t in
                ((0, -W2), (self.length, -W2), (self.length, W2), (0, W2))]

    def ground_strips(self):
        """Corner lists of the GCPW side-ground strips (empty for microstrip)."""
        if self.ground is None:
            return []
        out = []
        for sg, g, w in self.ground.sides():
            t0, t1 = sorted((sg * (self.width / 2 + g), sg * (self.width / 2 + g + w)))
            out.append([tuple(self.point(s, t)) for s, t in
                        ((0, t0), (self.length, t0), (self.length, t1), (0, t1))])
        return out

    def ground_vias(self):
        """Fence vias of the GCPW side grounds: Via objects."""
        G = self.ground
        if G is None or not G.via_pitch:
            return []
        n = int(np.floor(self.length / G.via_pitch))
        s = (np.arange(n) + 0.5) * G.via_pitch
        out = []
        for sg, g, w in G.sides():
            pad = min(G.via_pad or 2 * G.via_drill, 0.9 * w)
            if pad < G.via_drill:
                raise ValueError(f"port {self.name}: ground strip {w * 1e3:.3g} mm too narrow "
                                 f"for {G.via_drill * 1e3:.3g} mm fence vias")
            t = sg * (self.width / 2 + g + w / 2)
            out += [Via(tuple(self.point(si, t)), G.via_drill, pad, top=self.layer, bottom=G.via_to,
                        connect=tuple(G.via_connect))
                    for si in s[s <= self.length - pad / 2]]
        return out

    def rect(self):
        if not self.aligned:
            raise ValueError(f"port {self.name}: oblique feed has no axis-aligned rectangle")
        a = self.axis
        c0, c1 = sorted((self.ref[a], self.coord(self.length)))
        t = self.ref[1 - a]
        lo, hi = t - self.width / 2, t + self.width / 2
        return (c0, lo, c1, hi) if a == 0 else (lo, c0, hi, c1)


@dataclass
class Via:
    """Plated via between two metal layers of the stackup.

    xy     : centre [m]
    drill  : finished hole diameter [m] (the current flows on its wall)
    pad    : pad diameter [m]; a pad is added on the `top` layer and on the
             `bottom` layer (unless that is the ground plane)
    top, bottom : layers the via connects ("gnd" = ground plane, or an
             inner plane declared with Layout.add_plane). On layers in
             between it also connects wherever there is copper at its
             position (cut an antipad into that copper to pass through)
    connect : inner planes between top and bottom the via is connected to;
             it passes all other planes through an antipad
    antipad : clearance diameter [m] in the planes it passes; None: the
             hole is treated as negligibly small (plane solid around the via);
             or {plane: diameter} per plane (planes not listed: no hole)
    The via current enters the triangles inside the hole on each connected
    layer (at least the mesh cell containing the centre).
    """

    xy: tuple
    drill: float
    pad: float = None
    name: str = None
    top: str = "top"
    bottom: str = "gnd"
    connect: tuple = ()
    antipad: float = None

    def __post_init__(self):
        self.xy = (float(self.xy[0]), float(self.xy[1]))
        self.connect = tuple(self.connect)
        aps = self.antipad.values() if isinstance(self.antipad, dict) else [self.antipad]
        if any(d is not None and d <= self.drill for d in aps):
            raise ValueError("via antipad must be larger than its drill")
        if self.pad is None:
            self.pad = 2 * self.drill
        if self.pad < self.drill:
            raise ValueError("via pad smaller than its drill")
        if self.top == self.bottom:
            raise ValueError("via top and bottom layer are the same")

    def pad_layers(self):
        return [n for n in (self.top, self.bottom) if n != "gnd"]

    def antipad_at(self, plane):
        """Clearance diameter [m] in `plane` (None: no modelled hole)."""
        if isinstance(self.antipad, dict):
            return self.antipad.get(plane)
        return self.antipad


@dataclass
class Pour:
    """Copper pour: `region` minus every other copper feature grown by
    `clearance`, optionally stitched to ground with vias.

    via_pitch  : spacing of the via fence that follows all pour edges
                 (None: floating pour, no vias)
    fill_pitch : additional square grid of vias over the pour interior
    via_offset : distance of fence via centres from the pour edge
                 (default: pad radius + drill / 2)
    layer      : metal layer of the pour
    via_to     : layer the stitching vias go to (default the ground plane)
    """

    region: object  # shapely geometry [m]
    clearance: float
    via_pitch: float = None
    via_drill: float = 0.3e-3
    via_pad: float = None
    via_offset: float = None
    fill_pitch: float = None
    layer: str = "top"
    via_to: str = "gnd"

    def resolve(self, signal):
        """(pour geometry, [Via, ...]) for the given other copper."""
        import shapely

        geom = self.region
        if signal is not None and not signal.is_empty:
            # mitred joins keep Manhattan copper Manhattan after the buffer
            geom = geom.difference(signal.buffer(self.clearance, join_style="mitre"))
        geom = shapely.union_all([p for p in getattr(geom, "geoms", [geom])
                                  if p.geom_type == "Polygon" and p.area > 0])
        vias = []
        if self.via_pitch or self.fill_pitch:
            pad = self.via_pad or 2 * self.via_drill
            off = self.via_offset if self.via_offset is not None else 0.5 * (pad + self.via_drill)
            core = geom.buffer(-off, join_style="mitre")
            pts = []
            if self.via_pitch:
                for p in getattr(core, "geoms", [core]):
                    if p.geom_type != "Polygon" or p.is_empty:
                        continue
                    for ring in [p.exterior, *p.interiors]:
                        n = max(int(np.ceil(ring.length / self.via_pitch)), 1)
                        pts += [ring.interpolate(t).coords[0]
                                for t in np.arange(n) * ring.length / n]
            if self.fill_pitch:
                x0, y0, x1, y1 = core.bounds
                xs = np.arange(x0, x1 + 1e-12, self.fill_pitch)
                ys = np.arange(y0, y1 + 1e-12, self.fill_pitch)
                X, Y = np.meshgrid(xs + 0.5 * (x1 - xs[-1]), ys + 0.5 * (y1 - ys[-1]))
                inside = shapely.contains_xy(core, X.ravel(), Y.ravel())
                pts += list(zip(X.ravel()[inside], Y.ravel()[inside]))
            # drop near-duplicates (fence corners, fill next to the fence)
            dmin = 0.5 * min(p for p in (self.via_pitch, self.fill_pitch) if p)
            kept = []
            for q in pts:
                if all(np.hypot(q[0] - k[0], q[1] - k[1]) >= dmin for k in kept):
                    kept.append(q)
            vias = [Via(q, self.via_drill, pad, top=self.layer, bottom=self.via_to) for q in kept]
        return geom, vias


@dataclass
class Layout:
    """Copper on named metal layers. Every add_* takes `layer=` (default
    "top"); layer names must be metal levels of the Stackup used to solve."""

    rects: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    polys: list = field(default_factory=list)  # shapely geometries [m]
    vias: list = field(default_factory=list)
    pours: list = field(default_factory=list)
    rect_layers: list = field(default_factory=list)
    poly_layers: list = field(default_factory=list)
    planes: list = field(default_factory=list)
    lumped: list = field(default_factory=list)  # internal ports with components
    components: list = field(default_factory=list)  # multi-port components over lumped gaps
    ground_rects: set = field(default_factory=set)  # indices of GCPW feed grounds
    ground_polys: set = field(default_factory=set)
    thick: dict = field(default_factory=dict)  # thick-metal layers: {layer: twin level}
    internal_ports: list = field(default_factory=list)  # gap / via ports: dict(kind, name, z_ref, ...)
    apertures: list = field(default_factory=list)  # (lower plane, upper plane, geometry [m], max_cell, edge_cell)

    def _touch(self):
        self._unions = {}

    def add_thickness(self, layer, twin):
        """Sonnet-style thick metal: all copper of `layer` is meshed twice,
        on `layer` and on `twin` -- the metal level of the Stackup at the
        copper's other face (one copper thickness above or below it, same
        stack section, no copper of its own) -- and the two sheets are
        joined along every copper edge by vertical side-wall currents
        (Mesh.wall). Ports and lumped gaps on `layer` span both sheets."""
        if layer == twin:
            raise ValueError("thick metal: layer and twin are the same")
        if layer in self.thick.values() or twin in self.thick or twin in self.thick.values():
            raise ValueError(f"thick metal {layer!r} / {twin!r}: level already used")
        self.thick[layer] = twin
        return self

    def add_rect(self, x0, y0, x1, y1, layer="top"):
        self.rects.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        self.rect_layers.append(layer)
        self._touch()
        return self

    def add_lumped(self, name, a, b, width, value=None, layer="top"):
        """Lumped component (internal port) from pad point a to pad point b
        (x, y) [m]: a copper bridge of `width` with a delta gap in its middle,
        loaded with `value` (lumped.R / L / C / series / parallel, a number in
        ohm, a callable f -> Z, a string like 'C 1p' or 'R 10 + L 0.5n';
        None = open gap, e.g. a port of a multi-port component). The
        voltage is taken a -> b. Returns the bridge's Port geometry."""
        from .lumped import as_value, parse

        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        length = float(np.hypot(*d))
        if length <= 0:
            raise ValueError("lumped component: a and b coincide")
        if any(q.name == name for q in self.lumped) or any(q.name == name for q in self.ports):
            raise ValueError(f"name {name!r} already used")
        g = Port(name, tuple(a), d, width, length, gap_from_end=0.5 * length, layer=layer)
        g.value = None if value is None else (parse(value) if isinstance(value, str) else as_value(value))
        self.lumped.append(g)
        if g.aligned:
            self.add_rect(*g.rect(), layer=layer)
        else:
            from shapely.geometry import Polygon

            self.add_polygon(Polygon(g.corners()), layer=layer)
        return g

    def add_sparams(self, name, model, ports, width, layer="top", ground_pad=None, via_drill=0.3e-3):
        """S-parameter block (Touchstone file path or lumped.Touchstone) in the
        layout: port k of the model is a gap from ports[k] = (signal point,
        reference point) [m], i.e. its voltage is signal minus reference.
        The reference points must be grounded copper -- or pass ground_pad
        (pad size [m]) to add a square pad with a via to the ground plane at
        each distinct reference point. Gaps are named '<name>.<k+1>'."""
        from .lumped import Touchstone

        if isinstance(model, str):
            model = Touchstone(model)
        if getattr(model, "n", len(ports)) != len(ports):
            raise ValueError(f"{model} has {model.n} ports, {len(ports)} given")
        gaps = []
        refs = []
        for k, (sig, ref) in enumerate(ports):
            gaps.append(self.add_lumped(f"{name}.{k + 1}", sig, ref, width, None, layer).name)
            if ground_pad and not any(np.allclose(ref, r) for r in refs):
                refs.append(ref)
                x, y = ref
                self.add_rect(x - ground_pad / 2, y - ground_pad / 2, x + ground_pad / 2, y + ground_pad / 2,
                              layer=layer)
                self.add_via(x, y, drill=via_drill, pad=ground_pad, top=layer, bottom="gnd")
        self.add_component(model, gaps)
        return gaps

    def add_gap_port(self, name, a, b, width, layer="top", z_ref=50.0):
        """Internal port (an S-parameter port of the solution, reference
        impedance z_ref [ohm]) across a delta gap: a copper bridge from pad
        point a to pad point b (x, y) [m] like add_lumped, voltage a -> b.
        E.g. a device pin (a) to its ground pad (b)."""
        g = self.add_lumped(name, a, b, width, None, layer)
        self.internal_ports.append(dict(kind="gap", name=name, z_ref=float(z_ref)))
        return g

    def add_via_port(self, name, via, z_ref=50.0):
        """Internal port at the bottom of a via that ends on the ground plane
        ("gnd"): a delta gap between the barrel and the plane, i.e. the via
        is fed through a hole in the ground (a coax-like port seen from the
        layout side; the hole's own capacitance is not modelled). `via`:
        index into Layout.vias, or the Via. Voltage barrel -> ground, port
        current flowing down the via. Two halves of a board split at a
        plane are joined by connecting their via ports (network.connect)."""
        k = via if isinstance(via, (int, np.integer)) else next(i for i, v in enumerate(self.vias) if v is via)
        v = self.vias[k]
        if v.bottom != "gnd":
            raise ValueError(f"via port {name!r}: the via must end on the ground plane (bottom='gnd')")
        names = [q.name for q in self.lumped] + [q.name for q in self.ports] + [q["name"] for q in self.internal_ports]
        if name in names:
            raise ValueError(f"name {name!r} already used")
        self.internal_ports.append(dict(kind="via", name=name, via=int(k), z_ref=float(z_ref)))
        return k

    def add_component(self, model, gaps):
        """Multi-port component (e.g. lumped.Touchstone or a .sNp path) across
        the named lumped gaps, in the model's port order (add those gaps with
        value=None)."""
        from .lumped import Touchstone

        if isinstance(model, str):
            model = Touchstone(model)
        names = [q.name for q in self.lumped]
        for n in gaps:
            if n not in names:
                raise ValueError(f"no lumped gap {n!r}")
        if getattr(model, "n", len(gaps)) != len(gaps):
            raise ValueError(f"{model} has {model.n} ports, {len(gaps)} gaps given")
        self.components.append((model, list(gaps)))
        return self

    def add_plane(self, name):
        """Declare a solid inner PEC plane of the Stackup (Stackup.planes)
        that vias may end on or connect to. Planes are not meshed."""
        if name not in self.planes:
            self.planes.append(name)
        self._touch()
        return self

    def add_aperture(self, plane, region, max_cell=None, edge_cell=None):
        """Opening (slot, keep-out) of any shape in a solid inner plane:
        `region` a shapely geometry [m]; `plane` a plane name, or (lower,
        upper) for a thick plane modelled by two planes at its faces with
        the same opening in both (a thin slot with PEC walls, see
        aperture.py). The opening carries magnetic-current unknowns on its
        own mesh: max_cell / edge_cell [m], default min(layout max_cell,
        2.5 x layout edge_cell) and half the layout edge_cell -- the slot
        field needs several cells across the opening (with the layout's
        0.3 mm cells a 0.6 mm slot under a CPW gap was 0.16 off in eps_eff)."""
        lo, hi = (plane, plane) if isinstance(plane, str) else tuple(plane)
        for p in (lo, hi):
            self.add_plane(p)
        self.apertures.append((lo, hi, region, max_cell, edge_cell))
        self._touch()
        return self

    def add_via(self, x, y, drill=0.3e-3, pad=None, name=None, top="top", bottom="gnd",
                connect=(), antipad=None):
        """Via at (x, y) from layer `top` to layer `bottom` ("gnd": the ground
        plane; or an inner plane). Its pads are added to the copper of both
        layers (not to planes). `connect`: inner planes in between it is
        connected to (it passes the others through antipads of diameter
        `antipad`, modelled when given)."""
        v = Via((x, y), drill, pad, name, top, bottom, connect, antipad)
        self.vias.append(v)
        self._touch()
        return v

    def add_pour(self, region, clearance, via_pitch=None, via_drill=0.3e-3, via_pad=None,
                 via_offset=None, fill_pitch=None, layer="top", via_to="gnd"):
        """Copper pour (normally ground) over `region` (shapely geometry or a
        rectangle (x0, y0, x1, y1), metres) on `layer`. All other copper of
        that layer -- rectangles, polygons, port feed lines and via pads --
        is cut out of it with the given `clearance` when the layout is
        meshed, so ports added later are respected. With `via_pitch` a via
        fence is placed along every pour edge (and with `fill_pitch` a grid of
        vias inside), from the pour to `via_to` (default the ground plane);
        without vias the pour floats. Needs shapely."""
        if isinstance(region, (tuple, list)):
            from shapely.geometry import box

            region = box(*region)
        self.pours.append(Pour(region, clearance, via_pitch, via_drill, via_pad, via_offset,
                               fill_pitch, layer, via_to))
        self._touch()
        self._pour_cache = None
        return self

    def layer_names(self):
        """Metal layers used, in order of first appearance."""
        names = list(self.rect_layers) + list(self.poly_layers) + [p.layer for p in self.pours]
        names += [n for v in self.vias for n in v.pad_layers()]
        names += [p.via_to for p in self.pours if p.via_to != "gnd" and (p.via_pitch or p.fill_pitch)]
        return [n for n in dict.fromkeys(names) if n not in self.planes]

    def resolved_pours(self):
        """[(geometry, vias), ...] of every pour against the current copper
        of its layer."""
        key = (len(self.rects), len(self.polys), len(self.vias), len(self.pours))
        cache = getattr(self, "_pour_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        out = []
        if self.pours:
            import shapely
            from shapely.geometry import Point, box

            for p in self.pours:
                sig = [box(*r) for k, (r, n) in enumerate(zip(self.rects, self.rect_layers))
                       if n == p.layer and k not in self.ground_rects]
                sig += [g for k, (g, n) in enumerate(zip(self.polys, self.poly_layers))
                        if n == p.layer and k not in self.ground_polys]
                gnd_xy = {v.xy for pt in self.ports for v in pt.ground_vias()}
                sig += [Point(v.xy).buffer(v.pad / 2, quad_segs=8) for v in self.vias
                        if p.layer in v.pad_layers() and v.xy not in gnd_xy]
                out.append(p.resolve(shapely.union_all(sig) if sig else None))
        self._pour_cache = (key, out)
        return out

    def all_vias(self):
        """Explicit vias plus the stitching vias of every pour."""
        return list(self.vias) + [v for _, vs in self.resolved_pours() for v in vs]

    def add_polygon(self, geom, layer="top"):
        """Add arbitrary copper (shapely Polygon/MultiPolygon, metres). Axis-
        aligned edges become grid lines; other edges are meshed as a
        staircase at the grid resolution."""
        if not geom.is_empty:
            self.polys.append(geom)
            self.poly_layers.append(layer)
            self._touch()
        return self

    def _poly_union(self, layer=None):
        """Union of polygons and pours of one layer (all layers for None);
        via pads are handled in contains()."""
        resolved = self.resolved_pours()
        if not hasattr(self, "_unions") or getattr(self, "_unions_src", None) is not resolved:
            self._unions, self._unions_src = {}, resolved
        if layer in self._unions:
            return self._unions[layer]
        geoms = [g for g, n in zip(self.polys, self.poly_layers) if layer is None or n == layer]
        geoms += [g for (g, _), p in zip(resolved, self.pours) if layer is None or p.layer == layer]
        u = None
        if geoms:
            import shapely

            u = shapely.union_all(geoms)
            shapely.prepare(u)
        self._unions[layer] = u
        return u

    def add_port(self, name, ref, direction, width, length, **kw):
        p = Port(name, tuple(ref), direction, width, length, **kw)
        self.ports.append(p)
        if p.aligned:
            self.add_rect(*p.rect(), layer=p.layer)
        else:
            from shapely.geometry import Polygon

            self.add_polygon(Polygon(p.corners()), layer=p.layer)
        # GCPW: side grounds are ground copper (pours join them instead of
        # clearing around them) with their fence vias
        for c in p.ground_strips():
            if p.aligned:
                xs, ys = zip(*c)
                self.ground_rects.add(len(self.rects))
                self.add_rect(min(xs), min(ys), max(xs), max(ys), layer=p.layer)
            else:
                from shapely.geometry import Polygon

                self.ground_polys.add(len(self.polys))
                self.add_polygon(Polygon(c), layer=p.layer)
        for v in p.ground_vias():
            self.vias.append(v)
        self._touch()
        return p

    def bbox(self):
        r = [tuple(q) for q in self.rects]
        u = self._poly_union()
        if u is not None:
            r.append(u.bounds)
        for v in self.all_vias():
            r.append((v.xy[0] - v.pad / 2, v.xy[1] - v.pad / 2,
                      v.xy[0] + v.pad / 2, v.xy[1] + v.pad / 2))
        r = np.array(r)
        return r[:, 0].min(), r[:, 1].min(), r[:, 2].max(), r[:, 3].max()

    def contains(self, x, y, layer=None):
        """Points inside the copper of `layer` (None: of any layer)."""
        inside = np.zeros(np.broadcast(x, y).shape, bool)
        for (x0, y0, x1, y1), n in zip(self.rects, self.rect_layers):
            if layer is None or n == layer:
                inside |= (x > x0) & (x < x1) & (y > y0) & (y < y1)
        u = self._poly_union(layer)
        if u is not None:
            import shapely

            inside |= shapely.contains_xy(u, x, y)
        # via pads are round copper that adds no grid lines
        for v in self.all_vias():
            if layer is None or layer in v.pad_layers():
                inside |= (x - v.xy[0]) ** 2 + (y - v.xy[1]) ** 2 < (v.pad / 2) ** 2
        return inside

    def polygon_grid_coords(self, tol):
        """x of vertical and y of horizontal polygon edges (all layers)."""
        xs, ys = [], []
        u = self._poly_union()
        if u is None:
            return xs, ys
        for p in getattr(u, "geoms", [u]):
            if p.geom_type != "Polygon":
                continue
            for ring in [p.exterior, *p.interiors]:
                c = np.asarray(ring.coords)
                d = np.diff(c, axis=0)
                xs.extend(c[:-1][np.abs(d[:, 0]) < tol, 0])
                ys.extend(c[:-1][np.abs(d[:, 1]) < tol, 1])
        return xs, ys

    def mesh(self, max_cell, edge_cell=None, snap=None, kind="grid", min_angle=20.0, max_aspect=5.0, like=None,
             size=None):
        """Mesh the layout.

        max_cell   : largest cell size [m]
        edge_cell  : if given, resolve the edge current singularity with
                     lines this far inside (grid: and outside) every edge
        snap       : grid: polygon edge coordinates closer than this to an
                     existing grid line are merged into it (default max_cell/4)
        kind       : "grid" (tensor grid, Manhattan staircase) or "tri"
                     (unstructured, exact polygons, local refinement; needs
                     the `triangle` package, see trimesh.py)
        min_angle  : tri: smallest triangle angle [deg] in the interior
        max_aspect : tri: cell aspect ratio in the edge band (edge cells are
                     edge_cell wide and up to max_aspect * edge_cell long)
        size       : tri: graded interior, a callable points (n, 2) [m] -> largest
                     cell size there [m] (capped by max_cell), e.g.
                     trimesh.graded_size(signal_copper, 0.15e-3, 0.8e-3): fine
                     near the lines, coarse in the far pour (decision 38)
        like       : tri: a Mesh of a slightly different version of this layout;
                     its triangles are re-used with the nodes moved to follow
                     the copper (morph.py: smooth S against geometry changes in
                     tuning loops). Falls back to a fresh mesh when the layouts
                     do not correspond; m.morphed tells which.
        """
        if kind == "tri" and like is not None:
            from .morph import morph_mesh

            m = morph_mesh(like, self)
            if m is not None:
                m.morphed = True
                return m
        if kind == "tri":
            m = Mesh.from_layout_tri(self, max_cell, edge_cell, min_angle, max_aspect, size=size)
            m.morphed = False
        elif kind == "grid":
            m = Mesh.from_layout(self, max_cell, edge_cell, snap)
        else:
            raise ValueError("kind must be 'grid' or 'tri'")
        if self.apertures:
            groups = []
            for lo, hi, region, mc, ec in self.apertures:
                sub = Layout()
                sub.add_polygon(region, layer="aperture")
                if ec is None:
                    ec = 0.5 * edge_cell if edge_cell else None
                if mc is None:
                    mc = min(max_cell, 2.5 * edge_cell) if edge_cell else max_cell
                am = sub.mesh(mc, ec, None, kind, min_angle, max_aspect)
                groups.append(dict(planes=(lo, hi), mesh=am))
            m.set_apertures(groups)
        return m


def _grid_lines(fixed, refine, lo, hi, max_cell, tol):
    c = np.unique(np.round(np.concatenate([fixed, refine]) / tol) * tol)
    c = c[(c >= lo - tol) & (c <= hi + tol)]
    out = [c[0]]
    for a, b in zip(c[:-1], c[1:]):
        n = int(np.ceil((b - a) / max_cell - 1e-9))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


class Mesh:
    """Triangles of all metal layers on one shared tensor grid. Each layer
    has its own nodes, so RWG edges never join triangles of different
    layers; `tri_layer` indexes `layer_names`."""

    def __init__(self, nodes, tris, layout=None, xg=None, yg=None, tri_layer=None,
                 layer_names=("top",), port_cuts=None):
        self.nodes = np.asarray(nodes, float)
        self.tris = np.asarray(tris, int)
        self.layout = layout
        self.xg, self.yg = xg, yg
        self.port_cuts = port_cuts or {}  # unstructured: port name -> cut positions s
        self.layer_names = list(layer_names)
        self.tri_layer = (np.zeros(len(self.tris), int) if tri_layer is None
                          else np.asarray(tri_layer, int))
        self._thicken(dict(getattr(layout, "thick", {}) or {}))
        self._geometry()
        self._rwg()
        self.edge_layer = self.tri_layer[self.tp]
        self.planes = list(getattr(layout, "planes", []))
        self._vias(layout.all_vias() if layout is not None else [])

    @classmethod
    def from_layout(cls, layout, max_cell, edge_cell=None, snap=None):
        x0, y0, x1, y1 = layout.bbox()
        tol = 1e-9 * max(x1 - x0, y1 - y0)
        fx = [c for r in layout.rects for c in (r[0], r[2])]
        fy = [c for r in layout.rects for c in (r[1], r[3])]
        for p in layout.ports:
            if not p.aligned:
                raise ValueError(f"port {p.name}: oblique feeds need mesh(kind='tri')")
            (fx if p.axis == 0 else fy).extend([p.coord(0), p.coord(p.s_gap)])
        for p in layout.lumped:  # internal ports: their gap is a grid line
            if not p.aligned:
                raise ValueError(f"lumped {p.name}: oblique bridges need mesh(kind='tri')")
            (fx if p.axis == 0 else fy).extend([p.coord(0), p.coord(p.s_gap), p.coord(p.length)])
        # polygon edges: merged into existing lines if closer than snap
        snap = 0.25 * max_cell if snap is None else snap
        px, py = layout.polygon_grid_coords(1e3 * tol)
        for f, extra in ((fx, px), (fy, py)):
            for c in sorted(extra):
                if not f or np.min(np.abs(np.asarray(f) - c)) > snap:
                    f.append(c)
        rx, ry = [], []
        if edge_cell:
            rx = [c + d for c in fx for d in (-edge_cell, edge_cell)]
            ry = [c + d for c in fy for d in (-edge_cell, edge_cell)]
            # only keep refinement lines that don't create slivers
            rx = [c for c in rx if np.min(np.abs(np.array(fx) - c)) > 0.5 * edge_cell]
            ry = [c for c in ry if np.min(np.abs(np.array(fy) - c)) > 0.5 * edge_cell]
        xg = _grid_lines(fx, rx, x0, x1, max_cell, tol)
        yg = _grid_lines(fy, ry, y0, y1, max_cell, tol)

        xc = 0.5 * (xg[1:] + xg[:-1])
        yc = 0.5 * (yg[1:] + yg[:-1])
        XC, YC = np.meshgrid(xc, yc, indexing="ij")
        names = layout.layer_names() or ["top"]
        nxy = len(xg) * len(yg)
        tris, tl = [], []
        for li, name in enumerate(names):
            ii, jj = np.nonzero(layout.contains(XC, YC, name))
            nid = lambda i, j: li * nxy + i * len(yg) + j  # noqa: E731
            a, b = nid(ii, jj), nid(ii + 1, jj)
            c, d = nid(ii + 1, jj + 1), nid(ii, jj + 1)
            # alternate the diagonal in a checkerboard to avoid a directional bias
            flip = (ii + jj) % 2 == 1
            t1 = np.where(flip[:, None], np.stack([a, b, d], 1), np.stack([a, b, c], 1))
            t2 = np.where(flip[:, None], np.stack([b, c, d], 1), np.stack([a, c, d], 1))
            tris += [t1, t2]
            tl += [np.full(2 * ii.size, li)]
        tris = np.concatenate(tris)
        used, inv = np.unique(tris, return_inverse=True)
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        nodes = np.stack([np.tile(X.ravel(), len(names))[used],
                          np.tile(Y.ravel(), len(names))[used]], 1)
        return cls(nodes, inv.reshape(tris.shape), layout, xg, yg, np.concatenate(tl), names)

    @classmethod
    def from_layout_tri(cls, layout, max_cell, edge_cell=None, min_angle=20.0, max_aspect=5.0, size=None):
        """Unstructured mesh (see trimesh.py): structured port feeds (any
        direction), anisotropic edge bands with aspect <= max_aspect,
        constrained Delaunay interior with all angles >= min_angle."""
        from .trimesh import triangulate_layout

        nodes, tris, tl, names, cuts = triangulate_layout(layout, max_cell, edge_cell, min_angle,
                                                          max_aspect, size=size)
        e = edge_cell
        s_edge = min(max_cell, max_aspect * e) if e else max_cell
        raw = dict(nodes=np.array(nodes, float), tris=np.array(tris, int), tri_layer=np.array(tl, int),
                   names=list(names), cuts=dict(cuts), snap=0.05 * min(e or max_cell, s_edge))
        m = cls(nodes, tris, layout, None, None, tl, names, cuts)
        m._raw = raw  # pre-thickening triangulation, for morph.morph_mesh (Layout.mesh(like=...))
        return m

    def quality(self):
        """(smallest angle [deg], largest aspect ratio R / 2r) over all
        triangles; an equilateral triangle has aspect ratio 1."""
        v = self.verts
        a, b, c = (np.linalg.norm(v[:, (i + 1) % 3] - v[:, (i + 2) % 3], axis=1) for i in range(3))
        cosA = np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1)
        cosB = np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1)
        ang = np.degrees(np.arccos(np.stack([cosA, cosB], 1)))
        ang = np.concatenate([ang, 180 - ang.sum(1, keepdims=True)], 1)
        s = 0.5 * (a + b + c)
        ar = a * b * c / (8 * (s - a) * (s - b) * (s - c))
        return float(ang.min()), float(ar.max())

    # -------------------------------------------------------------- geometry
    def _thicken(self, thick):
        """Thick metal (Layout.add_thickness): copy the triangles of each
        thick layer onto its twin level (new nodes, same coordinates and
        vertex order). self.twin_layer[li] = twin index or -1."""
        self.twin_layer = -np.ones(len(self.layer_names) + len(thick), int)
        self.twin_tri = None
        for base, twin in thick.items():
            if base not in self.layer_names:
                raise ValueError(f"thick metal: layer {base!r} has no copper")
            if twin in self.layer_names:
                raise ValueError(f"thick metal: twin level {twin!r} has copper of its own")
            li, lj = self.layer_names.index(base), len(self.layer_names)
            self.layer_names.append(twin)
            T = np.nonzero(self.tri_layer == li)[0]
            used, inv = np.unique(self.tris[T], return_inverse=True)
            if self.twin_tri is None:
                self.twin_tri = -np.ones(len(self.tris), int)
            self.twin_tri = np.r_[self.twin_tri, -np.ones(T.size, int)]
            self.twin_tri[T] = len(self.tris) + np.arange(T.size)
            self.tris = np.vstack([self.tris, len(self.nodes) + inv.reshape(-1, 3)])
            self.nodes = np.vstack([self.nodes, self.nodes[used]])
            self.tri_layer = np.r_[self.tri_layer, np.full(T.size, lj)]
            self.twin_layer[li] = lj
        self.twin_layer = self.twin_layer[:len(self.layer_names)]

    def _geometry(self):
        v = self.nodes[self.tris]  # (Nt,3,2)
        e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        # enforce counter-clockwise ordering
        cw = cross < 0
        self.tris[cw] = self.tris[cw][:, [0, 2, 1]]
        self.verts = self.nodes[self.tris]
        self.area = 0.5 * np.abs(cross)
        self.centroid = self.verts.mean(1)

    def _rwg(self):
        nt = len(self.tris)
        # local edge i is opposite local vertex i
        loc = [(1, 2), (2, 0), (0, 1)]
        pairs = np.concatenate([np.sort(self.tris[:, list(e)], 1) for e in loc])
        tri_id = np.tile(np.arange(nt), 3)
        loc_id = np.repeat(np.arange(3), nt)
        key = pairs[:, 0] * len(self.nodes) + pairs[:, 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = ks[1:] == ks[:-1]
        if np.any(same[1:] & same[:-1]):
            raise ValueError("non-manifold edge (more than two triangles)")
        i0 = order[:-1][same]
        i1 = order[1:][same]
        self.edge_nodes = pairs[i0]
        self.tp, self.vp = tri_id[i0], loc_id[i0]
        self.tm, self.vm = tri_id[i1], loc_id[i1]
        # thick metal: one side-wall basis per copper edge of a thick layer, a
        # "bent RWG" from the boundary triangle on the layer (T+) over the
        # wall to its copy on the twin level (T-); the vertical wall current
        # is added by the solver (wall_impedance)
        nw = 0
        if self.twin_tri is not None:
            single = np.ones(ks.size, bool)
            single[1:] &= ~same
            single[:-1] &= ~same
            bd = order[single]
            bd = bd[self.twin_tri[tri_id[bd]] >= 0]
            nw = bd.size
            self.edge_nodes = np.vstack([self.edge_nodes, pairs[bd]])
            self.tp, self.vp = np.r_[self.tp, tri_id[bd]], np.r_[self.vp, loc_id[bd]]
            self.tm, self.vm = np.r_[self.tm, self.twin_tri[tri_id[bd]]], np.r_[self.vm, loc_id[bd]]
        d = self.nodes[self.edge_nodes[:, 1]] - self.nodes[self.edge_nodes[:, 0]]
        self.edge_len = np.hypot(d[:, 0], d[:, 1])
        self.edge_mid = 0.5 * (self.nodes[self.edge_nodes[:, 0]] + self.nodes[self.edge_nodes[:, 1]])
        self.n_basis = len(self.tp)
        self.wall = np.zeros(self.n_basis, bool)  # side-wall bases (thick metal)
        self.wall[self.n_basis - nw:] = True
        self.n_walls = nw

        # per-triangle view: basis index / sign of the edge opposite vertex i
        self.tri_basis = -np.ones((nt, 3), int)
        self.tri_sign = np.zeros((nt, 3))
        idx = np.arange(self.n_basis)
        self.tri_basis[self.tp, self.vp] = idx
        self.tri_sign[self.tp, self.vp] = 1.0
        self.tri_basis[self.tm, self.vm] = idx
        self.tri_sign[self.tm, self.vm] = -1.0

    def _vias(self, vias):
        """Find, for every via, the triangles inside its hole on every layer
        that has copper there (at least the mesh cell containing its centre).
        assign_levels() then splits vias into vertical segments."""
        self.vias = list(vias)
        self.via_attach = []
        for v in self.vias:
            att = {}
            for li, name in enumerate(self.layer_names):
                t = self._attach(v, li)
                if t.size:
                    att[name] = t
            for n in v.pad_layers():
                if n in self.layer_names and n not in att:
                    x, y = v.xy
                    raise ValueError(f"via at ({x * 1e3:.4g}, {y * 1e3:.4g}) mm: no meshed copper "
                                     f"on layer {n}")
            self.via_attach.append(att)
        # provisional order; Solver re-assigns with the stack order
        self.assign_levels(self.layer_names + self.planes)

    def _attach(self, v, li):
        on = self.tri_layer == li
        c = self.centroid
        x, y = v.xy
        d2 = (c[:, 0] - x) ** 2 + (c[:, 1] - y) ** 2
        t = np.nonzero(on & (d2 < (v.drill / 2) ** 2))[0]
        if t.size == 0:
            t = self._containing_cell(x, y)
            t = t[on[t]]
        return t

    def assign_levels(self, order):
        """Split vias into vertical segments for a stack order (metal layer
        and inner plane names, bottom to top; "gnd" is below all of them).

        Unknown n_basis + k is the current of segment k flowing DOWN. It
        leaves the triangles of its upper layer with a uniform charge
        density and arrives in those of its lower layer (nothing for the
        ground plane or an inner plane): a half-RWG at each end, joined by
        the vertical current (a segment between two planes has no charge at
        all). The rest of each pad is ordinary RWG copper, so the horizontal
        current reaches the via wall. A segment may cross planes (antipads);
        its vertical current then runs through several stack sections."""
        rank = {n: i for i, n in enumerate(order)}
        rank["gnd"] = -1
        self.via_segments = []
        self.frills = []  # antipad frill nodes: one unknown each, after the via segments
        for k, (v, att) in enumerate(zip(self.vias, self.via_attach)):
            for n in (v.top, v.bottom) + v.connect:
                if n not in rank:
                    raise ValueError(f"via layer {n!r} is not in the stack order {order}")
            for n in v.connect:
                if n not in self.planes:
                    raise ValueError(f"via connects to {n!r}, which is not a declared plane")
            hi, lo = sorted((rank[v.top], rank[v.bottom]), reverse=True)
            # split points: attached copper, connected planes, and planes passed
            # through a modelled antipad (a frill node there, see antipad.py)
            frill_at = {n for n in order if lo < rank[n] < hi and n in self.planes
                        and n not in v.connect and v.antipad_at(n) is not None}
            conn = [n for n in order if lo < rank[n] < hi
                    and (n in att or n in v.connect or n in frill_at)]
            conn = [order[hi]] + sorted(conn, key=lambda n: -rank[n]) + [order[lo] if lo >= 0 else "gnd"]
            for a, b in zip(conn[:-1], conn[1:]):
                if a in frill_at:  # the segment below a frill node
                    self.frills[-1]["seg_low"] = len(self.via_segments)
                self.via_segments.append(dict(via=k, top=a, bottom=b, top_tris=att.get(a),
                                              bottom_tris=att.get(b)))
                if b in frill_at:
                    self.frills.append(dict(via=k, plane=b, seg_up=len(self.via_segments) - 1,
                                            seg_low=None, a=0.5 * v.drill, b=0.5 * v.antipad_at(b),
                                            xy=v.xy, nr=FRILL_RINGS))
        self.level_order = list(order)
        self.n_vias = len(self.via_segments)
        self.n_frills = sum(fr["nr"] for fr in self.frills)  # frill unknowns (rings)
        self.via_tris = [g["top_tris"] for g in self.via_segments]
        self._count()

    def _count(self):
        """Unknowns: RWG, via segments, antipad rings, aperture M (in that order)."""
        n = self.n_basis + self.n_vias + self.n_frills
        for g in getattr(self, "ap_groups", []):
            g["off"] = n
            n += g["mesh"].n_basis
        self.n_ap = n - (self.n_basis + self.n_vias + self.n_frills)
        self.n_unknowns = n

    def set_apertures(self, groups):
        """Aperture meshes (Layout.add_aperture): [dict(planes=(lo, hi), mesh=Mesh)]."""
        self.ap_groups = list(groups)
        self._count()

    def via_open_crossings(self):
        """Bool per via segment: it passes a plane level (without connecting to it) at a point inside one of that
        plane's apertures. Such a via runs through an opening, not through a solid plane: the section split of
        `solver.via_impedance` (closed-section TEM radial wave into an infinite plane pair) does not apply to it."""
        segs = getattr(self, "via_segments", [])
        out = np.zeros(len(segs), bool)
        groups = getattr(self, "ap_groups", [])
        if not segs or not groups:
            return out
        rank = {n: i for i, n in enumerate(self.level_order)}
        rank["gnd"] = -1
        for k, g in enumerate(segs):
            v = self.vias[g["via"]]
            hi, lo = sorted((rank[g["top"]], rank[g["bottom"]]), reverse=True)
            for grp in groups:
                if not any(lo < rank.get(n, -2) < hi for n in grp["planes"]):
                    continue
                am = grp["mesh"]
                if _point_in_tris(am.nodes, am.tris, v.xy):
                    out[k] = True
                    break
        return out

    def _containing_cell(self, x, y):
        """Triangles of the grid cell(s) containing (x, y) -- all cells
        touching it if it lies on a grid line, so the attachment stays
        symmetric -- or the containing triangle on a non-tensor mesh."""
        if self.xg is not None:
            def span(g, c):
                tol = 1e-6 * (g[-1] - g[0])
                k = np.nonzero(np.abs(g - c) < tol)[0]
                if k.size:
                    return g[max(k[0] - 1, 0)], g[min(k[0] + 1, len(g) - 1)]
                i = np.searchsorted(g, c) - 1
                return (g[i], g[i + 1]) if 0 <= i < len(g) - 1 else (c, c)

            (x0, x1), (y0, y1) = span(self.xg, x), span(self.yg, y)
            c = self.centroid
            return np.nonzero((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1))[0]
        v = self.verts - np.array([x, y])
        cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]  # noqa: E731
        s = np.stack([cr(v[:, i], v[:, (i + 1) % 3]) for i in range(3)], 1)
        return np.nonzero(np.all(s >= -1e-18, axis=1))[0]  # all layers; _attach filters

    def flow_direction(self):
        """Unit vector of positive RWG current flow across each edge (T+ -> T-)."""
        d = self.centroid[self.tm] - self.centroid[self.tp]
        n = np.linalg.norm(d, axis=1)
        return d / np.where(n > 0, n, 1.0)[:, None]  # side walls (thick metal): 0

    # ----------------------------------------------------------------- ports
    def cut_edges(self, port, s):
        """RWG edges on the transverse cut at outward distance s of a port,
        with signs so that positive current flows toward the circuit."""
        es, _ = port.local(self.nodes[self.edge_nodes])  # (Ne,2) along-feed coordinate
        _, tm = port.local(self.edge_mid)
        tol = 1e-6 * port.width
        on = (np.abs(es[:, 0] - s) < tol) & (np.abs(es[:, 1] - s) < tol)
        on &= np.abs(tm) < port.width / 2
        li = self.layer_names.index(port.layer)
        on &= (self.edge_layer == li) | (self.edge_layer == self.twin_layer[li])  # thick: both sheets
        on &= ~self.wall
        idx = np.nonzero(on)[0]
        sign = np.sign(self.flow_direction()[idx] @ -port.u)  # positive: inward
        return idx, sign

    def internal_port_vectors(self):
        """(N_unknowns x n_internal) delta-gap vectors of the layout's
        internal ports (Layout.add_gap_port / add_via_port): a gap port's
        gap cut (as lumped_vectors, current a -> b), a via port's bottom
        segment (current flowing down into the ground plane)."""
        ips = list(getattr(self.layout, "internal_ports", []))
        names = [q.name for q in getattr(self.layout, "lumped", [])]
        Lm = self.lumped_vectors() if names else None
        out = np.zeros((self.n_unknowns, len(ips)))
        for k, ip in enumerate(ips):
            if ip["kind"] == "gap":
                out[:, k] = Lm[:, names.index(ip["name"])]
            else:
                seg = [i for i, g in enumerate(self.via_segments) if g["via"] == ip["via"] and g["bottom"] == "gnd"]
                if len(seg) != 1:
                    raise ValueError(f"via port {ip['name']}: no via segment ending on the ground plane")
                out[self.n_basis + seg[0], k] = 1.0
        return out

    def lumped_vectors(self):
        """(N_unknowns x n_lumped) signed edge lengths l of each internal
        port's gap cut (delta-gap excitation; port current = l . x)."""
        lps = list(getattr(self.layout, "lumped", []))
        Lm = np.zeros((self.n_unknowns, len(lps)))
        for k, p in enumerate(lps):
            idx, sgn = self.cut_edges(p, p.s_gap)
            if idx.size == 0:
                raise ValueError(f"lumped {p.name}: no mesh edges on its gap")
            Lm[idx, k] = -sgn * self.edge_len[idx]  # current a -> b
        return Lm

    def cut_positions(self, port):
        if self.xg is None:
            if port.name not in self.port_cuts:
                raise ValueError(f"port {port.name}: no structured feed in this mesh")
            return np.sort(self.port_cuts[port.name])
        g = self.xg if port.axis == 0 else self.yg
        s = port.sgn * (g - port.ref[port.axis])
        s = s[(s > -1e-12) & (s <= port.length + 1e-12)]
        return np.sort(s)

    def plot(self, ax=None, values=None, layer=None, **kw):
        """Triangles (of one layer, or all) in mm; `values` per plotted triangle."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        ax = ax or plt.gca()
        sel = (slice(None) if layer is None
               else self.tri_layer == self.layer_names.index(layer))
        pc = PolyCollection(self.verts[sel] * 1e3, edgecolor="k", linewidth=0.2,
                            facecolor="none" if values is None else None, **kw)
        if values is not None:
            pc.set_array(values)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return pc


def _point_in_tris(nodes, tris, p):
    """True if point p (x, y) lies inside (or on the edge of) any triangle of (nodes, tris)."""
    a, b, c = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
    p = np.asarray(p, float)

    def cross(u, v, w):
        return (v[:, 0] - u[:, 0]) * (w[1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[0] - u[:, 0])

    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    tol = 1e-18
    neg = (d1 < -tol) | (d2 < -tol) | (d3 < -tol)
    pos = (d1 > tol) | (d2 > tol) | (d3 > tol)
    return bool(np.any(~(neg & pos)))
