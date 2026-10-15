"""Power leaking into the parallel-plate cavities between solid planes.

In a section closed by PEC on both sides (height d), only vertical currents
excite the TEM (radial) parallel-plate wave: its field is E_z and H_phi, so
horizontal copper currents (E_x, E_y = 0 for that mode) do not couple. A via
piece of length l carrying I launches the cavity voltage (U = Int E_z dz
across the gap)

    U(rho) = -(w mu0 / 4) l I H0(2)(k rho)

-- the TEM term of `solver.via_impedance`, whose real part is the power the
vias deliver to the (infinite) cavity:

    P      = (w mu0 / 8 d) sum_ab l_a l_b Re(I_a* I_b H0(2)(k rho_ab))
    P_a    = 1/2 Re(I_a* sum_b Z_ab I_b)             (per via, sums to P)
    dP/dphi = w mu0 / (16 pi d) |A(phi)|^2,  A = sum_a l_a I_a exp(j k rho_a . phi^)

Per-via terms can be negative: a stitching via next to a signal via carries
the opposite current and takes power back (it shields). The evanescent TM_n
fields are reactive and store no net power. Antipad apertures (frill
magnetic currents) excite the TEM wave only through their displacement
current, ~ jw eps (pi b^2) V / ln(b/a) -- 1e-4 of a via current here -- and
are left out. With a lossy dielectric (complex k) P is the power launched
into the cavity, part of it absorbed near the vias.

The pattern treats the vias as filaments (self term J0(0) = 1) while P_a uses
the solver's thin-wire self term Re H0(2)(k a) = J0(k a): they differ by
~(k a)^2 / 4 of each via's own power (~0.1 % of P_inc on the test case), which
sets the resolution where stitching vias cancel the signal via.

The cavity has no edges (no resonances): P is what a large board loses to
the plane pair, the power that reaches other circuits and the board edge.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import hankel2

from .constants import MU0


@dataclass
class Cavity:
    """One closed section: the via pieces in it and their leaked power."""

    index: int                # section index in the stack
    z: tuple                  # (z_bottom, z_top) [m]
    k: complex                # TEM wavenumber [1/m]
    segs: np.ndarray          # via segment indices with a piece in the section
    xy: np.ndarray            # (n, 2) via centres [m]
    length: np.ndarray        # (n,) piece lengths [m]
    radius: np.ndarray        # (n,) drill radii [m]
    current: np.ndarray       # (n,) complex via currents [A] (flowing down)
    p_via: np.ndarray = field(default=None)  # (n,) power per via [W]
    omega: float = 0.0

    @property
    def d(self):
        return self.z[1] - self.z[0]

    @property
    def power(self):
        return float(self.p_via.sum())

    def pattern(self, phi):
        """dP/dphi [W/rad] of the outgoing radial wave at angles phi."""
        kr = np.real(self.k)
        u = np.stack([np.cos(phi), np.sin(phi)], -1)
        A = np.exp(1j * kr * (u @ self.xy.T)) @ (self.length * self.current)
        return self.omega * MU0 / (16 * np.pi * self.d) * np.abs(A) ** 2

    def voltage(self, X, Y, rmin=None):
        """Cavity TEM voltage U(x, y) [V] (evanescent TM_n near fields left
        out: exact beyond ~d from a via); points closer than the drill
        radius (or rmin) to a via are clamped to it."""
        U = np.zeros(np.broadcast(X, Y).shape, complex)
        for (x0, y0), l, r, I in zip(self.xy, self.length, self.radius, self.current):
            rho = np.maximum(np.hypot(X - x0, Y - y0), r if rmin is None else rmin)
            U -= self.omega * MU0 / 4 * l * I * hankel2(0, self.k * rho)
        return U


def cavity_leakage(solver, x, freq):
    """Cavities of the stack with the via currents of solution x (one
    column: Solver.solve_currents / SweepResult.wave_currents) at freq.
    Returns [Cavity, ...] for every closed section holding via pieces. Via
    segments that pass a plane inside one of its openings are left out
    (Mesh.via_open_crossings): they are not between solid plates there."""
    m, st = solver.mesh, solver.stack
    x = np.asarray(x).reshape(-1)
    nb, nv = m.n_basis, m.n_vias
    if not nv:
        return []
    Iv = x[nb:nb + nv]
    z = {n: st.level_z(n) for n in list(m.layer_names) + list(getattr(m, "planes", []))}
    z["gnd"] = 0.0
    segs = m.via_segments
    xy = np.array([m.vias[g["via"]].xy for g in segs], float).reshape(-1, 2)
    rad = np.array([m.vias[g["via"]].drill / 2 for g in segs])
    lo = np.array([z[g["bottom"]] for g in segs])
    hi = np.array([z[g["top"]] for g in segs])
    w = 2 * np.pi * freq
    # vias passing a plane inside one of its openings launch no wave between solid plates there (decision 36)
    vo = m.via_open_crossings() if hasattr(m, "via_open_crossings") else None
    open_seg = np.zeros(len(segs), bool) if vo is None or len(vo) != len(segs) else np.asarray(vo, bool)
    out = []
    for s, (zb, zt, k) in enumerate(st.via_sections(freq)):
        if k is None:
            continue
        l = np.clip(hi, zb, zt) - np.clip(lo, zb, zt)
        on = np.nonzero((l > 1e-9 * (zt - zb)) & ~open_seg)[0]
        if not on.size:
            continue
        d = zt - zb
        rho = np.linalg.norm(xy[on, None] - xy[None, on], axis=-1)
        np.fill_diagonal(rho, rad[on])
        Z = w * MU0 / (4 * d) * np.outer(l[on], l[on]) * hankel2(0, k * rho)  # TEM part
        I = Iv[on]
        p = 0.5 * np.real(np.conj(I) * (Z @ I))
        out.append(Cavity(s, (zb, zt), k, on, xy[on], l[on], rad[on], I, p, w))
    return out


# ---------------------------------------------------------------- leakage map
FLOOR_DB = 40.0  # cavity-voltage map range below its maximum


def _copper(mesh):
    """{layer: [[x...], [y...]] outline rings in mm} from the mesh triangles."""
    from shapely import union_all
    from shapely.geometry import MultiPolygon, Polygon

    out = {}
    for li, name in enumerate(mesh.layer_names):
        tri = mesh.tris[mesh.tri_layer == li]
        if not len(tri):
            continue
        g = union_all([Polygon(mesh.nodes[t]) for t in tri]).buffer(1e-9).buffer(-1e-9).simplify(1e-7)
        rings = []
        for p in (g.geoms if isinstance(g, MultiPolygon) else [g]):
            for r in [p.exterior, *p.interiors]:
                xy = np.asarray(r.coords) / 1e-3
                rings.append([np.round(xy[:, 0], 4).tolist(), np.round(xy[:, 1], 4).tolist()])
        out[name] = rings
    return out


def leakage_case(solver, result, label, grid_dx=None, margin=2e-3, max_cells=40000, n_phi=181):
    """Data of one case (a solved layout) for `leakage_view`: for every
    frequency of `result` (sweep(..., keep_currents=True)) and every port
    driven with a matched incident wave (0.5 W), the cavity power per via,
    the cavity voltage map, the angular pattern and the power budget."""
    from .fields import Grid

    m, st = solver.mesh, solver.stack
    S50 = result.renormalize(50.0)
    (x0, y0), (x1, y1) = m.nodes.min(0) - margin, m.nodes.max(0) + margin
    dx = grid_dx or max(np.sqrt((x1 - x0) * (y1 - y0) / max_cells), 0.02e-3)
    g = Grid(float(x0), float(y0), dx, int(np.ceil((x1 - x0) / dx)), int(np.ceil((y1 - y0) / dx)))
    X, Y = np.meshgrid(g.xc, g.yc)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    names = list(result.port_names)
    segs = m.via_segments
    vias = []
    for k, sg in enumerate(segs):
        v = m.vias[sg["via"]]
        vias.append(dict(xy=[round(v.xy[0] / 1e-3, 4), round(v.xy[1] / 1e-3, 4)], r=v.drill / 2 / 1e-3,
                         span=f"{sg['top']}-{sg['bottom']}",
                         kind="signal" if (sg["top_tris"] is not None and sg["bottom_tris"] is not None)
                         or getattr(v, "antipad", None) else "ground"))
    secs = st.via_sections(result.freqs[0])
    level_at = {}
    for n in list(m.layer_names) + list(getattr(m, "planes", [])) + ["gnd"]:
        level_at[n] = 0.0 if n == "gnd" else st.level_z(n)

    def sec_name(zb, zt):
        lo = [n for n, z in level_at.items() if abs(z - zb) < 1e-9 and (n == "gnd" or n in st.planes)]
        hi = [n for n, z in level_at.items() if abs(z - zt) < 1e-9 and n in st.planes]
        return f"{(lo or ['z=%.3f' % (zb / 1e-3)])[0]} / {(hi or ['z=%.3f' % (zt / 1e-3)])[0]}"

    cav_names = {s: sec_name(zb, zt) for s, (zb, zt, k) in enumerate(secs) if k is not None}
    freqs = []
    for fi, f in enumerate(result.freqs):
        exc = []
        for p in range(len(names)):
            a = np.zeros(len(names)); a[p] = 1.0
            x = result.wave_currents(fi, a)
            cav = cavity_leakage(solver, x, f)
            pinc = 0.5
            vinc = np.sqrt(2 * 50.0 * pinc)
            budget = {("reflected" if q == p else f"to {names[q]}"): float(abs(S50[fi, q, p]) ** 2)
                      for q in range(len(names))}
            cavs = []
            for c in cav:
                U = np.abs(c.voltage(X, Y)) / vinc
                pv = np.zeros(len(segs)); pv[c.segs] = c.p_via / pinc
                cavs.append(dict(name=cav_names[c.index], power=c.power / pinc, p_via=np.round(pv, 7).tolist(),
                                 z=[round(c.z[0] / 1e-3, 6), round(c.z[1] / 1e-3, 6)],
                                 umax=float(U.max()), u=_enc_db(U, U.max()),
                                 pattern=np.round(c.pattern(phi) / pinc, 8).tolist()))
                budget[f"cavity {cav_names[c.index]}"] = c.power / pinc
            budget["other (radiation, conductor, dielectric)"] = 1.0 - sum(budget.values())
            iv = np.abs(x[m.n_basis:m.n_basis + m.n_vias]) / np.sqrt(2 * pinc / 50.0)
            exc.append(dict(cavities=cavs, budget=budget, i_via=np.round(iv, 5).tolist()))
        freqs.append(exc)
    return dict(label=label, ports=names, f=(result.freqs / 1e9).round(6).tolist(), vias=vias,
                copper=_copper(m), grid=dict(x0=g.x0 / 1e-3, y0=g.y0 / 1e-3, dx=dx / 1e-3, nx=g.nx, ny=g.ny),
                data=freqs, phi=np.round(np.degrees(phi), 3).tolist(), scene=_scene(solver, g))


def _scene(solver, g):
    """3-D geometry at true scale (mm): copper triangles per layer, inner planes with their antipad holes, the
    ground under the grid area, via segments; from view3d.Scene (same display z: solver z, mm)."""
    from shapely.geometry import box

    from .view3d import Scene

    sc = Scene(solver, outline=box(g.x0, g.y0, g.x0 + g.nx * g.dx, g.y0 + g.ny * g.dx), plane_dmax=None)
    layers = [dict(name=l["name"], z=l["z"], color=l["color"], nodes=l["nodes"], tris=l["tris"]) for l in sc.layers]
    return dict(layers=layers, planes=[dict(name=p["name"], z=p["z"], mesh=p["mesh"]) for p in sc.planes],
                pmeshes=sc.pmeshes, board=dict(nodes=sc.board["nodes"], tris=sc.board["tris"]),
                segs=[dict(xy=v["xy"], r=v["r"], z0=v["z0"], z1=v["z1"]) for v in sc.segs])


def _enc_db(v, vmax, floor=FLOOR_DB):
    """dB re vmax in [-floor, 0] -> uint8 (0 = floor, 250 = 0 dB), base64, rows = y."""
    import base64

    db = 20 * np.log10(np.maximum(v, 1e-30) / vmax)
    q = np.clip(np.round((db + floor) / floor * 250), 0, 250).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()


def leakage_view(cases, out, title="Parallel-plate leakage map", subtitle="", notes=()):
    """Write the self-contained leakage-map page for `cases`
    ([leakage_case(...), ...]) to `out`; returns the path."""
    import json
    import os

    try:
        from plotly.offline import get_plotlyjs
        js = "<script>" + get_plotlyjs() + "</script>"
    except ImportError:  # pragma: no cover - plotly not installed
        js = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
    tpl = open(os.path.join(os.path.dirname(__file__), "templates", "leakage.html")).read()
    data = dict(title=title, subtitle=subtitle, notes=list(notes), floor=FLOOR_DB, cases=cases)
    html = tpl.replace("__PLOTLY__", js).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(out, "w") as f:
        f.write(html)
    return out
