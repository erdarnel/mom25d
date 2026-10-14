"""3D view of the inner-plane transition model (geometry/plane_transition.py).

Writes results/plane_transition_3d_view.html from plane_transition_3d_template.html
(Plotly.js from a CDN, data inlined): mesh of both metal layers at true
height, the planes (L2 with its antipad hole), via barrels and the antipad
ring nodes of the frill model. ~5 s.
"""

import json
import os
import sys

import numpy as np
import triangle
from shapely.geometry import Point, box

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")

from mom25d import Solver, mm  # noqa: E402
from mom25d.antipad import ring_nodes  # noqa: E402
from plane_transition import ANTI, H, layout, stackup  # noqa: E402

st = stackup()
lay = layout()
m = lay.mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm)
sol = Solver(m, st, verbose=False)  # assigns via segments / antipad frills

z = {n: st.level_z(n) / mm for n in ("top", "inner", "L2")}
z["gnd"] = 0.0
layers = []
for li, name in enumerate(m.layer_names):
    sel = np.nonzero(m.tri_layer == li)[0]
    used, inv = np.unique(m.tris[sel], return_inverse=True)
    layers.append(dict(name=name, z=z[name], nodes=np.round(m.nodes[used] / mm, 5).tolist(),
                       tris=inv.reshape(-1, 3).tolist()))

# plane L2: board area minus the antipad hole, triangulated for display
x0, x1 = (-15.5, 15.5)
y0, y1 = (-3.0, 3.0)
hole = Point(0, 0).buffer(ANTI / 2 / mm, quad_segs=24)
plane = box(x0, y0, x1, y1).difference(hole)
ring = np.asarray(hole.exterior.coords)[:-1]
outer = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
V = np.r_[outer, ring]
nO, nR = len(outer), len(ring)
S = [(i, (i + 1) % nO) for i in range(nO)] + [(nO + i, nO + (i + 1) % nR) for i in range(nR)]
t = triangle.triangulate(dict(vertices=V, segments=S, holes=[[0.0, 0.0]]), "pQ")
plane_mesh = dict(nodes=np.round(t["vertices"], 5).tolist(), tris=t["triangles"].tolist())

vias = []
for v in lay.all_vias():
    top, bot = z[v.top], z[v.bottom]
    vias.append(dict(xy=[v.xy[0] / mm, v.xy[1] / mm], r=v.drill / 2 / mm, z0=bot, z1=top,
                     kind="signal" if v.antipad else "stitch"))
rings = []
for fr in m.frills:
    nodes = ring_nodes(fr["a"], fr["b"], fr["nr"]) / mm
    rings.append(dict(xy=[fr["xy"][0] / mm, fr["xy"][1] / mm], z=z[fr["plane"]], r=nodes.tolist()))
ports = [dict(name=p.name, layer=p.layer, z=z[p.layer], ref=[p.ref[0] / mm, p.ref[1] / mm],
              gap=[float(c) / mm for c in p.point(p.s_gap)], w=p.width / mm,
              end=[float(c) / mm for c in p.point(p.length)]) for p in lay.ports]

data = dict(layers=layers, plane=plane_mesh, vias=vias, rings=rings, ports=ports,
            planes=dict(L2=z["L2"], gnd=0.0), bounds=[x0, x1, y0, y1], h=H / mm,
            stats=dict(N=int(m.n_unknowns), nb=int(m.n_basis), nv=int(m.n_vias), nf=int(m.n_frills),
                       nt=int(len(m.tris))))
tpl = open(os.path.join(here, "plane_transition_3d_template.html")).read()
out = os.path.join(RESULTS, "plane_transition_3d_view.html")
open(out, "w").write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"N = {m.n_unknowns} ({m.n_basis} RWG, {m.n_vias} via segments, {m.n_frills} antipad rings); "
      f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
