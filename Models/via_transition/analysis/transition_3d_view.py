"""3D view data for the top -> via -> buried-line transition (two 0.254 mm
RO4350B-like layers, see Engine/tests/test_multilayer.py and the openEMS reference).

Solves at a few frequencies with currents kept, and writes
results/transition_3d_data.json: per layer the triangles (mm, true z) with the nodal
|J| (A/m, area-weighted average of the centroid values) for a unit incident
wave on each port with the other port matched, plus the via current and S.
The viewer page results/transition_3d_view.html is transition_3d_template.html
with this JSON inlined (replacing __DATA__).
Feed lines beyond the delta gaps are cut away (unphysical). ~1.5 min.
"""

import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")

from mom25d import GHz, Solver, mm  # noqa: E402
from via_transition import DRILL, ER, H1, H2, PAD, REF, WI, WT, layout, stackup  # noqa: E402

lay = layout()
mesh = lay.mesh(max_cell=0.2 * mm, edge_cell=0.05 * mm)
st = stackup()
sol = Solver(mesh, st)
freqs = np.array([2.0, 8.0, 14.0]) * GHz
res = sol.sweep(freqs, keep_currents=True)
S = res.renormalize(50)

XMAX = REF + 6 * mm  # show 6 mm of each feed past the reference planes
keep = np.abs(mesh.centroid[:, 0]) < XMAX
wA = np.repeat(mesh.area, 3)


def nodal(v, tl):
    """area-weighted node average of per-triangle values on one layer."""
    on = mesh.tri_layer == tl
    t = mesh.tris[on]
    num = np.bincount(t.ravel(), weights=np.repeat(v[on], 3) * np.repeat(mesh.area[on], 3),
                      minlength=len(mesh.nodes))
    den = np.bincount(t.ravel(), weights=np.repeat(mesh.area[on], 3), minlength=len(mesh.nodes))
    return num / np.maximum(den, 1e-300)


layers = []
for li, name in enumerate(mesh.layer_names):
    on = (mesh.tri_layer == li) & keep
    tris = mesh.tris[on]
    used, inv = np.unique(tris, return_inverse=True)
    layers.append(dict(name=name, z=st.level_z(name) / mm,
                       x=np.round(mesh.nodes[used, 0] / mm, 5).tolist(),
                       y=np.round(mesh.nodes[used, 1] / mm, 5).tolist(),
                       i=inv.reshape(-1, 3)[:, 0].tolist(), j=inv.reshape(-1, 3)[:, 1].tolist(),
                       k=inv.reshape(-1, 3)[:, 2].tolist(), used=used, on=on))

frames = []
for fi, f in enumerate(freqs):
    fr = dict(f_ghz=float(f / GHz), S={}, ports=[])
    for a in range(2):
        for b in range(2):
            fr["S"][f"S{a + 1}{b + 1}"] = [float(S[fi, a, b].real), float(S[fi, a, b].imag)]
    for p in range(2):
        x = res.matched_currents(fi, p)
        J = sol.current_density(x[:, None])[:, :, 0]  # (Nt, 2) complex
        mag = np.sqrt(np.abs(J[:, 0]) ** 2 + np.abs(J[:, 1]) ** 2)
        per = {}
        for li, L in enumerate(layers):
            per[L["name"]] = np.round(nodal(mag, li)[L["used"]], 4).tolist()
        fr["ports"].append(dict(J=per, via=[float(abs(c)) for c in sol.via_currents(x)]))
    frames.append(fr)

data = dict(
    title="Top microstrip -> via -> buried line",
    stack=dict(layers=[dict(t=H1 / mm, er=ER, name="layer 1"), dict(t=H2 / mm, er=ER, name="layer 2")],
               levels={L["name"]: L["z"] for L in layers}),
    lines=dict(top=WT / mm, inner=WI / mm), ref=REF / mm, xmax=XMAX / mm,
    via=dict(x=0.0, y=0.0, r=DRILL / 2 / mm, rpad=PAD / 2 / mm, z0=H1 / mm, z1=(H1 + H2) / mm),
    ports=[dict(name=p.name, layer=p.layer, x=p.ref[0] / mm, w=p.width / mm) for p in lay.ports],
    eeff=np.round(res.eeff, 3).tolist(), z0=np.round(res.z0_line.real, 1).tolist(),
    n_unknowns=int(mesh.n_unknowns),
    layers=[{k: v for k, v in L.items() if k not in ("used", "on")} for L in layers],
    frames=frames,
)
out = os.path.join(RESULTS, "transition_3d_data.json")
with open(out, "w") as fh:
    json.dump(data, fh, separators=(",", ":"))
print("saved", out, f"{os.path.getsize(out) / 1e6:.2f} MB")
tpl = open(os.path.join(here, "transition_3d_template.html")).read()
html = os.path.join(RESULTS, "transition_3d_view.html")
open(html, "w").write(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print("saved", html)
