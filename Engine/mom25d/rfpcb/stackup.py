"""stackup.json (layers top -> bottom: kind mask / cu / diel with t_um, eps, tand; vias with from / to / pad_mm /
dxf_layers) -> the planar stack of a section model and the parameters of the quasi-static cross-section.

Model stack, z = 0 on the ground layer (stack.ground):
    air | mask | cover (signal copper thickness, eps cover_eps: the thin sheet sits at its bottom) | signal sheet |
    dielectrics and inner foils down to the ground, each foil merged into the dielectric BELOW it (a foil voided under
    the line is resin of that layer) | ground plane
An inner foil can be a solid plane instead (model.plane = {"name": ...}): it sits at the foil's top face.
"""
import json

from .engine import mom25d


class Stack:
    def __init__(self, cfg):
        st = cfg["stack"]
        data = json.loads(cfg.path(cfg["stackup"]).read_text())
        L = data["layers"]
        names = [l.get("name") for l in L]
        i_sig, i_gnd = names.index(st["signal"]), names.index(st["ground"])
        self.signal, self.ground = st["signal"], st["ground"]
        self.t_sig = L[i_sig]["t_um"] * 1e-3
        mask = [l for l in L[:i_sig] if l["kind"] == "mask"]
        self.mask = (mask[-1]["t_um"] * 1e-3, mask[-1]["eps"], mask[-1].get("tand", 0.0)) if mask else None
        # layers between signal and ground, top -> bottom
        mid = L[i_sig + 1:i_gnd]
        self.between = mid
        # merge each foil into the dielectric below it
        lam, pend = [], 0.0          # lam: top -> bottom [(t, eps, tand, plane_above_name)]
        foil = None
        for l in mid:
            if l["kind"] == "cu":
                pend += l["t_um"] * 1e-3
                foil = l["name"]
            else:
                lam.append([l["t_um"] * 1e-3 + pend, l["eps"], l.get("tand", 0.0), foil])
                pend, foil = 0.0, None
        if pend:
            raise ValueError("a copper foil directly above the ground layer is not supported")
        self.lam = lam
        self.sigma = st["sigma"]
        self.thin = st["thin"]
        self.cover_eps = st["cover_eps"]
        self.edge_delta = st["edge_delta"] if self.thin else 0.0
        self.tand_min = st["tand_min"]
        self.vias = data.get("vias", [])
        self.plane = (cfg.m.get("plane") or {}).get("name")
        if self.plane and self.plane not in [x[3] for x in lam]:
            raise ValueError(f"plane {self.plane} is not a foil between {self.signal} and {self.ground}")

    def via_layers(self, override=None):
        """{dxf layer: {"pad": mm, "to": 'gnd' | plane name}} for vias starting on the signal layer. A via that ends on
        an inner foil goes to that foil (used when the foil is modelled as a plane, else taken to the ground)."""
        if override:
            return override
        names = [self.signal] + [x[3] for x in self.lam if x[3]] + [self.ground]
        out = {}
        for v in self.vias:
            if v["from"] != self.signal:
                continue
            to = v["to"]
            spec = {"pad": v["pad_mm"], "to": to if to in names[1:-1] else "gnd"}
            for lay in v.get("dxf_layers", []):
                out[lay] = spec
        return out

    def mom(self, lossless=False, plane=None):
        """mom25d Stackup; plane = foil name modelled as a solid plane (None: as resin)."""
        m = mom25d()
        lam = []
        planes = {}
        for t, eps, tand, foil in reversed(self.lam):                       # bottom -> top
            lam.append(m.Dielectric(t * 1e-3, eps, self.tand_min if lossless else tand))
            if plane and foil == plane:
                planes[plane] = len(lam)
        n_sig = len(lam)
        if self.thin:
            lam.append(m.Dielectric(self.t_sig * 1e-3, self.cover_eps, 0.0))
            metals = {self.signal: n_sig}
        else:
            raise NotImplementedError("thick copper (two sheets) is not generalised yet; use stack.thin = true")
        if self.mask:
            t, eps, tand = self.mask
            lam.append(m.Dielectric(t * 1e-3, eps, self.tand_min if lossless else tand))
        return m.Stackup(lam, metals=metals, planes=planes, top="air", sigma=None if lossless else self.sigma,
                         t=self.t_sig * 1e-3)

    def substrate(self):
        """(eps, tand) of the dielectric right under the signal layer (line loss estimate)."""
        return self.lam[0][1], self.lam[0][2]

    def qs2d_params(self):
        """Cross-section parameters for circuit.qs2d: signal over (dielectric, optional voided foil, dielectric)."""
        p = dict(t_l1=self.t_sig, er_sr=self.mask[1] if self.mask else 1.0, t_sr=self.mask[0] if self.mask else 0.0)
        mid = self.between
        diel = [l for l in mid if l["kind"] == "diel"]
        cu = [l for l in mid if l["kind"] == "cu"]
        p["h12"], p["er12"] = diel[0]["t_um"] * 1e-3, diel[0]["eps"]
        if len(diel) >= 2:
            p["h23"], p["er23"] = sum(d["t_um"] for d in diel[1:]) * 1e-3, diel[1]["eps"]
            p["t_l2"] = sum(c["t_um"] for c in cu) * 1e-3
        else:
            p["h23"], p["er23"], p["t_l2"] = 0.0, diel[0]["eps"], 0.0
        return p
