"""Board configuration (board.json): paths, DXF layer names, via spans, the cascade chain, port kinds, model and
circuit settings. Every length is in mm, frequencies in GHz. Paths are relative to the board.json folder.

Minimal board.json:

    {
      "name": "my board Tx",
      "dxf": "layout_defeatured.dxf",
      "stackup": "stackup.json",
      "out": "out",
      "stack": {"signal": "L1", "ground": "L3"},
      "vias": {"VIA_DRILL_L1-L3_D152": {"pad": 0.305, "to": "L3"}},
      "chain": [
        {"line": "TL1", "from": "IN", "to": "R1.1"},
        {"part": "R1", "s2p": "parts/jumper.s2p", "label": "0 ohm"},
        {"line": "TL2", "from": "R1.2", "to": "OUT"}
      ],
      "ports": {"OUT": {"kind": "feed"}}
    }

Port kinds (by label): a label on the GAP_PORTS layer is a drawn gap port ("gap"); a label on the PORTS layer is a
gap port to a separate stitched pad ("pad") unless "ports": {label: {"kind": "feed"}} makes it a GCPW feed.
"""
import copy
import json
from pathlib import Path

DEFAULTS = {
    "layers": {"metal": "{sig}_METAL", "not_metal": "{sig}_NOT_METAL", "ports": "PORTS", "gap_ports": "GAP_PORTS",
               "outline": "COMP_OUTLINE", "plane_voids": "{plane}_NOT_METAL"},
    "stack": {"signal": "L1", "ground": "L3", "thin": True, "edge_delta": 0.034, "cover_eps": 1.0, "sigma": 5.8e7,
              "tand_min": 1e-5},
    "model": {
        "ground_band": 3.0,          # L1 ground kept within this of the RF copper
        "ownership": True,           # each ground point belongs to the nearest section's RF copper (Voronoi)
        "drop_islands": True,        # ground pieces without a via / not bordering the line are dropped
        "rf_cut_half": 1.0,          # half-width of the boxes beyond the port planes that cut the RF copper
        "gap_keep": 0.5,             # ground kept beyond a drawn gap port's slot
        "land_whole": 30.0,          # mm2: a gap port landing on a piece smaller than this keeps the piece whole
        "land_half": 0.8,            # lateral extent of a gap port's land beyond the line edge, each side
        "slot": 0.15, "bridge_in": 0.04, "snap_rf": 0.05,
        "pad_depth": 0.30, "pad_side": 0.10, "pad_via": 0.152,
        "feed_length": 5.0, "fence_pitch": 0.40, "fence_drill": 0.152, "fence_pad": 0.305,
        "edge_cell": 0.03, "max_cell": 0.20,
        "plane": None,               # e.g. {"name": "L2", "ap_max": 0.15, "ap_edge": 0.04}: inner plane, DXF voids open;
                                     # also "void_margin" (0.05 mm past each gap) and "part_port_void" (0.30 mm past a
                                     # drawn gap port's line) for the reference lines' voids; validated: openings along
                                     # a line up to 4.5 mm wide (LIMITATIONS.md)
        "merge": [],                 # e.g. [["TL4", "TL5"]]: lines modelled as one (a part with a grounded paddle)
        "complete_band": 5.0,        # ground band of the complete model
    },
    "circuit": {"f_start": 20.0, "f_stop": 40.0, "f_step": 0.05, "roughness": 1.15, "step": 0.01, "bend_keep": 1.0,
                "path": None,        # optional {"points": [[x, y], ...], "bend_r": r}: centreline override
                "qs2d": {}},         # overrides of the quasi-static cross-section (see circuit.qs2d)
    "freqs": {"start": 20.0, "stop": 40.0, "step": 1.0},
    "band": [27.5, 31.0],
    "f0": 29.25,
    "ref_lengths": [1.5, 3.0],
    "ports": {},
    "out": "out",
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """board.json merged over DEFAULTS; .path(key) resolves a path relative to the board.json folder."""

    def __init__(self, data, root):
        super().__init__(_merge(DEFAULTS, data))
        self.root = Path(root).resolve()
        sig = self["stack"]["signal"]
        for k in ("metal", "not_metal"):
            self["layers"][k] = self["layers"][k].format(sig=sig)

    def path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def dxf(self):
        return self.path(self["dxf"])

    @property
    def out(self):
        o = self.path(self["out"])
        o.mkdir(parents=True, exist_ok=True)
        return o

    @property
    def m(self):
        return self["model"]

    def lines(self):
        return [e for e in self["chain"] if "line" in e]

    def parts(self):
        return [e for e in self["chain"] if "part" in e]

    def freqs_hz(self):
        import numpy as np
        f = self["freqs"]
        return np.arange(f["start"], f["stop"] + 1e-9, f["step"]) * 1e9

    def check_chain(self):
        ch = self["chain"]
        if not ch or "line" not in ch[0] or "line" not in ch[-1]:
            raise ValueError("chain must start and end with a line")
        for a, b in zip(ch[:-1], ch[1:]):
            if ("line" in a) == ("line" in b):
                raise ValueError("chain must alternate line / part")


def load(path):
    path = Path(path)
    cfg = Config(json.loads(path.read_text()), path.parent)
    cfg.check_chain()
    return cfg
