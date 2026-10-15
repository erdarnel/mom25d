"""Parallel-plate leakage map of the inner-plane transition (geometry/plane_transition.py).

Two cases: as built (four stitching vias L2 -> gnd) and without stitching. For
1.5-15 GHz and each driven port: power the vias launch into the L2 / gnd cavity
(per via, total, angular pattern), the cavity voltage map and the power budget.
Writes results/leakage_map.html (self-contained, dark) and results/leakage_cases.json
(--replot rebuilds the page from it). ~2 min.
"""

import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "geometry"), os.path.join(here, "..", "..", "..", "Engine")]
RESULTS = os.path.join(here, "..", "results")

from mom25d import GHz, Solver, mm  # noqa: E402
from mom25d.leakage import leakage_case, leakage_view  # noqa: E402
from plane_transition import ANTI, DRILL, PAD, WI, WT, stackup  # noqa: E402
from mom25d import Layout  # noqa: E402

FREQS = np.arange(1.5, 15.01, 1.5) * GHz


def layout(stitch):
    lay = Layout()
    lay.add_plane("L2")
    lay.add_rect(-3 * mm, -WT / 2, 0, WT / 2, layer="top")
    lay.add_rect(0, -WI / 2, 3 * mm, WI / 2, layer="inner")
    lay.add_port("P1", (-3 * mm, 0), "-x", WT, 12 * mm, layer="top")
    lay.add_port("P2", (3 * mm, 0), "+x", WI, 12 * mm, layer="inner")
    lay.add_via(0, 0, drill=DRILL, pad=PAD, top="top", bottom="inner", antipad=ANTI)
    for a in np.pi / 4 + np.arange(4 if stitch else 0) * np.pi / 2:
        lay.add_via(1 * mm * np.cos(a), 1 * mm * np.sin(a), drill=DRILL, pad=PAD, top="L2", bottom="gnd")
    return lay


os.makedirs(RESULTS, exist_ok=True)
CACHE = os.path.join(RESULTS, "leakage_cases.json")
if "--replot" in sys.argv:
    cases = json.load(open(CACHE))
else:
    cases = []
    for label, stitch in (("as built: 4 stitching vias", True), ("no stitching vias", False)):
        sol = Solver(layout(stitch).mesh(max_cell=0.3 * mm, edge_cell=0.05 * mm), stackup(), verbose=False)
        res = sol.sweep(FREQS, keep_currents=True)
        cases.append(leakage_case(sol, res, label, margin=3 * mm))
        tot = [sum(c["power"] for c in fr[0]["cavities"]) for fr in cases[-1]["data"]]
        print(f"{label}: cavity power, P1 driven: " + ", ".join(f"{100 * t:.2f} %" for t in tot), flush=True)
    json.dump(cases, open(CACHE, "w"), separators=(",", ":"))

out = leakage_view(cases, os.path.join(RESULTS, "leakage_map.html"),
                   title="Inner-plane transition: power leaking into the L2 / gnd parallel plate",
                   subtitle="MoM2.5 (mom25d). Top microstrip W 0.56 mm -> via (0.3 mm drill, 0.7 mm antipad) -> "
                            "stripline W 0.24 mm; 3 x 0.254 mm, eps_r 3.66, lossless",
                   notes=["Cavity power = real part of the TEM (radial-wave) via coupling: what the via currents "
                          "launch into an infinite plane pair. Horizontal copper currents do not excite that mode; "
                          "antipad aperture fields are left out (their displacement current is ~1e-4 of a via current).",
                          "Per-via shares add up to the cavity total. Negative shares (blue): the via carries current "
                          "opposite to the signal via and takes power back -- it shields.",
                          "Map: TEM cavity voltage |U| = |Int E_z dz| re the incident line voltage (0.5 W into 50 ohm); "
                          "the evanescent TM_n near fields within ~one cavity height of a via are not drawn.",
                          "'other' = 1 - |S|^2 - cavity: radiation from the open top, plus conductor and dielectric "
                          "loss when present.",
                          "No board edges: on a real board this power reaches the edges and other circuits, "
                          "and the cavity resonates.",
                          "Resolution ~0.1 % of P_inc per via (thin-wire self term, see leakage.py). With the four "
                          "stitching vias the cavity total is within +-0.02 % of zero against 0.7-7 % without them; "
                          "there the individual shares (signal -0.10 %, each stitching via +0.02 % at 15 GHz) are at "
                          "the resolution and their signs carry no meaning. A slightly negative total also reflects "
                          "the antipad frills' own TEM excitation, which is left out."])
print(out)
