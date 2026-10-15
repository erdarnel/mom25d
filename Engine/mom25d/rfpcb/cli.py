"""python -m mom25d.rfpcb <board.json> <command> [options]

  check                      ports, sections, ground, vias, unknown counts (no solve) + 3-D model pages
  circuit                    circuit model (quasi-static lines + parts) -> <out>/circuit.s2p, response page
  sections --solve           references + sections (dense; run it through simmgr: `submit sections`)
  sections --post            de-embed, cascade with the MoM sections -> response page
  complete --solve --freqs 27,29,31   complete model, H-matrix, currents + waves kept (through simmgr)
  complete --post            complete model at its solved frequencies -> response page
  submit sections|complete [--freqs ...] [--need-gb G] [--est-min M] [--dry]   queue in simmgr
"""
import sys
from pathlib import Path

import numpy as np

from . import config


def _db(v):
    return 20 * np.log10(np.maximum(np.abs(v), 1e-12))


def check(cfg):
    from .geometry import Board
    from .place import Placer
    from .viewers import model_page, index_page
    b = Board(cfg)
    pl = Placer(b)
    st = b.stack.mom(False, pl.plane)
    pages = []
    print(f"{len(b.metal.geoms) if hasattr(b.metal, 'geoms') else 1} copper pieces, {len(b.vias)} vias, "
          f"{len(b.gaps)} gap ports, {len(b.plines)} port lines")
    items = [(s, None, s["name"]) for s in b.sections()] + [(b.complete(), cfg.m["complete_band"], "complete")]
    for s, band, nm in items:
        lay, mesh = pl.build(s, band=band)
        print(f"{nm}: ports {list(zip(s['labels'], s['kinds']))}, ground {s['gnd'].area:.2f} mm2, "
              f"{len(s['vias'])} vias, N = {mesh.n_unknowns}")
        pages.append((nm, model_page(lay, mesh, st, cfg.out / "models" / f"{nm}.html", f"{cfg['name']}: {nm}",
                                     origin=s["origin"], notes=[f"N = {mesh.n_unknowns}"]), f"N = {mesh.n_unknowns}"))
    print(index_page(pages, cfg.out / "models" / "index.html", f"{cfg['name']} — models"))


def circuit(cfg):
    from .circuit import Circuit, write_s2p
    from .geometry import Board
    from .viewers import response_page
    ci = Circuit(Board(cfg))
    S = ci.chain()
    write_s2p(cfg.out / "circuit.s2p", ci.F, np.transpose(S, (2, 0, 1)), f"{cfg['name']}: circuit model")
    k0 = int(np.argmin(abs(ci.F - cfg["f0"] * 1e9)))
    print(f"circuit S21 at {cfg['f0']} GHz: {_db(S[1, 0, k0]):.2f} dB")
    print(response_page(cfg.out / "circuit_response.html", f"{cfg['name']} — circuit model", ci.F,
                        [("RF circuit, quasi-static lines", S, "#ffb347")], band=cfg["band"]))
    return ci, S


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__); return
    cfg = config.load(argv[0])
    cmd, opts = argv[1], argv[2:]
    if cmd == "check":
        check(cfg)
    elif cmd == "circuit":
        circuit(cfg)
    elif cmd in ("sections", "complete"):
        from .models import Project
        from .viewers import response_page
        pr = Project(cfg)
        if "--solve" in opts and cmd == "sections":
            pr.solve_sections()
        if "--solve" in opts and cmd == "complete":
            fr = [float(v) for v in opts[opts.index("--freqs") + 1].split(",")] if "--freqs" in opts else [cfg["f0"]]
            pr.solve_complete(np.array(fr) * 1e9)
        if "--post" in opts:
            if list(pr.out.glob("sec_*.npz")):
                F, Sq, Sm, fits = pr.post_sections()
                curves = [("RF circuit, quasi-static lines", Sq, "#ffb347"), ("MoM2.5 (mom25d) sections", Sm, "#4fc3f7")]
            else:
                from .circuit import Circuit
                ci = Circuit(pr.b)
                F, Sq, Sm, fits = ci.F, ci.chain(), None, {}
                curves = [("RF circuit, quasi-static lines", Sq, "#ffb347")]
            pts = []
            if cmd == "complete":
                sec, res = pr.post_complete()
                fs = sorted(res)
                pts = [("MoM2.5 complete model", fs, [res[f] for f in fs])]
                for f in fs:
                    print(f"complete model {f / 1e9:g} GHz: S21 {_db(res[f][1, 0]):.2f} dB")
            k0 = int(np.argmin(abs(F - cfg["f0"] * 1e9)))
            print(f"S21 at {cfg['f0']} GHz: circuit {_db(Sq[1, 0, k0]):.2f} dB"
                  + (f", MoM sections {_db(Sm[1, 0, k0]):.2f} dB" if Sm is not None else ""))
            print(response_page(cfg.out / f"{cmd}_response.html", f"{cfg['name']} — {cmd}", F, curves, band=cfg["band"],
                                points=pts, notes=f"box fits (residual, Z0): {fits}"))
    elif cmd == "submit":
        from .jobs import submit, preflight
        what = opts[0]
        need = float(opts[opts.index("--need-gb") + 1]) if "--need-gb" in opts else (20.0 if what == "complete" else 3.0)
        est = float(opts[opts.index("--est-min") + 1]) if "--est-min" in opts else (60.0 if what == "complete" else 30.0)
        cmdl = [sys.executable, "-m", "mom25d.rfpcb", str(Path(argv[0]).resolve()), what, "--solve"]
        if "--freqs" in opts:
            cmdl += ["--freqs", opts[opts.index("--freqs") + 1]]
        print(preflight())
        print(submit(cfg["name"], f"rfpcb {what}", cmdl, Path(__file__).resolve().parents[1], need, est,
                     cfg.out / "models" / "index.html", dry="--dry" in opts))
    else:
        print(__doc__)
