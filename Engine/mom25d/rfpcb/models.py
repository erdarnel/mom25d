"""EM runs of a board: straight reference lines per port type, the sections (or merged groups), the complete model,
and their post-processing (de-embedding, parts connected, cascade).

Outputs go to <out>/mom/: ref_*.npz, sec_*.npz, complete_f*.npz, *_deembedded.sNp, box_fits.json.
"""
import json
import time

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from . import circuit as CI
from . import deembed as DE
from .engine import mom25d
from .geometry import Board, r5
from .place import Placer


def r3(v):
    return round(v, 3)


class Project:
    def __init__(self, cfg):
        self.cfg = cfg
        self.b = Board(cfg)
        self.pl = Placer(self.b)
        self.out = cfg.out / "mom"
        self.out.mkdir(parents=True, exist_ok=True)
        self.F = cfg.freqs_hz()

    # ---------------------------------------------------------------- port types and references
    def port_key(self, sec, i):
        """Port type of port i of a section: (kind, w, g, slot); None for a feed (the engine de-embeds it)."""
        name, p, u, w, g = sec["ports"][i]
        kind = sec["kinds"][i]
        if kind == "feed":
            return None
        if kind == "gap":
            gp = sec["gapports"][name]
            metal = self.b.metal
            from shapely.geometry import Point
            piece = next((q for q in getattr(metal, "geoms", [metal]) if q.buffer(0.005).contains(Point(gp["gnd"]))), None)
            k = "block" if (piece is not None and piece.area < self.cfg.m["land_whole"]) else "pour"
            return (k, r5(w), r5(g), round(gp["slot"], 3))
        return ("pad", r5(w), r5(g), 0.0)

    @staticmethod
    def tag(key):
        kind, w, g, slot = key
        return f"{kind}_w{w:.3f}_g{g:.3f}" + (f"_s{slot:.3f}" if slot else "")

    def ref_void(self, key, L, feed=False):
        """Plane void of a straight reference line (local mm), as at the board's ports of that type: under the line and
        its gaps (+ margin); through the pads at pad ports and under the whole feed for feed lines; `part_port_void`
        mm past the port line at drawn gap ports (the void typically ends inside the part's footprint). None without
        a plane."""
        if not self.pl.plane:
            return None
        pl = self.cfg.m.get("plane") or {}
        kind, w, g = key[0], key[1], key[2]
        h = w / 2 + g + pl.get("void_margin", 0.05)
        ext = self.cfg.m["feed_length"] + 1.0 if (feed or kind == "pad") else pl.get("part_port_void", 0.30)
        return box(-L / 2 - ext, -h, L / 2 + ext, h)

    def feed_ref(self, key):
        """GCPW-feed reference line (gamma, Z0). With a plane (apertures) the feed's line voltage is not valid
        (engine decision 33): the far end is then a pad gap port and Z0 comes from reciprocity with it."""
        w, g = key[1], key[2]
        self.pl.void_override = self.ref_void(key, 3.0, feed=True)
        return self.straight(w, g, 3.0, "feed+pad" if self.pl.plane else "feed")

    def straight(self, w, g, L, ports):
        """Straight line along x, length L between the port lines, ground band either side; ports 'feed' / 'pad' /
        'feed+pad' (feed at P1, pad gap port at P2)."""
        m = mom25d()
        band = self.cfg.m["ground_band"]
        rf = box(-L / 2, -w / 2, L / 2, w / 2)
        y = w / 2 + g + (band - g) / 2
        vias = [(x, sy * y, 0.152, 0.305, "", "gnd") for x in np.arange(-L / 2 + 0.2, L / 2 - 0.05, 0.4) for sy in (1, -1)]
        gnd = unary_union([box(-L / 2, w / 2 + g, L / 2, w / 2 + band), box(-L / 2, -w / 2 - band, L / 2, -w / 2 - g)])
        ends = (("P1", np.array([-L / 2, 0.0]), np.array([-1.0, 0.0])), ("P2", np.array([L / 2, 0.0]), np.array([1.0, 0.0])))
        pads, gp, fd = [], [], []
        for name, p, u in ends:
            if ports == "feed" or (ports == "feed+pad" and name == "P1"):
                fd.append((name, p, u, w, g))
            else:
                pad, slot, s0, r0, pv = self.pl.port_geom(p, u, w, g)
                pads.append((pad, pv)); gp.append((name, s0, r0, w))
        lay = m.Layout()
        self.pl.place(lay, [rf], gnd, vias, (0.0, 0.0), gp, feeds=fd, pads=pads)
        return lay

    def gap_case(self, w, g, slot, L, joined):
        """Straight line with drawn-type gap ports: the line end bridged across `slot` to a ground block joined to the
        coplanar ground (pour type) or a separate stitched block (block type)."""
        m = mom25d()
        M = self.cfg.m
        B = M["ground_band"]
        rf = box(-L / 2, -w / 2, L / 2, w / 2)
        gnd = [box(-L / 2, w / 2 + g, L / 2, w / 2 + B), box(-L / 2, -w / 2 - B, L / 2, -w / 2 - g)]
        y = w / 2 + g + 0.25
        vias = [(x, sy * yy) for x in np.arange(-L / 2 + 0.2, L / 2 - 0.05, 0.4) for sy in (1, -1) for yy in (y, y + 0.4)]
        gp = []
        for name, sx in (("P1", -1), ("P2", 1)):
            xe = sx * L / 2
            x0, x1 = sorted((xe + sx * slot, xe + sx * (slot + M["gap_keep"])))
            H = w / 2 + B
            if joined:
                a, b = sorted((xe, xe + sx * slot))
                gnd += [box(x0, -H, x1, H), box(a, w / 2 + g, b, H), box(a, -H, b, -w / 2 - g)]
            else:
                gnd.append(box(x0, -(w / 2 + g), x1, w / 2 + g))
            xc = 0.5 * (x0 + x1)
            vias += [(xc, -0.15), (xc, 0.0), (xc, 0.15)] if not joined else [(xc, -0.3), (xc, 0.0), (xc, 0.3)]
            gp.append((name, np.array([xe - sx * M["bridge_in"], 0.0]), np.array([xe + sx * (slot + M["bridge_in"]), 0.0]), w))
        lay = m.Layout()
        ground = unary_union(gnd).difference(rf.buffer(min(g, slot) - 0.01))
        self.pl.place(lay, [rf], ground, [(x, yy, 0.152, 0.305, "", "gnd") for x, yy in vias], (0.0, 0.0), gp)
        return lay

    def ref_layout(self, key, L):
        kind, w, g, slot = key
        self.pl.void_override = self.ref_void(key, L)
        if kind == "pad":
            return self.straight(w, g, L, "pad")
        return self.gap_case(w, g, slot, L, joined=(kind == "pour"))

    def sections(self):
        return self.b.sections()

    def jobs(self, include_refs=True, lengths=None):
        """[(name, layout builder, origin, note)] of a sections run: references per port type, then the sections."""
        secs = self.sections()
        keys = sorted({self.port_key(s, i) for s in secs for i in range(len(s["ports"]))} - {None})
        Ls = lengths or self.cfg["ref_lengths"]
        jobs = []
        if include_refs:
            for key in keys:
                w, g = key[1], key[2]
                jobs.append((f"ref_feed_w{w:.3f}_g{g:.3f}", lambda key=key: self.feed_ref(key), (0, 0),
                             f"reference line 3 mm, GCPW feed{' + gap port (Z0 by reciprocity)' if self.pl.plane else 's'}, "
                             f"line {w} / gap {g} mm"))
                for L in Ls:
                    jobs.append((f"ref_{self.tag(key)}_L{L}", lambda key=key, L=L: self.ref_layout(key, L), (0, 0),
                                 f"reference line {L} mm, gap ports '{key[0]}' type"))
        for s in secs:
            jobs.append((f"sec_{s['name']}", lambda s=s: (setattr(self.pl, "void_override", None), self.pl.build(s)[0])[1],
                         tuple(s["origin"]), f"section {s['name']}"))
        seen = set()
        return [j for j in jobs if not (j[0] in seen or seen.add(j[0]))]

    # ---------------------------------------------------------------- solving
    def mesh(self, lay):
        M = self.cfg.m
        return lay.mesh(M["max_cell"] * 1e-3, edge_cell=M["edge_cell"] * 1e-3, kind="tri")

    def run(self, name, lay, freqs=None, method="dense", keep=False, lossless=False):
        m = mom25d()
        p = self.out / f"{name}.npz"
        if p.exists():
            print(f"{name}: done already", flush=True)
            return p
        mesh = self.mesh(lay)
        print(f"{name}: N = {mesh.n_unknowns}", flush=True)
        t0 = time.time()
        r = m.Solver(mesh, self.b.stack.mom(lossless, self.pl.plane), method=method, verbose=False).sweep(
            self.F if freqs is None else freqs, keep_currents=keep, interp=None)
        extra = dict(x=np.asarray(r.currents), waves=np.asarray(r.waves)) if keep else {}
        np.savez(p, f=r.freqs, S=r.S, z0=r.z0, gamma=r.gamma, zline=r.z0_line, ports=np.array(r.port_names),
                 N=mesh.n_unknowns, seconds=time.time() - t0, **extra)
        print(f"PROGRESS-ITEM {name} {time.time() - t0:.0f} s", flush=True)
        return p

    def solve_sections(self):
        jobs = self.jobs()
        for i, (name, mk, o, note) in enumerate(jobs, 1):
            self.run(name, mk())
            print(f"PROGRESS {i}/{len(jobs)}", flush=True)

    def solve_complete(self, freqs, keep=True):
        """Complete model, H-matrix solver, one file per frequency (currents + port waves kept)."""
        m = mom25d()
        sec = self.b.complete()
        self.pl.void_override = None
        lay, mesh = self.pl.build(sec, band=self.cfg.m["complete_band"])
        print(f"complete model: N = {mesh.n_unknowns}, {len(sec['vias'])} vias", flush=True)
        sol = m.Solver(mesh, self.b.stack.mom(False, self.pl.plane), method="hmatrix", verbose=True)
        for i, f in enumerate(freqs, 1):
            p = self.out / f"complete_f{f / 1e9:.2f}.npz"
            if p.exists() and (not keep or "waves" in np.load(p).files):
                continue
            t1 = time.time()
            r = sol.sweep(np.array([f]), keep_currents=keep, interp=None)
            extra = dict(x=np.asarray(r.currents[0]), waves=np.asarray(r.waves[0])) if keep else {}
            np.savez(p, f=r.freqs, S=r.S, z0=r.z0, ports=np.array(r.port_names), N=mesh.n_unknowns,
                     seconds=time.time() - t1, **extra)
            print(f"{f / 1e9:.2f} GHz: {time.time() - t1:.0f} s", flush=True)
            print(f"PROGRESS {i}/{len(freqs)}", flush=True)

    # ---------------------------------------------------------------- post
    def box_for(self, key):
        """Error boxes of a port type on the run grid (from its references), max fit residual, median line Z0."""
        w, g = key[1], key[2]
        fd = np.load(self.out / f"ref_feed_w{w:.3f}_g{g:.3f}.npz")
        gam, zl = fd["gamma"][:, 0], np.real(fd["zline"][:, 0])
        Ms = {}
        for L in self.cfg["ref_lengths"]:
            d = np.load(self.out / f"ref_{self.tag(key)}_L{L}.npz")
            Ms[L] = DE.s2a(DE.renorm(d["S"], d["z0"]))
        Xs, r = DE.box_from_refs(Ms, gam, zl)
        return Xs, r, float(np.median(zl))

    def deembedded(self, sec, npz):
        """De-embedded S (nf, n, n) of a section / group / complete model file."""
        d = np.load(npz)
        nm = [str(n) for n in d["ports"]]
        order = [p[0] for p in sec["ports"]]
        S = DE.renorm(d["S"], d["z0"])[:, [nm.index(q) for q in order]][:, :, [nm.index(q) for q in order]]
        keys = [self.port_key(sec, i) for i in range(len(order))]
        boxes = {k: self.box_for(k) for k in set(keys) if k}
        fi = [int(np.argmin(abs(self.F - f))) for f in d["f"]]
        I2 = np.eye(2, dtype=complex)
        return d["f"], np.array([DE.deembed_nport(S[i], [boxes[k][0][fi[i]] if k else I2 for k in keys])
                                 for i in range(len(S))]), boxes

    def group_two_port(self, sec, f, Sd):
        """Connect the parts inside a merged group (between its member lines) -> (nf, 2, 2)."""
        ci = CI.Circuit(self.b)
        ch = self.cfg["chain"]
        out = []
        parts_inside = []
        for k in range(len(sec["members"]) - 1):
            ia = next(i for i, e in enumerate(ch) if e.get("line") == sec["members"][k])
            parts_inside.append(ch[ia + 1])
        P = [np.transpose(ci.part(e, f), (2, 0, 1)) for e in parts_inside]
        for i in range(len(f)):
            cur, idx = Sd[i], [p[0] for p in sec["ports"]]
            for k, Pk in enumerate(P):
                a, b = f"P{2 * k + 2}", f"P{2 * k + 3}"
                cur = DE.terminate(cur, (idx.index(a), idx.index(b)), Pk[i])
                idx = [q for q in idx if q not in (a, b)]
            out.append(cur)
        return np.array(out)

    def post_sections(self):
        """De-embedded section files and the cascade with the MoM sections; returns (F, S_qs, S_mom, fits)."""
        ci = CI.Circuit(self.b)
        lines, fits = {}, {}
        for s in self.sections():
            f, Sd, boxes = self.deembedded(s, self.out / f"sec_{s['name']}.npz")
            fits.update({self.tag(k): (round(v[1], 4), round(v[2], 1)) for k, v in boxes.items()})
            n = Sd.shape[1]
            if n == 2:
                CI.write_s2p(self.out / f"{s['name'].lower()}_deembedded.s2p", f, Sd, f"MoM2.5 (mom25d) {s['name']}, de-embedded")
                S2 = Sd
            else:
                S2 = self.group_two_port(s, f, Sd)
            Si = CI.interp_s(f, np.transpose(S2, (1, 2, 0)), ci.F)
            A = CI.s_to_abcd(Si)
            for mname in s["members"]:
                lines[mname] = A if mname == s["members"][0] else np.broadcast_to(np.eye(2)[:, :, None], A.shape).astype(complex)
            if n > 2:                              # parts inside the group are already connected: mark them ideal
                lines[("__group__", s["members"][0])] = s["members"]
        json.dump(fits, open(self.out / "box_fits.json", "w"), indent=1)
        S_qs = ci.chain()
        S_mom = self._chain_with(ci, lines)
        return ci.F, S_qs, S_mom, fits

    def _chain_with(self, ci, lines):
        groups = {v[0]: v for k, v in lines.items() if isinstance(k, tuple)}
        inside = set()
        for g in groups.values():
            ch = self.cfg["chain"]
            for k in range(len(g) - 1):
                ia = next(i for i, e in enumerate(ch) if e.get("line") == g[k])
                inside.add(ch[ia + 1]["part"])
        seq = []
        for e in self.cfg["chain"]:
            if "line" in e:
                seq.append(lines[e["line"]])
            elif e["part"] not in inside:
                seq.append(CI.s_to_abcd(ci.part(e)))
        return CI.abcd_to_s(CI.cascade(seq))

    def post_complete(self):
        """{f: 2-port S} of the complete model (de-embedded, every part connected)."""
        sec = self.b.complete()
        out = {}
        for p in sorted(self.out.glob("complete_f*.npz")):
            f, Sd, _ = self.deembedded(sec, p)
            out[float(f[0])] = self.group_two_port(sec, f, Sd)[0]
        return sec, out
