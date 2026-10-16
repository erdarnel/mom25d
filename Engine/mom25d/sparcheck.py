"""Checks of S-parameter data (Touchstone) before it goes into a circuit or EM cascade.

Measured or vendor data is often slightly non-passive (noise near DC, de-embedding), non-reciprocal, under-sampled
or not consistent with any causal system; connected into a cascade it then shows gain or ripple that is not real.

Data checks (numpy only):
  passivity      largest singular value of S per frequency (passive parts: <= 1)
  reciprocity    max |S_ij - S_ji| (passive parts)
  stability      active 2-ports (amplifiers): Rollett K, |Delta| and mu (Edwards-Sinsky; > 1: unconditionally stable)
  sampling       largest phase step of each transmission term between frequency points (> 90 deg: too coarse to
                 interpolate) and the DC / band coverage
Model checks (vector fitting with scikit-rf, `pip install scikit-rf`):
  causality      a stable rational model (Gustavsen vector fitting) fitted with increasing order; data that a
                 causal model of modest order cannot follow is noisy, under-sampled or not causal. Per term, a
                 causal fit is compared with one allowed anti-causal poles: a term only the latter follows is not
                 causal (seen on de-embedded amplifier reflections)
  passivity      of the fitted model (Hamiltonian test); optionally enforced, and the model written back as a
                 Touchstone file and / or a SPICE subcircuit

    python -m mom25d.sparcheck part.s2p [more.s2p ...] [--band 27.5 31] [--out DIR] [--enforce] [--spice]
                                        [--active auto|yes|no] [--tol 0.01] [--max-poles 60] [--no-fit]

writes DIR/<name>_check.json, a combined DIR/sparcheck.md and a dark-mode DIR/sparcheck.html.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

PASSIVE_TOL = 1e-3          # singular values up to 1 + this count as passive (rounding in the files)


def read(path):
    """(f [Hz], S (nf, n, n), z0) of a Touchstone file."""
    from .rfpcb.circuit import read_touchstone

    f, S, z0 = read_touchstone(path)
    return f, np.moveaxis(S, -1, 0), z0


def _bands(f, mask):
    """[[f0, f1], ...] (Hz) of the runs where mask is true."""
    out, start = [], None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        if (not m or k == len(mask) - 1) and start is not None:
            end = k if m else k - 1
            out.append([float(f[start]), float(f[end])])
            start = None
    return out


def stability(S):
    """Rollett K, |Delta|, mu (load side) per frequency of a 2-port S (nf, 2, 2)."""
    s11, s12, s21, s22 = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]
    d = s11 * s22 - s12 * s21
    K = (1 - abs(s11) ** 2 - abs(s22) ** 2 + abs(d) ** 2) / (2 * abs(s12 * s21) + 1e-30)
    mu = (1 - abs(s11) ** 2) / (abs(s22 - d * np.conj(s11)) + abs(s12 * s21) + 1e-30)
    return K, abs(d), mu


def data_checks(f, S, band=None, active="auto"):
    n = S.shape[1]
    sv = np.linalg.svd(S, compute_uv=False).max(axis=1)
    offdiag = [abs(S[:, i, j]) for i in range(n) for j in range(n) if i != j]
    is_active = (max(float(np.max(x)) for x in offdiag) > 10 ** (1 / 20)) if (active == "auto" and n > 1) else \
        (active == "yes")
    inb = np.ones(len(f), bool) if band is None else (f >= band[0] * 1e9) & (f <= band[1] * 1e9)
    r = dict(n_ports=n, n_freq=len(f), f_min_ghz=float(f[0] / 1e9), f_max_ghz=float(f[-1] / 1e9),
             active=bool(is_active), has_dc=bool(f[0] == 0.0),
             band_covered=None if band is None else bool(f[0] <= band[0] * 1e9 and f[-1] >= band[1] * 1e9))
    steps = [0.0]
    for i in range(n):
        for j in range(n):
            if i != j:      # only where the term matters (> -30 dB re its peak): stopband noise jumps freely
                a = np.abs(S[:, i, j])
                if a.max() < 0.01:          # never above -40 dB: noise
                    continue
                ok = (a[1:] > 0.03 * a.max()) & (a[:-1] > 0.03 * a.max())
                d = np.abs(np.diff(np.unwrap(np.angle(S[:, i, j]))))[ok]
                steps.append(float(np.degrees(d.max())) if len(d) else 0.0)
    r["max_phase_step_deg"] = max(steps)
    if not is_active:
        r["sv_max"] = float(sv.max())
        r["sv_max_f_ghz"] = float(f[sv.argmax()] / 1e9)
        r["sv_max_band"] = float(sv[inb].max()) if inb.any() else None
        r["passivity_violations_ghz"] = [[a / 1e9, b / 1e9] for a, b in _bands(f, sv > 1 + PASSIVE_TOL)]
        r["reciprocity_max"] = float(max((np.max(abs(S[:, i, j] - S[:, j, i])) for i in range(n)
                                          for j in range(i + 1, n)), default=0.0))
    elif n == 2:
        K, D, mu = stability(S)
        r["K_min"], r["mu_min"] = float(K.min()), float(mu.min())
        r["K_min_band"] = float(K[inb].min()) if inb.any() else None
        r["mu_min_band"] = float(mu[inb].min()) if inb.any() else None
        r["potentially_unstable_ghz"] = [[a / 1e9, b / 1e9] for a, b in _bands(f, mu <= 1)]
        r["gain_max_db"] = float(20 * np.log10(abs(S[:, 1, 0]).max()))
    r["_sv"] = sv
    return r


def group_delays(f, S, band=None):
    """Median group delay (s) of each transmission term where it is within 20 dB of its peak (inside the band if
    given). A negative bulk delay is a causality violation (e.g. reference planes moved past the part)."""
    n = S.shape[1]
    tau = np.zeros((n, n))
    inb = np.ones(len(f), bool) if band is None else (f >= band[0] * 1e9) & (f <= band[1] * 1e9)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = np.abs(S[:, i, j])
            gd = -np.gradient(np.unwrap(np.angle(S[:, i, j])), 2 * np.pi * f)
            m = (a > 0.1 * a.max()) & inb
            # terms that never rise above -40 dB (an amplifier's S12) are noise: no delay read from them
            tau[i, j] = float(np.median(gd[m])) if m.sum() > 3 and a.max() > 0.01 else 0.0
    return tau


def fit_range(f, band=None, widen=0.3):
    """Fit range (Hz): the band widened by `widen` x its edges (clipped to the data), else the data without DC."""
    lo, hi = (f[f > 0][0], f[-1]) if band is None else (max(band[0] * 1e9 * (1 - widen), f[0]),
                                                         min(band[1] * 1e9 * (1 + widen), f[-1]))
    return float(lo), float(hi)


def fit(f, S, z0=50.0, tol=0.01, max_poles=60, band=None, frange=None):
    """Vector fit (common poles) with increasing order until every term's rms error is below tol x its rms
    magnitude (or below 1e-3 absolute, the floor of e.g. an amplifier's S12). Each term is normalised by its rms
    (at least 0.03, so noise-floor terms do not steer the poles) before the fit, so a 25 dB S21 does not set the poles alone; the residues are scaled back afterwards, so the
    VectorFitting object describes the real S (passivity test, SPICE export). Returns (VectorFitting, summary)."""
    import warnings

    import skrf
    from skrf.vectorFitting import VectorFitting

    lo, hi = frange or fit_range(f, band)
    m = (f >= lo) & (f <= hi) & (f > 0)
    fm, Sm = f[m], S[m]
    n = S.shape[1]
    rms_t = np.sqrt(np.mean(np.abs(Sm) ** 2, axis=0))                  # (n, n) rms per term
    w = np.maximum(rms_t, 0.03)       # normalise, but not terms near the noise floor (an amplifier's S12)
    ntw = skrf.Network(frequency=skrf.Frequency.from_f(fm, unit="hz"), s=Sm / w[None], z0=z0)
    best = None
    for nc in range(4, max_poles // 2 + 1, 4):
        vf = VectorFitting(ntw)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            vf.vector_fit(n_poles_real=2, n_poles_cmplx=nc)
        wk = w.ravel()
        vf.residues = vf.residues * wk[:, None]
        vf.constant_coeff = vf.constant_coeff * wk
        vf.proportional_coeff = vf.proportional_coeff * wk
        M = model_response(vf, fm, n)
        rms = np.sqrt(np.mean(np.abs(M - Sm) ** 2, axis=0))
        rel = np.where(rms < 1e-3, 0.0, rms / np.maximum(rms_t, 1e-12))
        score = float(rel.max())
        if best is None or score < best[1]:
            best = (vf, score, rel, [str(x.message)[:120] for x in ws])
        if score < tol:
            break
    vf, score, rel, warns = best
    model = model_response(vf, f, n)
    inb = (f >= lo) & (f <= hi) if band is None else (f >= band[0] * 1e9) & (f <= band[1] * 1e9)
    dev = np.abs(model - S).max(axis=(1, 2))
    out = dict(n_poles=int(len(vf.poles) + np.sum(vf.poles.imag != 0)), fit_range_ghz=[lo / 1e9, hi / 1e9],
               rel_rms_error=score, rel_rms_per_term=rel.round(4).tolist(), tol=tol, converged=bool(score < tol),
               max_abs_dev_band=float(dev[inb].max()) if inb.any() else None,
               unstable_poles=int(np.sum(vf.poles.real > 0)), warnings=warns)
    out["_model"] = model
    return vf, out


def _vf_term(s, H, n_poles=24, iters=20, causal=True):
    """Plain vector fitting of one response H(s) (s in j*GHz); causal=True flips unstable poles each iteration
    (the usual constraint), causal=False lets them stay. Returns the relative rms error."""
    w = np.linspace(abs(s).min(), abs(s).max(), n_poles // 2)
    p = np.r_[-w / 100 + 1j * w, -w / 100 - 1j * w]
    for _ in range(iters):
        C = 1 / (s[:, None] - p[None])
        A = np.c_[C, np.ones(len(s)), -H[:, None] * C]
        x = np.linalg.lstsq(A, H, rcond=None)[0]
        p = np.linalg.eigvals(np.diag(p) - np.outer(np.ones(len(p)), x[len(p) + 1:]))
        if causal:
            p = np.where(p.real > 0, -p.real + 1j * p.imag, p)
    C = np.c_[1 / (s[:, None] - p[None]), np.ones(len(s))]
    r = C @ np.linalg.lstsq(C, H, rcond=None)[0]
    return float(np.sqrt(np.mean(abs(r - H) ** 2)) / (np.sqrt(np.mean(abs(H) ** 2)) + 1e-30))


def causality_test(f, S, lo, hi, n_poles=24):
    """Per term: relative error of a causal (stable-pole) rational fit and of one allowed anti-causal poles over
    [lo, hi]. A term that only the unconstrained model follows is not causal: typically reflections of a part whose
    reference planes were de-embedded past its terminals. Terms near the noise floor (rms < 3e-3) are skipped."""
    m = (f >= lo) & (f <= hi) & (f > 0)
    s = 2j * np.pi * f[m] / 1e9
    n = S.shape[1]
    n_poles = int(min(n_poles, max(4, m.sum() // 8 * 2)))   # at least ~4 points per pole, or any data "fits"
    out = {}
    for i in range(n):
        for j in range(n):
            H = S[m, i, j]
            if np.sqrt(np.mean(abs(H) ** 2)) < 3e-3:
                continue
            out[f"S{i + 1}{j + 1}"] = dict(causal=round(_vf_term(s, H, n_poles, causal=True), 4),
                                         free=round(_vf_term(s, H, n_poles, causal=False), 4))
    return out


def model_response(vf, f, n):
    return np.stack([np.stack([vf.get_model_response(i, j, f) for j in range(n)], -1) for i in range(n)], 1)


def model_passivity(vf, n, lo, hi, npts=2000):
    """Bands (GHz) inside [lo, hi] where the fitted model's largest singular value exceeds 1 + PASSIVE_TOL, and that
    largest value."""
    fd = np.linspace(lo, hi, npts)
    sv = np.linalg.svd(model_response(vf, fd, n), compute_uv=False).max(axis=1)
    return [[a / 1e9, b / 1e9] for a, b in _bands(fd, sv > 1 + PASSIVE_TOL)], float(sv.max())


def check(path, band=None, active="auto", tol=0.01, max_poles=60, do_fit=True, enforce=False, spice=False,
          out=None, frange=None):
    """All checks of one Touchstone file; returns the summary dict (and writes enforced / SPICE models to out)."""
    path = Path(path)
    f, S, z0 = read(path)
    n = S.shape[1]
    r = dict(file=path.name, **data_checks(f, S, band, active))
    tau = group_delays(f, S, band)
    r["group_delay_ps"] = (tau * 1e12).round(1).tolist()
    r["negative_delay"] = bool((tau < -1e-12).any())
    r["_f"], r["_S"] = f, S
    if do_fit:
        try:
            vf, fr = fit(f, S, z0, tol, max_poles, band, frange)
        except ImportError:
            r["fit"] = dict(error="scikit-rf not installed (pip install scikit-rf)")
        else:
            r["fit"] = fr
            lo, hi = (x * 1e9 for x in fr["fit_range_ghz"])
            ct = causality_test(f, S, lo, hi)
            fr["causality"] = ct
            fr["non_causal_terms"] = [k for k, e in ct.items() if e["causal"] > max(tol, 0.03) and
                                      e["free"] < e["causal"] / 5]
            if not r["active"]:
                fr["model_passivity_violations_ghz"], fr["model_sv_max"] = model_passivity(vf, n, lo, hi)
                if enforce and fr["model_passivity_violations_ghz"]:
                    vf.passivity_enforce(f_max=hi)
                    fr["enforced"] = True
                    fr["model_passivity_after_ghz"], fr["model_sv_max_after"] = model_passivity(vf, n, lo, hi)
                    fr["_model"] = model_response(vf, f, n)
            if out is not None and (enforce or spice):
                o = Path(out)
                o.mkdir(parents=True, exist_ok=True)
                if enforce and n == 2:
                    from .rfpcb.circuit import write_s2p

                    mf = (f >= lo) & (f <= hi)
                    fn = o / f"{path.stem}_vf.s2p"
                    write_s2p(fn, f[mf], fr["_model"][mf], comment=f"vector-fitted model of {path.name}, "
                              f"{fr['n_poles']} poles, {lo / 1e9:g}-{hi / 1e9:g} GHz"
                              + (", passivity enforced" if fr.get("enforced") else ""))
                    fr["model_file"] = fn.name
                if spice:
                    fn = o / f"{path.stem}.sp"
                    vf.write_spice_subcircuit_s(str(fn))
                    fr["spice_file"] = fn.name
    r["verdict"] = verdict(r, band)
    return r


def verdict(r, band=None):
    """[(level, text)] with level 'error' / 'warning' / 'ok'."""
    v = []
    if r.get("band_covered") is False:
        v.append(("error", f"data {r['f_min_ghz']:g}-{r['f_max_ghz']:g} GHz does not cover the band"))
    if r["negative_delay"]:
        v.append(("error", f"negative group delay {r['group_delay_ps']} ps: not causal (reference planes moved past "
                           "the part?)"))
    if r["max_phase_step_deg"] > 90:
        v.append(("warning", f"phase steps up to {r['max_phase_step_deg']:.0f} deg between points: too coarse to "
                             "interpolate"))
    if not r["active"]:
        if r["sv_max_band"] is not None and r["sv_max_band"] > 1 + PASSIVE_TOL:
            v.append(("error", f"non-passive in the band (largest singular value {r['sv_max_band']:.3f})"))
        elif r["sv_max"] > 1 + PASSIVE_TOL:
            v.append(("warning", f"non-passive outside the band: singular value {r['sv_max']:.3f} at "
                                 f"{r['sv_max_f_ghz']:g} GHz"))
        if r["reciprocity_max"] > 0.02:
            v.append(("warning", f"non-reciprocal: max |Sij - Sji| = {r['reciprocity_max']:.3f}"))
    elif "mu_min" in r:
        if r["mu_min_band"] is not None and r["mu_min_band"] <= 1:
            v.append(("warning", f"potentially unstable in the band (mu min {r['mu_min_band']:.2f})"))
        elif r["mu_min"] <= 1:
            v.append(("warning", f"potentially unstable outside the band (mu min {r['mu_min']:.2f}): check with the "
                                 "real source / load impedances"))
    fr = r.get("fit")
    if fr and "error" not in fr:
        rng = f"{fr['fit_range_ghz'][0]:g}-{fr['fit_range_ghz'][1]:g} GHz"
        for k in fr.get("non_causal_terms", []):
            e = fr["causality"][k]
            v.append(("error", f"{k} is not causal over {rng}: a causal model misses it by {e['causal']:.1%}, one with "
                               f"anti-causal poles fits to {e['free']:.1%} (reference planes de-embedded past the part?)"))
        if not fr["converged"] and not fr.get("non_causal_terms"):
            v.append(("warning", f"no causal rational model within {fr['tol']:g} over {rng} (worst term "
                                 f"{fr['rel_rms_error']:.3f} with {fr['n_poles']} poles): noisy, under-sampled or "
                                 "non-causal data"))
        bad = fr.get("model_passivity_after_ghz" if fr.get("enforced") else "model_passivity_violations_ghz")
        if bad:
            v.append(("warning", f"fitted model non-passive in " + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in bad[:4])
                      + f" GHz (sv max {fr.get('model_sv_max_after', fr.get('model_sv_max')):.3f})"
                      + ("" if fr.get("enforced") else "; --enforce writes a passive model")))
    if not v:
        v.append(("ok", "passes all checks"))
    return v


# --------------------------------------------------------------------------- outputs

def _clean(r):
    def c(x):
        if isinstance(x, dict):
            return {k: c(v) for k, v in x.items() if not k.startswith("_")}
        if isinstance(x, (list, tuple)):
            return [c(v) for v in x]
        return x
    return c(r)


def write_markdown(results, path, band=None):
    L = ["# S-parameter data checks\n", "Band: " + (f"{band[0]:g}-{band[1]:g} GHz" if band else "whole file") + "\n",
         "| file | ports | f [GHz] | kind | passivity / stability | reciprocity | causal fit | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["active"]:
            ps = f"mu min {r.get('mu_min', float('nan')):.2f} (band {r.get('mu_min_band') or float('nan'):.2f})"
            rc = "-"
        else:
            ps = f"sv max {r['sv_max']:.3f} @ {r['sv_max_f_ghz']:g} GHz (band {r['sv_max_band'] or float('nan'):.3f})"
            rc = f"{r['reciprocity_max']:.3f}"
        fr = r.get("fit") or {}
        fit_s = fr.get("error") or (f"{fr['n_poles']} poles over {fr['fit_range_ghz'][0]:g}-{fr['fit_range_ghz'][1]:g} GHz, "
                                    f"worst term rel rms {fr['rel_rms_error']:.4f}" if fr else "-")
        vd = "; ".join(f"{lv}: {t}" for lv, t in r["verdict"])
        L.append(f"| {r['file']} | {r['n_ports']} | {r['f_min_ghz']:g}-{r['f_max_ghz']:g} | "
                 f"{'active' if r['active'] else 'passive'} | {ps} | {rc} | {fit_s} | {vd} |")
    Path(path).write_text("\n".join(L) + "\n")
    return path


def write_html(results, path, band=None):
    try:
        from plotly.offline import get_plotlyjs
        js = "<script>" + get_plotlyjs() + "</script>"
    except ImportError:  # pragma: no cover
        js = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
    data = []
    for r in results:
        f = (r["_f"] / 1e9).round(5).tolist()
        S = r["_S"]
        n = S.shape[1]
        d = dict(file=r["file"], f=f, active=r["active"], verdict=r["verdict"],
                 s={f"S{i + 1}{j + 1}": (20 * np.log10(abs(S[:, i, j]) + 1e-15)).round(3).tolist()
                    for i in range(n) for j in range(n)})
        fr = r.get("fit") or {}
        if "_model" in fr:
            M = fr["_model"].copy()
            lo, hi = fr["fit_range_ghz"]
            M[(r["_f"] < lo * 1e9 - 1) | (r["_f"] > hi * 1e9 + 1)] = np.nan       # model shown over its fit range only
            d["m"] = {f"S{i + 1}{j + 1}": (20 * np.log10(abs(M[:, i, j]) + 1e-15)).round(3).tolist()
                      for i in range(n) for j in range(n)}
            d["fit"] = f"{fr['n_poles']} poles, {fr['fit_range_ghz'][0]:g}-{fr['fit_range_ghz'][1]:g} GHz, worst term rel rms {fr['rel_rms_error']:.4f}"
        if r["active"] and n == 2:
            K, D, mu = stability(S)
            d["mu"] = np.minimum(mu, 5).round(4).tolist()
            d["K"] = np.minimum(K, 5).round(4).tolist()
        else:
            d["sv"] = r["_sv"].round(5).tolist()
        data.append(d)
    html = _TPL.replace("__PLOTLY__", js).replace("__DATA__", json.dumps(dict(items=data, band=band)))
    Path(path).write_text(html)
    return path


_TPL = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>S-parameter data checks</title>__PLOTLY__
<style>body{margin:0;background:#111418;color:#d8dde3;font:13px/1.4 system-ui,sans-serif}h1{font-size:17px;margin:10px 16px}
h2{font-size:14px;margin:6px 4px}.card{background:#181c22;border:1px solid #3a4048;border-radius:6px;margin:8px 12px;padding:6px}
.row{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:8px}.p{height:340px}
.v{padding:2px 6px}.error{color:#ff7b72}.warning{color:#ffb454}.ok{color:#91cf60}</style></head><body>
<h1>S-parameter data checks</h1><div id="items"></div><script>
const D=__DATA__;const C=['#5ad1ff','#ffb454','#91cf60','#ff7b72','#c792ea','#ffd75a','#7fdbca','#f78c6c'];
const L=(yt)=>({paper_bgcolor:'#181c22',plot_bgcolor:'#111418',font:{color:'#d8dde3'},margin:{l:55,r:10,t:10,b:40},
 xaxis:{title:{text:'f [GHz]'},gridcolor:'#3a4048',zerolinecolor:'#3a4048'},yaxis:{title:{text:yt},gridcolor:'#3a4048',zerolinecolor:'#3a4048'},
 legend:{orientation:'h',y:1.12},shapes:D.band?[{type:'rect',xref:'x',yref:'paper',x0:D.band[0],x1:D.band[1],y0:0,y1:1,fillcolor:'#ffffff',opacity:0.06,line:{width:0}}]:[]});
D.items.forEach((it,k)=>{const c=document.createElement('div');c.className='card';
 c.innerHTML=`<h2>${it.file} <span style="color:#8a939e">(${it.active?'active':'passive'}${it.fit?', fit: '+it.fit:''})</span></h2>`+
 it.verdict.map(v=>`<div class="v ${v[0]}">${v[0]}: ${v[1]}</div>`).join('')+`<div class="row"><div id="s${k}" class="p"></div><div id="q${k}" class="p"></div></div>`;
 document.getElementById('items').appendChild(c);
 const tr=[];Object.keys(it.s).forEach((n,i)=>{tr.push({x:it.f,y:it.s[n],name:n+' data',line:{color:C[i%8],width:1.5}});
  if(it.m)tr.push({x:it.f,y:it.m[n],name:n+' fit',line:{color:C[i%8],width:1,dash:'dot'}});});
 Plotly.newPlot('s'+k,tr,L('|S| [dB]'),{displaylogo:false,responsive:true});
 const q=it.sv?[{x:it.f,y:it.sv,name:'largest singular value',line:{color:'#5ad1ff'}},{x:[it.f[0],it.f[it.f.length-1]],y:[1,1],name:'passive limit',line:{color:'#ff7b72',dash:'dash'}}]
  :[{x:it.f,y:it.mu,name:'mu (capped at 5)',line:{color:'#91cf60'}},{x:it.f,y:it.K,name:'K (capped at 5)',line:{color:'#ffb454'}},{x:[it.f[0],it.f[it.f.length-1]],y:[1,1],name:'1',line:{color:'#ff7b72',dash:'dash'}}];
 Plotly.newPlot('q'+k,q,L(it.sv?'singular value':'stability factor'),{displaylogo:false,responsive:true});});
</script></body></html>"""


def run(paths, band=None, out=".", **kw):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for p in paths:
        r = check(p, band=band, out=out, **kw)
        results.append(r)
        (out / f"{Path(p).stem}_check.json").write_text(json.dumps(_clean(r), indent=1))
        print(f"{r['file']}: " + "; ".join(f"{lv}: {t}" for lv, t in r["verdict"]), flush=True)
    write_markdown(results, out / "sparcheck.md", band)
    write_html(results, out / "sparcheck.html", band)
    print("wrote", out / "sparcheck.md", "and", out / "sparcheck.html")
    return results


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m mom25d.sparcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--band", nargs=2, type=float, metavar=("F0", "F1"), help="GHz")
    ap.add_argument("--out", default=".")
    ap.add_argument("--active", default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--tol", type=float, default=0.01, help="relative rms error of the fit")
    ap.add_argument("--max-poles", type=int, default=60)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--fit-range", nargs=2, type=float, metavar=("F0", "F1"),
                    help="GHz (default: the band widened by 30 %%, else the whole file without DC)")
    ap.add_argument("--enforce", action="store_true", help="enforce passivity of the fitted model, write <name>_vf.s2p")
    ap.add_argument("--spice", action="store_true", help="write the fitted model as a SPICE subcircuit")
    a = ap.parse_args(argv)
    run(a.files, band=a.band, out=a.out, active=a.active, tol=a.tol, max_poles=a.max_poles, do_fit=not a.no_fit,
        enforce=a.enforce, spice=a.spice, frange=None if a.fit_range is None else [x * 1e9 for x in a.fit_range])


if __name__ == "__main__":
    main()
