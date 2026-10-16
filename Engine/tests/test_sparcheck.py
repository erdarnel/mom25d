"""S-parameter data checks (mom25d.sparcheck) on synthetic Touchstone files."""
import numpy as np
import pytest

from mom25d import sparcheck as sc
from mom25d.rfpcb.circuit import write_s2p

F = np.linspace(1e9, 40e9, 391)


def _line(f, z=60.0, length=0.02, eeff=3.0, loss_db_m=5.0):
    """S of a lossy transmission line of impedance z in a 50 ohm system (passive, reciprocal, causal)."""
    g = 1j * 2 * np.pi * f * np.sqrt(eeff) / 3e8 + loss_db_m / 8.686
    t = np.exp(-g * length)
    r = (z - 50) / (z + 50)
    d = 1 - r ** 2 * t ** 2
    S = np.zeros((len(f), 2, 2), complex)
    S[:, 0, 0] = S[:, 1, 1] = r * (1 - t ** 2) / d
    S[:, 0, 1] = S[:, 1, 0] = t * (1 - r ** 2) / d
    return S


def _write(tmp_path, name, S, f=F):
    p = tmp_path / name
    write_s2p(p, f, S)
    return p


def test_passive_line_passes(tmp_path):
    r = sc.check(_write(tmp_path, "line.s2p", _line(F)), band=(27.5, 31), do_fit=False)
    assert not r["active"] and r["sv_max"] <= 1 and r["reciprocity_max"] < 1e-9
    assert r["verdict"] == [("ok", "passes all checks")]
    assert r["group_delay_ps"][1][0] > 0


def test_non_passive_and_non_reciprocal(tmp_path):
    S = _line(F) * 1.05
    S[:, 0, 1] *= 0.9
    v = dict(sc.verdict(sc.check(_write(tmp_path, "gain.s2p", S), band=(27.5, 31), do_fit=False)))
    assert "non-passive in the band" in v["error"]


def test_band_not_covered(tmp_path):
    r = sc.check(_write(tmp_path, "short.s2p", _line(F)[:100], F[:100]), band=(27.5, 31), do_fit=False)
    assert r["band_covered"] is False and r["verdict"][0][0] == "error"


def test_amplifier_stability(tmp_path):
    S = np.zeros((len(F), 2, 2), complex)
    S[:, 0, 0], S[:, 1, 1] = 0.2, 0.3
    S[:, 1, 0] = 10 * np.exp(-2j * np.pi * F * 100e-12)
    S[:, 0, 1] = 0.001
    r = sc.check(_write(tmp_path, "amp.s2p", S), band=(27.5, 31), do_fit=False)
    assert r["active"] and r["mu_min"] > 1 and r["gain_max_db"] == pytest.approx(20, abs=0.01)
    S[:, 0, 1] = 0.2                                     # strong feedback: potentially unstable
    r = sc.check(_write(tmp_path, "amp2.s2p", S), band=(27.5, 31), do_fit=False)
    assert r["mu_min_band"] < 1


def test_causality_test_flags_advanced_reflection():
    S = _line(F)
    lo, hi = 19e9, 40e9
    ok = sc.causality_test(F, S, lo, hi)
    assert all(e["causal"] < 0.03 for e in ok.values())
    Sb = S.copy()
    Sb[:, 0, 0] *= np.exp(+2j * np.pi * F * 150e-12)     # reflection advanced by 150 ps: anti-causal
    bad = sc.causality_test(F, Sb, lo, hi)
    assert bad["S11"]["causal"] > 5 * bad["S11"]["free"]


def test_vector_fit_and_outputs(tmp_path):
    pytest.importorskip("skrf")
    p = _write(tmp_path, "line.s2p", _line(F))
    res = sc.run([p], band=(27.5, 31), out=tmp_path / "o", enforce=True)
    fr = res[0]["fit"]
    assert fr["converged"] and fr["unstable_poles"] == 0 and not fr["non_causal_terms"]
    assert (tmp_path / "o" / "sparcheck.html").exists() and (tmp_path / "o" / "line_check.json").exists()
