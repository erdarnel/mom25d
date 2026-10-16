# S-parameter data checks (`mom25d.sparcheck`)

Checks Touchstone data before it goes into a circuit cascade, an EM model (`Layout.add_sparams`) or a
post-solve connection of parts to a layout. Vendor and measured files are often slightly non-passive, non-reciprocal,
under-sampled, or not consistent with any causal system (reference planes de-embedded past the part); connected
into a cascade they then produce gain, ripple or resonances that are not real.

```
python -m mom25d.sparcheck parts/*.s2p --band 27.5 31 --out checks [--enforce] [--spice]
```

writes `checks/<name>_check.json`, a summary `checks/sparcheck.md` and a dark-mode page `checks/sparcheck.html`
(|S| data with the fitted model dotted, the largest singular value or the stability factors against frequency, the
band shaded, the verdicts). Needs `scikit-rf` for the model checks (`pip install -e Engine[vf]`); `--no-fit` runs
the data checks alone.

## Checks

| Check | How | Flags |
|---|---|---|
| Band coverage | data range vs `--band` | error if the band is not covered |
| Passivity (passive parts) | largest singular value of S per frequency | error > 1.001 in the band, warning outside |
| Reciprocity (passive parts) | max \|S_ij - S_ji\| | warning > 0.02 |
| Stability (active 2-ports) | Rollett K, \|Delta\|, mu (Edwards-Sinsky) | warning where mu <= 1 |
| Sampling | largest phase step of each transmission term between points (terms above -40 dB, within 30 dB of their peak) | warning > 90 deg: too coarse to interpolate |
| Bulk delay | median group delay of each transmission term in the band | error if negative |
| Causality | per term: vector fit with stable poles only vs one allowed anti-causal poles | error if only the latter follows the data |
| Rational model | scikit-rf vector fitting, common poles, order raised until every term's rms error < `--tol` x its rms | warning if not reached |
| Model passivity | largest singular value of the fitted model on a dense grid over the fit range | warning; `--enforce` makes it passive and writes `<name>_vf.s2p` |

An active part is detected when an off-diagonal term exceeds 0 dB (`--active yes|no` overrides).

## Notes

- **Fit range**: the band widened by 30 % at each edge (clipped to the data), else the whole file without the DC
  point; `--fit-range` overrides. Wide fits of amplifier data are dominated by stopband noise.
- **Term weights**: every term is normalised by its rms (at least 0.03) before the common-pole fit, so a 25 dB
  S21 does not set the poles alone and noise-floor terms (an amplifier's S12) do not steer them; the residues are
  scaled back, so the fitted object describes the real S.
- **Causality test**: a band-limited fit with anti-causal poles allowed follows any smooth data; a causal fit that
  misses a term by more than 5x as much marks it as not causal. Seen on a vendor amplifier file de-embedded to the
  die (S11 / S22: causal fit 29 % / 64 % error, unconstrained 1 %); its S21 was causal. The pole count is limited to
  ~1 per 4 frequency points so short files cannot be "fitted" trivially.
- **SPICE**: `--spice` writes `VectorFitting.write_spice_subcircuit_s` for time-domain circuit simulation.
- Not a substitute for checking that the data describes the right part, bias and reference planes.
