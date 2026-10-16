# Oscillation risk screen (`mom25d.oscillation`)

A closed-form screen of a PCB with RF gain stages for feedback loops through layout coupling. It runs in seconds
to minutes from the layout data, ranks every loop, and points at the few couplings that need an EM model. It does
not solve any field problem itself.

```
python -m mom25d.oscillation board_osc.json      # -> <out>/oscillation.json, .md, .html
```

## What it computes

1. Each chain (one per channel) is traced along the copper centre line (medial axis of the net's copper, from the
   ODB++ export) or taken from explicit points, sampled every `step` mm.
2. Signal level along the chain: line loss per mm (scaled with sqrt(f)), each part's gain (number, `gain_vs_f`
   table, S2P |S21|, or named `states`), a temperature coefficient for active parts, via loss. Every active part
   starts a new stage.
3. Loops: a downstream sample (aggressor, later stage, higher level) coupled to an upstream one (victim).
   Loop gain = level difference + coupling, at its worst phase.
4. Couplings (formulas in the module docstring, all upper bounds):
   - **line-line** on the same layer (bare microstrip, quasi-static; coherent sum along the run; layers assumed
     shielded from each other);
   - **via radiation**: a signal via's vertical current into upstream lines on each layer it passes, times its
     ground-via fence transmission;
   - **via cavity**: a signal via's radial parallel-plate wave in each plane-pair cavity it crosses, to upstream
     signal vias, times both fences.
5. Corners: every frequency x temperature (and part states) given.

| Worst loop gain | Meaning |
|---|---|
| >= 0 dB | oscillation possible: change the layout |
| -10 .. 0 dB | small margin; the formulas cannot decide: EM model of that coupling |
| -20 .. -10 dB | no oscillation; ripple (about +-0.9 dB at -20 dB) |
| < -20 dB | low risk |

## Config

See the docstring of `oscillation.load` for the full schema. Key points:

- `layers`: per routing layer `h` (to its reference plane), `eeff`, optional `t`, `eps` (substrate, for fence
  wavelengths), `z0`, `loss_db_per_mm` at `loss_f_ghz`, `kc_ratio` (kC / kL, 0.7 by default),
  `coupling_offset_db` (a correction fitted to a 2-D coupled-line solve or an EM window, e.g. for GCPW whose side
  grounds the formula ignores), `fence_drill_layers` (ground vias that fence a signal via on this layer).
- `cavities`: plane pairs (`eps`, `d`, `fence_drill_layers`) that signal vias cross (per via: `"cavities": [...]`,
  default all).
- `parts`: gains; `"active": true` for gain stages; `tc_db_per_c` (negative when gain rises in the cold).
- `channels`: ordered paths of `part`, `line` and `via` items; `repeat` with `{n}` placeholders for identical
  channels. Line endpoints by refdes, `REF.PIN`, part-number prefix (`"ADL8141*"`), `via` / `via:NET` or
  coordinates; `split_at` inserts a part (e.g. a Wilkinson) partway along a traced line.
- `parallel`: a section of two or more branches between a splitter and a combiner (balanced amplifier pairs,
  Wilkinson-split stages): `{"parallel": [[branch items], [branch items]], "label": ..., "split_db": ...,
  "combine_db": ...}`. Each branch is an ordinary path (lines, parts, vias) that starts where the previous line
  ended, e.g. a line to `U2_1`, the amplifier, a line from `U2_1` on; the other branch the same through `U20_1`.
  The centre line all branches share at the start (up to the split junction) and at the end (from the combine
  junction) is taken out as one line. Level: split -10 log10(n) dB by default; the branch end levels combine as an
  ideal in-phase combiner (sum of amplitudes / sqrt(n)). A signal coupled into one branch reaches the combined
  output 20 log10(n) dB below the plain level difference (the branch level already carries the -10 log10(n)
  split, and only 1 / sqrt(n) of one branch's amplitude is combined), so loops whose victim is in a branch and
  whose aggressor is not in that same branch are reduced by 20 log10(n). Every branch is sampled and searched for
  loops, also across branches of the same section; the combiner's isolation is not counted there, so those loops
  are upper bounds (marked in the note). Nested parallel sections are not supported.
- `corners`: `{"f": [...], "t": [...]}` or a list of `{"f", "t", "states", "label"}`.
- `links`: `[{"label", "href"}]` shown in the viewer's toolbar (e.g. the layout viewer, the DXF).

## Viewer

`oscillation.html` (self-contained, dark): one map per layer with the copper near the paths, path samples
coloured by their margin to oscillation (worst loop through the point), signal vias coloured by their worst loop
as a source, aggressor -> victim arrows of the worst loops; filters by corner, chain and mechanism; a sortable loop
table (click a row to highlight and zoom to it); signal level along each path with the parts marked; thresholds,
assumptions and the layer parameters.

## Limits

- Coupling between different chains (e.g. channels that meet in a combiner) is not counted.
- Loops across branches of a parallel section ignore the isolation of the splitter / combiner.
- Bias, control and supply lines, part-internal feedback and package coupling are not modelled.
- Couplings are worst-phase magnitudes; formulas for bare microstrip overestimate fenced / GCPW lines unless
  `coupling_offset_db` is fitted.
- The centre-line tracer needs the RF nets as named copper in the ODB++ export; for other sources give the
  centre lines as `points`.
