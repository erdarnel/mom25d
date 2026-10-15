# Changelog

## Unreleased

- New: `add_aperture(..., size=)` graded opening meshes; graded meshes checked: max |dS| 0.003 vs uniform on a line section.

- **New: `mom25d.oscillation`** — oscillation risk screen for PCBs with RF gain stages: chains traced along the
  copper centre lines (ODB++) or given as points, signal level from line loss and part gains (tables, S2P,
  states, temperature coefficients), loop gain for line-line, via-radiation and plane-cavity couplings at every
  frequency / temperature corner; JSON, Markdown and a self-contained viewer (`templates/oscillation.html`).
  Docs `Engine/docs/oscillation.md`; tests `test_oscillation.py`.

- Fix: coincident vias in a closed plane-to-plane section (e.g. a DXF via on a GCPW feed's fence via) divided by
  zero in `via_impedance`'s TM_n sum; such pairs now count as one barrel (distance = drill radius).

## 0.2.0 — 2026-10-15

**Results change for existing `kind="tri"` models** (see the first item): re-mesh and expect more unknowns at the
same `max_cell` and slightly different S. Pin 0.1.0 to reproduce older numbers.

- **Fix: triangle meshes now honour `max_cell` in their interiors** (decision 37). The interior area bound was passed
  to Triangle in exponent notation, which it does not parse, so interiors were bounded only by the angle limit
  (pours and plane openings had 1-3 mm² triangles at max_cell 0.15-0.2 mm). Plane-opening fields under narrow lines
  were not resolved at all.
- **Fix: vias through a plane opening** (decision 36): a via that passes a plane inside one of its apertures is no
  longer treated as a via between solid plates (that added a fake radial-wave loss); one quasi-static piece over its
  span. `Mesh.via_open_crossings()`.
- **New: graded interior meshes** (decision 38): `layout.mesh(..., kind="tri", size=f)` with a cell-size field, e.g.
  `trimesh.graded_size(signal_copper, fine_cell, max_cell, slope)` — fine near the lines, coarse in the far pour.
  A 16-port board model: 59 836 -> 33 261 unknowns. Opt-in.
- **New: `leakage.py`** — power each via launches into closed plane pairs (per via, total, pattern, voltage map) and a
  self-contained map page; vias through openings are left out.
- **New: `mom25d.rfpcb`** — config-driven RF PCB cascade workflow (DXF -> circuit, de-embedded line sections, merged
  groups, complete H-matrix model, viewers, simulation-manager submission); docs `Engine/docs/rfpcb.md`.
- **New: `Solver(tand_floor=)`** opt-in floor for exactly lossless dielectrics with internal ports (decision 35);
  plane apertures drawn in `view3d`.
- **Docs: `LIMITATIONS.md` and `ASSUMPTIONS.md`** — what fails / what every result assumes; read before modelling.
- Tests: `test_regressions_mesh_planes.py` (area bound, graded mesh, gap ports over a plane opening; full-opening
  equivalence as an expected failure until fixed), `test_rfpcb.py`, cavity-leakage test in `test_planes.py`.

## 0.1.0 — 2026-10-13

First public version.
