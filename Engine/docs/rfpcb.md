# mom25d.rfpcb — RF PCB cascades from a DXF (board workflow on top of mom25d)

A config-driven package that turns a de-featured layout DXF plus a board stack-up into a circuit model, de-embedded
MoM2.5 line-section models, merged groups and a complete one-solve model of an RF cascade, with self-contained dark
viewers. It is a subpackage of mom25d (`mom25d.rfpcb`); put `Engine/` on `PYTHONPATH` (or set `MOM25D_PATH` to the folder
holding `Engine/` for scripts that add it). Simulation-manager submission: `RFPCB_SIMMGR`, optional `RFPCB_PREFLIGHT`.

## Workflow

1. **ODB++ → DXF.** Extract the cascade region (whole pads and package bodies, so connectors are not clipped):
   `L<n>_METAL` / `L<n>_NOT_METAL` per copper layer, drill circles per via span (`VIA_DRILL_L1-L3_D152`, …),
   `COMP_OUTLINE`, and a `stackup.json` (layers top → bottom with `kind` mask/cu/diel, `t_um`, `eps`, `tand`; vias
   with `from`, `to`, `pad_mm`, `dxf_layers`).
2. **The designer de-features the DXF and draws the ports** (conventions below). The edited DXF is the master: back
   it up before any tool edit (`dxfio.clean` does, and refuses while the editor's `~file` shows unsaved edits).
3. **board.json** (see `rfpcb/config.py`): DXF, stack-up, the chain (line, part, line, …) with each part's S2P and
   the port labels at each line end, port kinds, model settings.
4. `python -m mom25d.rfpcb board.json check` — sections, ground, vias, unknown counts, 3-D model pages (no solve).
5. `python -m mom25d.rfpcb board.json circuit` — quasi-static line table (2-D finite-volume cross-section) + parts:
   the reference for every EM step.
6. `python -m mom25d.rfpcb board.json submit sections` then `sections --post` — straight reference lines per port type, the
   line sections (dense solves), de-embedding, cascade with the MoM sections.
7. Merge the lines on both sides of any part with a grounded paddle (`model.merge`), then
   `submit complete --freqs 27,29,31` and `complete --post` — the complete model (H-matrix solver; currents and port
   waves kept for current / field pages).

Every EM run goes through the simulation manager (`rfpcb.jobs.submit`, pre-flight memory check, 3-D model link).

## DXF conventions

- Copper by the painting rule: `METAL` rings combine even-odd, `NOT_METAL` areas are removed, `METAL` islands inside a
  `NOT_METAL` area are added back. Close every polyline; no zero-length lines.
- `PORTS`: a LINE across the trace = a reference plane (a gap port to a separate stitched pad; `"kind": "feed"` in
  board.json makes it a GCPW feed). Label = TEXT on the same layer next to the line.
- `GAP_PORTS`: a LINE from the RF line end to the ground copper it lands on (either drawing direction) = a drawn gap
  port. Label as above.
- Inner planes are implied everywhere; their `NOT_METAL` rings are voids (openings when `model.plane` is set).

## What is validated

- Gap ports to a separate stitched pad or to the coplanar pour across a slot: passive and symmetric on a uniform line
  with a ground band ≥ 2–3 mm; their bridge / slot parasitics are removed with reciprocal error boxes fitted on two
  straight reference lines (check with a third length: `deembed.predict_check`).
- GCPW feeds (engine de-embedded) at a model's outer end.
- Parity with the board scripts this package was generalised from: identical section ground / vias / ports, unknown
  counts of a merged group and a complete model, circuit cascade to 0.01 dB, re-de-embedded sections to 1e-8.

## Known limits

- An L1 ground plane with the line inside its opening does not work in mom25d (even feeds fail with the sheet 2 µm
  above the plane level; at 10 µm eps_eff is +16 %). Inner planes on their own level (an L2 under the pour) are fine
  in principle — validate on a straight line first.
- Never use tan δ exactly 0 with internal ports ("lossless" here means tan δ = `stack.tand_min`).
- Ground band ≥ 3 mm; a truncated band leaves resonant fragments and fake coupling. Near a high-gain part, check that
  coupling converges with more ground (no ownership, wider band).
- A part with a grounded paddle split between two section models makes each half resonate: merge those lines.
- Without a plane, vias that end on an inner foil are taken to the ground plane.
- Thick (two-sheet) copper is not generalised (use the thin sheet + `stack.edge_delta`, fitted with a quasi-static
  planar model so the sheet line has the thick line's Z0).
- Circuit centreline: the medial axis of the RF copper (bends follow the drawn copper); `circuit.path` overrides it.

## Layout

`config` (board.json) · `dxfio` (DXF) · `stackup` · `geometry` (sections, ownership, lands, merge, centreline) ·
`place` (mom25d layouts: gap / pad / shared-pad / feed ports, optional plane) · `circuit` (Touchstone, ABCD, qs2d line
table) · `deembed` (boxes, N-port de-embedding, part insertion) · `models` (runs and post-processing) · `viewers`
(3-D model, currents, response) · `jobs` (simulation manager) · `cli`. Tests: `pytest -q Engine/tests/test_rfpcb.py` (synthetic board).
