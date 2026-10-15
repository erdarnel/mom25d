# mom25d — limitations and what to avoid

Read before setting up a model. Each item: what does not work (or is not yet trustworthy), the symptom you would
see, what to do instead, and how to check. Evidence: the numbered decisions (Dn) in [CLAUDE.md](CLAUDE.md). What every
result assumes anyway (infinite layers, PEC ground, sheet copper, ...) is in [ASSUMPTIONS.md](ASSUMPTIONS.md).
**Add new failures here when you find them.**

## Before trusting any new kind of model

- Validate a new port type, ground arrangement or plane configuration on a **uniform straight line** first:
  |S11|² + |S21|² <= 1 at every frequency, S11 = S22 for a symmetric line, |S12 - S21| ~ 1e-6, and the
  **third-length test** (error box fitted on two lengths predicts a third; |dS| <~ 0.05 is good).
- Check mesh convergence once per structure type (halve `edge_cell`, change `max_cell`).
- Cross-check one representative window against openEMS / FEM when the result matters.

## Ports

| Limitation | Symptom | Do instead / check |
|---|---|---|
| Gap and via ports with **tan d exactly 0** in every dielectric (D35) | S11 several dB off, non-passive at some f | Use tan d >= 1e-5 (`Solver(tand_floor=1e-5)` opt-in, or set it in the stack); never "lossless" = 0 with internal ports |
| Gap-port **parasitics** (bridge + slot) are in S (D35) | S11 -10..-18 dB on a matched line | De-embed with straight reference lines of the same port type (two lengths + a third to check) |
| **Truncated ground** at a gap port's ground terminal (narrow band, one-via fragments) (D35) | Narrow resonances in missing power, "port failures" | Ground band >= 2-3 mm, drop via-less islands; stitch the terminal |
| Feed **fit window in a near field** (stub, resistor, bend within ~1 mm of the reference plane) (D28) | Non-reciprocal raw S (|S_ij - S_ji| large), odd Z0 | Longer feeds / larger `margin`; `Solver(z0_recip_min=)`; check raw reciprocity first |
| Line voltage invalid where the strip's return is the plane **above** (flipped boards, dummy grounds) (D28) | Z0 drifting with f, eps_eff biased 3-4 % | `Solver(v_weight=0)` (window >~ lambda_g/2), `Port(z0_from="reciprocity")` |
| **Feed-only** models with plane apertures: Z0 from the line voltage is wrong (D33) | Z0 ~ half the expected value | Z0 from an internal port (reciprocity) or `Solver(z0=)`; gamma is fine |
| Two pad ports closer than two pad depths | Overlapping pads (shorted ports) | One shared stitched pad for both (mom25d.rfpcb does it) |
| No **differential / multimode** ports | — | Single-ended ports + post-processing |
| Delta-gap feeds make a closed resonator; raw currents on unexcited feeds ring (D8) | Unphysical currents on feeds | Plot `matched_currents` / `wave_currents`; ignore feed sections beyond the gaps |

## Ground, truncation and section models

| Limitation | Symptom | Do instead / check |
|---|---|---|
| The modelled ground **ends** (band model): energy reaching its edge is lost / reflected | Missing power rising with f; coupling between ports that changes with band width | Band >= 3 mm; near high-gain parts check coupling **converges** as the ground grows (e.g. -26 dB at 3 mm became -40 dB at 5-8 mm) |
| A part's **grounded paddle split** between two section models | Each half resonates (lambda_g/2 patch), 10-25 % missing in narrow peaks | Model the lines on both sides of the part as one model; insert the part's S2P after de-embedding |
| Cutting ground by **boxes** at port planes | Unstitched fingers or a section's own ground removed at bends | Nearest-RF ownership (Voronoi) of the ground |
| Part bodies (packages, lids, resistor bodies) are **not modelled** | Coupling across a part (e.g. amplifier output -> input) without the package's shielding | Treat such coupling as indicative; cross-check with a 3-D solver |

## Planes and openings

| Limitation | Symptom | Do instead / check |
|---|---|---|
| A **meshed sheet on the same level as a plane** (e.g. L1 pour as a plane with the line inside its opening) | Even feed ports fail with the sheet 2 um above the plane level (eps_eff 0.7-17); at 10 um eps_eff +16 % | Keep the pour as meshed copper; put planes only on their own levels |
| **Large plane openings** (an opening over a whole model) (D37 open point) | Does not reproduce the same model without the plane (|dS| ~1, S21 phase far off) | Openings along a line validated up to 4.5 mm wide (passive, S11 = S22, third-length \|dS\| 0.04-0.07; 1.3 mm: 0.007-0.06); an opening over a whole model fails. Test the size you need on a straight line first |
| Openings must have their **own fine mesh** (D33, D37) | eps_eff / Z0 off by 0.1-3 ohm with coarse cells | `add_aperture(max_cell=, edge_cell=)` with several cells across the slot (meshes before 2026-10-15 ignored max_cell inside openings) |
| **Thick planes** (two faces sharing one M) disabled (D33) | Z0 4 ohm low for an 18 um foil | Zero-thickness planes |
| Planes are infinite: **no cavity / board-edge resonances** (D18) | Agrees with an absorbing-boundary reference, not a real board | Cavity model or openEMS for board-level resonances |

## Vias

| Limitation | Symptom | Do instead / check |
|---|---|---|
| Quasi-static barrel (D15, scaling notes): OK while k1 h is small (~0.3 per 0.254 mm layer at 31 GHz); long through-vias are not full-wave | Via inductance / resonance error at mm-wave on tall vias | Keep via spans short or cross-check in openEMS |
| A via must attach to the cells **on both sides** if centred on a grid line (D15) | S11 != S22 by ~10 deg on a symmetric layout | `kind="tri"` or off-grid via centres |
| **Antipads are experimental** (D22) | Plate coupling errors with one frill ring | Use >= the default rings; validate |
| Without an inner plane in the stack, vias that end on an inner foil go to the ground plane | Wrong return path | Add the plane |

## Meshing

| Limitation | Symptom | Do instead / check |
|---|---|---|
| Before 2026-10-15 `kind="tri"` meshes ignored `max_cell` in interiors (D37) | Old models: coarse pours; plane openings unresolved | Re-mesh old models; at the same `max_cell` N now grows (~45 % on a line section): choose `max_cell` for pours on purpose |
| Grid mesh staircases oblique copper | +0.9 mm electrical length on an 8.5 mm 45 deg line | `kind="tri"` |
| CAD slivers / near-duplicate points | Tiny angles, hanging nodes | Clean the DXF (close polylines, delete zero-length lines); copper snaps to 0.05 edge_cell |
| Wide pours cost many unknowns since the mesh fix (D37) | Large N from smooth pour interiors | `mesh(size=trimesh.graded_size(...))` (D38): fine near the lines, coarse far away; check S against a uniform mesh once |

## Size, memory and run time

- Dense solve memory ~48 N² bytes (three complex N x N matrices): ~22 k unknowns in 30 GB.
- H-matrix: memory, not speed, is the gain near N ~ 1e4; measured N = 30 554 (16 ports): 19 GB peak, ~49 min per
  frequency, the per-port GMRES solves dominate. GMRES needs more iterations at low frequency.
- `Solver(method="auto")` may pick the H-matrix; `memory_estimate()` then returns None (use `method="dense"` to get it).
- `FreqInterp` stores dense matrices and is off for H-matrix and aperture models.
- Whole boards (1e7-1e8 unknowns): do not solve whole — cut into windows, cascade (scaling notes in CLAUDE.md).

## Losses and materials

- No conductor roughness model; no frequency-dispersive eps / tan d.
- With losses, alpha is fitted freely: de-embedding noise can bias |S21| by 0.2-0.5 % (longer feeds help).
- Ground-plane loss model valid for k1 h << 1 (D13).

## Not modelled at all

Board edges and enclosures; part bodies; soldermask conformity on thick copper (a cover layer only); copper
roughness; differential / multimode ports; magnetic materials; nonlinear or active devices (only as S-parameter
blocks); thermal effects.
