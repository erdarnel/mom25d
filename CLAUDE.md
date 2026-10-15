# mom25d — notes for continuing work

2.5D method of moments (MPIE + RWG) for planar circuits: conductors on any
interface of a multilayer stack over a ground plane (air or PEC top, solid
inner PEC planes), with vias between layers and planes. Pure Python/numpy/scipy. See README.md
for the formulation, API and validation table; this file records *why* things
are the way they are and the traps already hit.

## Folder layout (since 2026-10-04)

```
Engine/                  the solver: everything not specific to one model
  mom25d/                Python package
  tests/ (+ data/)       validation suite, openEMS reference scripts + data
  tools/                 generic CLIs: odb_import.py, odb_3d_view.py (+ template)
  docs/                  ODB++ spec (odb_spec_user.pdf)
  pyproject.toml
Models/<model>/
  geometry/              <model>.py: layout(), substrate() / stackup(), constants
                         (or the CAD input, e.g. an ODB++ .tgz)
  analysis/              scripts: mesh settings, sweeps, plots, page templates
  results/               what analysis writes (.png, .sNp, .html, .json)
Models/CLAUDE.md         notes for running and post-processing models
```

- Analysis scripts put `../geometry` and `Engine/` on `sys.path` themselves:
  `python Models/<model>/analysis/<script>.py` works from any directory
  without installing (an installed mom25d is shadowed by Engine/).
- Mesh settings stay in the analysis scripts (the notch's S-parameter and
  current-map runs use different meshes of the same geometry).
- Engine tests do not import Models (they build their own small layouts);
  only `test_sample_design_stack` reads the ODB++ sample from
  `Models/odb_rigidflex_sample/geometry/` (skipped if absent).
- A proprietary board gets its own `Models/<board>/`, which must stay out
  of any shared copy / version control (board exports and anything derived).

## Setup on a new workstation

```bash
cd mom25d/Engine
python3 -m venv .venv && .venv/bin/pip install -e '.[odb,tri,fast,dxf]' pytest   # numpy, scipy, matplotlib, shapely, triangle, numba, ezdxf
.venv/bin/python -m pytest -q        # 112 tests (2026-10-12: 13.5 min with other jobs running); must all pass before/after changes
cd ..
MPLBACKEND=Agg Engine/.venv/bin/python Models/notch_filter/analysis/vs_openems.py       # ~1 min
MPLBACKEND=Agg Engine/.venv/bin/python Models/notch_filter/analysis/current_density.py  # ~20 s, N=2417
MPLBACKEND=Agg Engine/.venv/bin/python Models/stepped_impedance_lpf/analysis/sweep.py   # ~8 min, N=3173
```

- shapely (>= 2) is only needed for ODB++ import (`Engine/mom25d/odb.py`,
  `Engine/tests/test_odb.py` skips without it). Core solver: numpy/scipy/matplotlib.
- `Engine/tools/odb_import.py` needs a board export (an extracted ODB++
  tree); pass its path. Its `cache/` directory is regenerable (so is the
  sample's unpacked tree + cache in `Models/odb_rigidflex_sample/geometry/`). Board exports, their outputs and anything derived from them
  stay out of the repo (proprietary).

- Do NOT rely on a distro scipy: on the original machine (Ubuntu 22.04 / WSL)
  system scipy 1.8 was binary-incompatible with numpy 2.2. Always use the venv.
- `Engine/mom25d/__init__.py` sets `OPENBLAS_NUM_THREADS=4` if unset. With OpenBLAS's
  default thread count (22 cores, WSL) dense LU was ~80x slower. If LU looks
  slow on a new box, benchmark thread counts first.
- `.venv/`, `*.egg-info`, `__pycache__`, `.pytest_cache` are machine-local;
  don't copy them. Everything else (incl. `Engine/tests/data/`) is needed.
- **Venv for the Google Drive copy** (on a drvfs mount):
  `~/venvs/mom25d` (Linux side, python 3.10, numba/shapely/triangle/plotly/
  pytest). A venv cannot live on the Drive mount (drvfs: no symlinks), and
  `pip install -e Engine` fails there too (setuptools cannot write its
  egg-info on drvfs: "Operation not permitted"). So the deps are installed
  directly and `site-packages/mom25d_drive.pth` holds the path of `Engine/`
  (= what an editable install does). Use:
  `cd Engine && ~/venvs/mom25d/bin/python -m pytest -q -p no:cacheprovider`;
  `~/venvs/mom25d/bin/python Models/<model>/analysis/<script>.py`.
  `~/mom25d` (+ its .venv) is the old flat-layout clone, stale since 2026-10-03.
  - 2026-10-05: `~/venvs/mom25d` was missing on the WSL box and was
    recreated (python 3.10, plain `pip install numpy scipy matplotlib shapely
    triangle numba threadpoolctl ezdxf pytest plotly`) WITHOUT the .pth:
    run the tests from `Engine/` (`python -m pytest` puts the cwd first);
    model folders outside this tree find the engine via the `MOM25D_PATH`
    environment variable (= this folder), set by the user's scripts / shell.
    Before changing the engine: back up `Engine/` (no version control yet;
    archives in `~/mom25d_backups/`).

## Layout of the code

Bare module names are in `Engine/mom25d/`.

| file | role |
|---|---|
| `greens.py` | single-slab Sommerfeld Green's functions gA, gV, gW; tabulated + cubic spline (used whenever the model is one layer) |
| `layered.py` | `Stackup`/`Dielectric`, `LayeredGreens`: TL-form kernels for every pair of metal levels; `SectionedGreens` / `make_greens` for stacks with inner planes |
| `integrals.py` | Dunavant 7/3-pt rules; closed-form ∫1/R and ∫(r'-r)/R incl. out-of-plane observers d (Wilton 1984) |
| `mesh.py` | `Layout` (rectangles, shapely polygons, `Port`s, `Via`s, `Pour`s), tensor-grid triangulation, RWG edges, port cuts, via attachment, `Mesh.quality()` |
| `trimesh.py` | unstructured mesh (`kind="tri"`): structured port feeds + constrained Delaunay (Triangle) with `min_angle` |
| `freqinterp.py` | `FreqInterp`: Z filled at anchor frequencies, W = jw Z e^{+jk_e R} Lagrange-interpolated in between (`sweep(..., interp=)`) |
| `antipad.py` | `FrillModel`: via-through-plane antipads, ring-expanded aperture field, C / C_ff blocks |
| `lumped.py` | component values (R, L, C, series, parallel, text, Touchstone, series_from_s2p) |
| `_fast.py` | optional numba kernels for the fill (`group_vals`, `scatter`, `far_entries`, scipy-identical spline lookup); used when numba imports and `MOM25D_NUMBA` != 0 |
| `hmatrix.py` | `HOperator`: cluster tree, ACA far blocks, exact sparse near field, dense via/antipad parts, GMRES with near-field LU (`Solver(method="hmatrix")`) |
| `network.py` | `innerconnect` / `connect` / `renormalize`, `Network` (+ Touchstone writer); `SweepResult.connect(model, pairs)` |
| `solver.py` | `assemble` (Z matrix incl. via rows/cols), `via_inductance`, `via_impedance` (per-section via A-term), `PortModel`, `fit_waves`, `Solver.sweep`, `SweepResult` |
| `analytic.py` | Hammerstad-Jensen / Kirschning-Jansen / open-end formulas (reference + initial β guess) |
| `dxf.py` | DXF import (ezdxf): closed polylines with bulges + circles per layer in metres, keyhole-slit cleanup, `outline` (closes seams between touching outline pieces), `split_by_vias` (signal vs stitched ground copper), `header_units` (report only) |
| `Engine/tests/test_dxf.py` | synthetic DXF (header cm, data mm): bulge arc area, keyhole -> interior ring, signal/ground split, outline seam (~3 s) |
| `leakage.py` + `templates/leakage.html` | parallel-plate cavity leakage: `cavity_leakage` (per closed section: TEM power per via P_a = 1/2 Re(I_a* Z_TEM I), total, angular pattern, cavity voltage map), `leakage_case` / `leakage_view` (self-contained dark page: map + per-via table + polar + power budget). Lossless transition without stitching: cavity 93-97 % of 1-|S|^2 (rest = top radiation); resolution ~0.1 % P_inc per via (J0(ka) self term). Model: `Models/plane_transition/analysis/leakage_map.py` |
| `rfpcb/` (+ `Engine/docs/rfpcb.md`, `tests/test_rfpcb.py`) | board workflow on top of the solver: `board.json` config, DXF reader (painting rule, PORTS / GAP_PORTS lines, outlines, drill spans), stack-up, line sections cut at the port lines with nearest-RF ground ownership, merged groups and a complete model, gap / pad / shared-pad / GCPW-feed ports, optional inner plane with DXF voids, quasi-static circuit cascade, N-port de-embedding + part insertion, H-matrix complete model (saves port waves), viewers, simulation-manager submission; CLI `python -m mom25d.rfpcb board.json check|circuit|sections|complete|submit` |
| `fields.py` | derived fields on display grids: `plane_currents` (image + via-driven + parallel-plate plane currents), `efield` (E on any height), `current_at_quad`, `Grid` |
| `view3d.py` + `templates/` | generic Plotly HTML viewers for any solved model: `model_view`, `current_view`, `field_view` (cut planes), `sparams_view` (N-port S pages from .sNp / arrays), `wave_cases` |
| `Engine/tests/test_fields.py` | kernels vs images / interface conditions, -Int E_z = line voltage, ground return current = -strip current, viewers write (~15 s) |
| `Engine/tests/test_thick.py` | thick metal: twin mesh / walls, thick microstrip vs 2D FD, H-matrix == dense, per-sheet loss (~30 s) |
| `Engine/tests/test_port_options.py` | struve nan repair, `Port(z0_from="reciprocity")`, `Solver(z0_recip_min=)`, current-only fit (`v_weight=0`) on an inverted strip over a dummy ground (~7 s) |
| `odb.py` | ODB++ reader (features, symbols, surfaces, EDA nets, matrix, drill-layer vias), `find_ports`, `layout_from_copper`, `stitched_pour` |
| `Engine/tools/odb_import.py` | CLI: window + nets from ODB++ -> preview / sweep / .sNp (`--list-nets`, `--preview`) |
| `Models/*/geometry/*.py` | notch_filter, lpf (stepped-impedance, textbook sections), gcpw_pour, via_transition, plane_transition: `layout()` + `substrate()` / `stackup()` |
| `Models/notch_filter/analysis/current_density.py` | smooth (Gouraud) |J| maps + direction arrows, matched-port excitation |
| `Models/stepped_impedance_lpf/analysis/lpf_3d_view.py` | Plotly 3D HTML (true scale): trace + ground |J| in dB, checkboxes, freq select |
| `Models/stepped_impedance_lpf/analysis/ground_plane_current.py` | quasi-static ground return current J_gnd = -P_h * K of the LPF (2 freqs, ~1 min) |
| `Engine/tests/test_odb.py` | synthetic inch-unit ODB++ tree (incl. out-of-order surface hole) + solve |
| `Engine/tests/test_multilayer.py` | layered kernels vs slab / images, split substrate, stripline vs Cohn, via segments, top->inner transition vs openEMS (~45 s) |
| `Engine/tests/data/ref_multilayer_openems.py` | openEMS script for `transition_openems.txt` (~10 min) |
| `Engine/tests/test_planes.py` | inner PEC planes: sections vs separate stacks, parallel-plate via impedance, top -> via through L2 -> stripline with / without stitching (~40 s) |
| `Engine/tests/data/ref_plane_openems.py` | openEMS script for `plane_openems.txt` / `plane_stitch_openems.txt` (`SIM_DIR=` local tmp) |
| `Engine/tests/test_trimesh.py` | tri mesh: angle bound, port cuts, exact area, through line, notch vs openEMS, 45-degree line length, shunt via vs openEMS (~45 s) |
| `Engine/tests/test_gcpw.py` | GCPW ports: strip/via geometry, pour joins feed grounds, grouping, `port_grounds`, GCPW feeds vs openEMS (~50 s) |
| `Engine/tests/test_freqinterp.py` | Z interpolation: Lagrange window/weights, interpolated vs exact notch sweep (< 1e-3, exact at anchors), error report (~1.5 min) |
| `Engine/tests/test_antipad.py` | antipad frills: ring Hankel transforms, via split / unknowns, capacitance vs rings & hole size, trend vs openEMS (0.4 / 0.7 / 1.2 mm) (~2.5 min) |
| `Engine/tests/test_lumped.py` | lumped components: values / Touchstone, series & shunt R vs circuit, Woodbury == direct loading (2-port component), overrides, open gap, series C vs openEMS (~1 min) |
| `Engine/tests/test_hmatrix.py` | compressed solver: partition tiles Z exactly (random + grid points), matvec vs dense, S vs dense (GCPW + vias, + lumped C), auto method (~2.5 min) |
| `Engine/tests/test_fast.py` | compiled fill == numpy fill (dense GCPW + vias + loss, layered via transition, H-matrix matvec) to 1e-12 (~20 s) |
| `Engine/tests/test_network.py` | S-parameter blocks: interconnection vs ABCD cascade, renormalisation, `SweepResult.connect` + Touchstone round trip, attenuator `add_sparams` in a layout (~30 s) |
| `Engine/tests/test_vias.py` | via attachment, pour clearance/fence, shunt via vs openEMS (~35 s) |
| `Engine/tests/data/ref_vias_openems.py` | openEMS script for `via_shunt_openems.txt` / `gcpw_pour_openems.txt` (run in `~/opt/openEMS/venv`; ~5 + 10 min) |
| `Models/via_transition/analysis/transition_3d_view.py` | multilayer transition currents -> `transition_3d_data.json`; `transition_3d_template.html` (Plotly.js, `__DATA__` placeholder) -> `transition_3d_view.html` (~1.5 min) |
| `Models/plane_transition/analysis/plane_transition_3d_view.py` | 3D view of the inner-plane transition (L2 + antipad, stripline, stitching vias, antipad ring nodes): `plane_transition_3d_template.html` -> `plane_transition_3d_view.html`; artifact KbXzasNEzFdvmZ25po6K4v |
| `Engine/tools/odb_3d_view.py` | 3D view of any ODB++ board (copper layers at stackup heights from `ODB.stack`, outline, vias): `odb_3d_template.html` -> HTML. Board pages embed copper geometry: keep proprietary ones local. Sample (public ODB++ rigid-flex, 218k triangles, 7.8 MB, 27 s): artifact G8B625wjP7dNNdPFTWYMco |
| `Models/notch_filter/analysis/sparams_currents.py` | notch filter, port 2 terminated by matched / open / short .s1p blocks via `connect`: |J| maps from one solve (`sparams_currents.png`, ~30 s) |
| `Models/gcpw_pour/analysis/vs_openems.py` | via-stitched GND pour (GCPW) vs openEMS, ~5 min, N≈5000 |
| `Engine/tests/data/notch_openems.txt` | openEMS FDTD reference (MSL_NotchFilter tutorial: W=0.6, h=0.254, εr=3.66, stub 12 mm) |

Units are SI (metres, Hz); `mm`, `GHz` helpers exported. Time convention e^{jωt}.

## Hard-won decisions (don't undo without re-running the tests)

1. **Green's functions.** Asymptotic term c·e^{-jk̄r}/(4πr) is extracted with
   k̄ chosen so the spectral remainder decays ~k_ρ⁻⁵ (k_a² = (k0²+k1²)/2 for gA,
   k_v² = (εr k0²+k1²)/(εr+1), c_v = 2/(εr+1) for gV/gW). Path is lifted into
   the **first** quadrant (e^{jωt}), lift a = min(k0/2, 1/ρ_max) to keep
   J0(complex) growth benign. Integration is machine-precise; the only error
   source was interpolation: **linear interp gave 3e-4 errors, cubic spline
   gives 1e-7** — keep the spline.
2. **Line-voltage kernel gW~ = u0 tanh(u1 h)/(u1 D_TM)** (true −∫E_z dz, TM
   part, charge-driven). Using the scalar potential gV as "voltage" made Z0
   fall with frequency (gauge/A_z error ~(k1 h)², 3 % at 10 GHz). gW equals
   gV in the static limit (checked) and gives Z0 rising with f, as it should.
3. **Voltage sampling** is done half a cell on *both* sides of each cut and
   averaged: sampling on the cut gave a ±1.2 % even/odd ripple from the
   checkerboard triangle diagonals + piecewise-constant charge.
4. **Wave fit** (`fit_waves`): I = A e^{γs}+B e^{−γs}, V = Z0(A e^{γs}−B e^{−γs}),
   γ and Z0 shared by all excitations and all ports of equal width.
   - Fitting V together with I (weight `VOLTAGE_WEIGHT = 1`) is what makes β
     well conditioned when the fit window ≪ λ (I-only fit gave 2 % εeff error
     at 1 GHz).
   - **Lossless substrates: γ is constrained to jβ.** A free fit returned a
     spurious α≈0.28 Np/m at 10 GHz → |S21| = 1.0075 on a through line. With
     losses α is free, and that bias (≈0.2–0.5 % in |S21| near 10 GHz) remains
     — known open issue.
5. **Z0 normalisation between different-width ports** comes from reciprocity
   (`_reciprocal_z0`, TRL-style); the absolute level from the gW-based Z0.
   Using Hammerstad Z0 or gV-based Z0 violated unitarity by 1–2 %.
6. **Assembly speed.** numpy's batched `matmul` of complex × real (or
   non-contiguous complex) arrays falls into a pathological slow loop — it
   cost 45 of 70 s. Use `_cmatmul` (split real/imag) or `einsum`. Three
   interaction tiers (analytic+7pt < 3·size < 3pt < 10·size < centroid) change
   Z by <1e-6 relative. Z is symmetrised at the end (outer quadrature leaves
   ≤0.5 % asymmetry on adjacent pairs).
7. **Port defaults**: fit window margin = max(1.5 W, 4 h) from reference plane
   and gap; gap at W from the open end. Wide ports need longer feeds
   (error message reports the window).
8. The delta-gap feeds with open ends make the whole structure a closed
   resonator: raw delta-gap currents on unexcited feeds can ring. For field /
   current plots use `SweepResult.matched_currents(fi, port)` (needs
   `sweep(..., keep_currents=True)`); feed sections beyond the gaps are still
   unphysical.
9. **Plots:** the user wants smooth shading on 2D maps -- average triangle
   values to nodes (area-weighted) and use `tripcolor(..., shading="gouraud")`
   (see `Models/notch_filter/analysis/current_density.py`), not flat per-triangle colours.
10. **Polygons in `Layout`:** axis-aligned polygon edges become grid lines
   (merged into existing lines within `snap`, default max_cell/4); everything
   else is a staircase decided by cell-centre-inside tests. Port feed rects are
   added as rectangles so their edges are exact grid lines.
   **Trap (2026-10-10):** the default snap moves polygon edges by up to
   max_cell/4 (60 um at 0.25 mm cells). On a geometry-sensitive design (a
   broadside L1/L2 coupled-line bandpass filter, a few um matter) the grid
   then disagreed with the tri mesh by ~10 dB in worst return loss; with
   `snap=1e-6` the grid agrees with the tri mesh to |dS| ~ 0.1, the same as
   the tri mesh's own edge-cell refinement 0.03 -> 0.015 mm. Pass a small snap
   when edge positions matter.
11. **Geometry cache.** `AssemblyGeometry` (built once per `Solver`, reused
   by every `assemble` in a sweep) holds block order, near/mid/far masks and
   the closed-form near 1/R integrals. Z is bit-identical to the uncached path;
   fill at N=3173 went 17.4 -> 14.7 s/freq. Pairwise distances are NOT cached
   (~300 MB at N=3173, ~1 GB at N ~ 6000).
12. **Green's lookup stays on scipy CubicSpline.** A uniform-grid cubic
   Hermite table (numpy fancy indexing) was 1.7-1.9x *slower* at equal
   accuracy; beating it needs numba/C (done: decision 26 evaluates the
   same spline coefficients in compiled code).
13. **Ground-plane loss** (`Substrate.ground_loss`, default True): the
   ground's return current is J_gnd = -P_h * K (P_h = Poisson kernel of height
   h, image-doubled Biot-Savart, valid for k1 h << 1), and P_h * P_h = P_2h, so
   its dissipation is folded into the tabulated gA as + Zs/(jw mu0) P_2h(rho)
   (`GroundedSlabGreens(zs_ground=...)`), at no extra fill cost. Test:
   Z-term power == direct grid integral of |J_gnd|^2 (3e-6 rel). Effect on the
   50-ohm RO4003 line: conductor alpha +21 % at 2 GHz. Loss results with sigma
   (e.g. the ODB lost-power numbers below) were strip-only before this.
14. `fit_waves` (lossless) searches beta in 0.7-1.3 x the KJ estimate and now
   warns (RuntimeWarning) if the optimum lands on a bound.
15. **Vias** (top copper -> ground plane): one unknown per via (index
   n_basis + k, current flowing DOWN). Top side = half-RWG draining the
   triangles **inside the hole** (fallback: all cells touching the centre)
   with uniform charge; couples via gV like RWG charges. Vertical part:
   quasi-static partial L of via+image (2h wire, half the flux), no mutual
   L to horizontal currents (PEEC, Lorenz gauge). Traps hit:
   - Attaching to the whole pad (pad = W) stopped the strip current at the
     pad edge -> missing ~W/2 of line L each side -> via L_eff 0.019 vs
     openEMS 0.030 nH. Hole attachment: 0.029 nH, mesh-converged.
   - The via's top charge is not neutral, so the constant Im gV(0) gave a
     spurious ~0.09 ohm radiation resistance (lossless power 0.99 @10 GHz,
     0.97 @14 GHz). It is subtracted from the via-via block only (RWG
     charges are neutral, so cross terms are unaffected) -> 0.998.
   - Retarding the via A-term with homogeneous k1 made it worse (0.964):
     the thin slab radiates far less than a homogeneous dielectric. Keep it
     quasi-static (`via_inductance(..., k=0)`).
   - A via centred on a grid line must attach to the cells on both sides,
     or S11 != S22 by ~10 deg (a symmetric layout became asymmetric).
16. **Multilayer** (`layered.py`, all features take `layer=`):
   - Kernels: gA~ = V^h, gV~ = (V^e + k0^2 V^h)/kr^2 (normalised TL voltages,
     TE admittance u, TM eps/u), gW~ = sum_k dV^e_k/u_k^2. The +k0^2 sign was
     checked against the slab's closed form (a hand derivation had it wrong).
   - Leading term c e^{-jkR}/(4 pi R), R = sqrt(rho^2+dz^2): same level ->
     the slab's two-media c, k; cross level -> c from 2u e^{u dz} g~ at
     kr = 15/t_min, normalised with u (not kr) so air layers give exactly 1;
     gW with the source BELOW the observer has c = 0 (finite at rho = 0,
     only a tiny kr^-3 tail).
   - Single-layer models still go through GroundedSlabGreens (Solver.single)
     -> bit-identical S; the layered path on a split slab agrees to 3e-8.
   - Mesh: one shared tensor grid, separate nodes per layer (RWGs never
     join layers), tri_layer / layer_names. Assembly blocks are single-layer;
     near/mid/far use 3D centroid distance.
   - Vias are split into segments (Mesh.assign_levels, called by Solver with
     the stack order): connected = top, bottom, and in-between layers with
     copper at the hole. Segment inductance: Grover parallel filaments
     [z0,z1] + image [-z1,-z0]; equals the old 2h-wire/2 for z0 = 0. The Im
     gV(0) fix generalises to sum s_a s_b Im gV_ab(0) over end charges.
   - Ports on covered layers: beta search 0.7-1.8 x KJ (KJ with h = level
     height and eps averaged below underestimates eps_eff).
   - Fit windows: a 1.1 mm line with 6 mm feeds had a 0.8 mm window ->
     ~1-2 % eps_eff jitter (same on one layer). Use ~12 mm feeds.
17. **Pours** (`Layout.add_pour`) are resolved lazily at mesh time:
   region minus (all rects/polys/feeds/via pads).buffer(clearance, mitre);
   fence vias on pour.buffer(-(pad+drill)/2) rings at `via_pitch`. Pad
   circles are copper (`contains`) but add no grid lines. The pour shares
   the tensor grid -> unknowns grow fast; clip pours to a band.
18. **Inner PEC planes** (`Stackup(planes={"L2": k})`, `Layout.add_plane`):
   - A solid plane splits the stack into sections that do not couple. Each
     section is its own Stackup (`section_stack`: ground = plane below,
     cover = plane above) -> reuses the tested TL code; cross-section pairs
     are `_ZeroPair` and skipped in assembly. gW, `level_h`, `eps_below`
     and port margins / beta guesses are section-local (voltage to the
     plane below). Each section's ground loss is the plane below it.
   - Vias: segments are split only at layers they attach to (copper or
     `connect=` planes), so a via through an antipad is one unknown whose
     vertical current runs through several sections; segments ending on
     planes have no charge (`top_tris`/`bottom_tris` None, both for plane
     -> gnd stitching vias). Antipads are otherwise not modelled.
   - `via_impedance`: per section piece. Air top: unchanged quasi-static
     image L (bit-identical). PEC both sides (planes or a top="pec" cover):
     the image series DIVERGES (stacked collinear images = infinite wire,
     the radial parallel-plate line), so use modes: TEM w mu l_a l_b/(4d)
     H0(2)(k rho) + TM_n jw mu/(pi d) K0(q_n rho) c_n c_n. k from the
     series-averaged eps. Checked: full height = closed form, partial
     pieces = image series (differences between radii, 1e-4).
   - Checked that a lossless stripline gV has Im gV = 0 (no TEM pole in the
     charge kernel), so the radial-wave loss comes only from the TEM term
     -- no double counting with the Im gV(0) fix.
   - Planes are infinite: Re Z (TEM) = power escaping to infinity. Real
     boards have cavity resonances at the board edges; openEMS references
     need PML on all sides of the plane region to compare.

19. **Triangular mesh** (`layout.mesh(..., kind="tri", min_angle=20)`):
   - Port feeds stay structured strips (`_Feed`): cuts at <= s_edge, lines
     at the copper edges and edge_cell inside; `Mesh.port_cuts` replaces
     xg/yg in `cut_positions`, so PortModel/fit_waves are unchanged.
   - Rest: Triangle `pq{angle}a{max_cell^2/2}Y`. `Y` (no Steiner points on
     the outer boundary) keeps feed/region nodes shared = conforming; so
     the boundary must be pre-divided (s_edge = min(max_cell, edge_cell /
     tan(min_angle))) and region boundary segments lying on a feed use
     exactly the feed's nodes. `_check_conforming` refuses free edges
     inside copper (hanging nodes) -- RWG would silently cut the current.
   - Edge-offset constraint lines (copper.buffer(-edge_cell)) must run up
     to the feed (their ends become feed lines); stopping them short gave
     11 deg triangles at every junction.
   - Traps hit on real ODB++ copper: near-duplicate arc points -> nm triangles
     that collapsed in the node merge (non-manifold edges); micron steps at
     trace joins -> 0.7 deg slivers (boundary can't be refined with Y);
     snapped vertices 0.12 um from an automatic feed line -> a whole row of
     0.05 deg slivers. Fixes: set_precision(simplify(snap), snap) with snap
     = 0.05 * min(edge_cell, s_edge); tol >= snap; feed lines / region
     vertices closer than 0.15 * that are merged / snapped to feed nodes.
   - Isotropic angle bound was costly on Manhattan layouts (N x1.4-4: the
     grid's edge cells have aspect ~10). Fix: anisotropic edge band
     (`max_aspect`, default 5). Two Triangle passes on the same PSLG: band
     ("pYYQ", no refinement, interior marked as holes) and interior
     ("pq..a..YYQ", band marked as holes); YY = no Steiner points on any
     segment, so both share the band/feed nodes. Hole points of pass 1 must
     avoid closed loops inside the interior (drill hexagons) -- one landed
     in a drill and the interior was meshed twice (overlap check catches).
     Drill rings are hexagons (12-point circles cost ~40 % extra N on the
     GCPW pour). N vs grid: notch 1256/757, LPF 3369/3173, GCPW 4419/5068;
     on oblique board copper the tri mesh needed ~half the grid's N.
   - **shapely.set_precision attaches a precision model**: every later
     op (difference, buffer) then runs on that grid -> cut vertices off
     the feed boundary, and buffer(-snap) of an oblique 0.56 x 4 mm strip
     returned EMPTY (the whole line vanished from the mesh). Always
     set_precision(g, snap) then set_precision(g, 0).
   - Oblique ports (`Port.direction` = angle / vector; `u`, `n`, `point`,
     `local`, `corners`): cut_edges / PortModel / _Feed use (s, t) feed
     coordinates; axis-aligned ports keep bit-identical behaviour. Grid
     mesh refuses oblique ports. Rotation test: 0/17/30/45/60 deg identical
     to 5e-4 in S.
   - **The w/2.5 staircase of the grid is wrong for 45-degree traces**:
     +0.9 mm electrical length on a 6 mm diagonal (w/4: +0.16), lossless
     |S21| 0.98. The tri mesh is converged (-0.07 mm = mitred bends). On a
     board window with 45-degree traces this was a 20-50 deg S21 phase
     error at 15-35 GHz -- use --mesh tri for non-Manhattan boards.

20. **GCPW ports** (`Port(ground=CoplanarGround(gap, width, via_pitch, ...))`):
   - Side ground strips (+ fence vias on their centre lines, from s =
     pitch/2) along the whole feed, added by `add_port`. They are flagged in
     `Layout.ground_rects/ground_polys` and their vias skipped in
     `resolved_pours`'s "signal": otherwise an `add_pour` ending at the
     reference plane clears G around them -> the pour and the feed grounds
     do not touch (disconnected return path at the ref plane).
   - Port cuts / voltages stay on the centre strip only (cut_edges uses
     |t| < W/2), so PortModel and fit_waves are unchanged; the mode is
     single as long as the grounds are via-stitched.
   - Port groups key now includes `ground.key(h)`: a GCPW and a microstrip
     port of the same width are different lines (eeff 2.72 vs 2.86 on the
     test line) and must not share gamma / Z0.
   - ODB: `port_grounds` measures (left, right) gap and width of the pour on
     the window edge in feed coordinates (left = +t = u rotated +90 deg).
     Fence pitch default = median nearest-neighbour spacing of the pour vias
     drill = their median.
   - Check: pour stopped at the ref planes + GCPW feeds == pour everywhere to
     0.01 deg / 2e-3 in S; microstrip feeds on the windowed pour extract the
     microstrip line (Z0 50.6 vs 47.8 ohm) and miss openEMS by 0.2-1 deg
     more.

21. **Frequency interpolation** (`FreqInterp`, `sweep(interp=)`):
   - Interpolate W = jw Z exp(+j k_e R_mn) (R = 3D distance of basis
     centres: edge midpoints, vias at mid-span), not Z: removes the 1/(jw)
     of the charge term and the far-pair phase. Benchmarks (max|dS|, notch
     1-9 GHz): cubic / 2 GHz 2.4e-4, quadratic / 2 GHz 1.1e-3; on a larger
     (37 mm) layout k_e = 0 was similar in max but 5x worse in median.
     Defaults: cubic, k_e = sqrt(eps_eff) k0, step 0.35 c / (k_ratio D).
   - **Precision trap:** at 1 GHz the Z of the notch mesh has condition
     ~1e5, so a 2e-8 relative change of Z gave 3e-3 in the currents and
     5e-5 in S. Two sources: float32 distance R made exp(jkR) complex64
     (round trip not exact) -> R must be float64; complex64 anchor storage
     (1.7e-8) -> complex128 unless memory forbids.
   - **Via-via block: no phase extraction.** It holds the large Im gV(0)
     correction (decision 15), identical for every via pair whatever their
     distance; exp(+jkR) on it made that block interpolate to 0.6 % (GCPW
     line) / 6.4 % (a pour with 141 vias) while the RWG block was at 1e-8.
     Fixed by R = 0 for via-via pairs (-> 7e-11). `report_error` prints the
     error per block: one Frobenius norm is dominated by the via entries.
   - Window = anchor interval holding f, centred (monotone in f): the
     earlier "nearest anchors" window flipped back and forth and refilled
     evicted anchors (7 fills for 5 anchors).
   - Cost per interpolated point ~6 s at N 8980 (4 complex matrices +
     phase, row blocks) + LU 15 s, vs fill 82 s. Gain is small at N ~ 2000
     (solve + combine ~ fill), large at N ~ 9000.
   - The sweep ETC counts the remaining anchor fills separately (they come
     in bursts; the plain average said 320 min after the first point).

22. **Antipads** (`add_via(antipad=)`, `antipad.py`, experimental, 2026-10-03):
   - Formulation: close the hole, put M = E x n on the aperture (+M above,
     -M below), axisymmetric E_r = -dU/dr with U on `FRILL_RINGS` (4)
     piecewise-linear rings (geometric nodes a..b, ring 0 = 1 for r < a).
     Via split at the plane (two segments, no charge there). Testing H
     continuity: circulation of H at the plate = conduction (via, ring 0
     only) + jw x induced plate charge inside the loop -> couplings to
     charges via Phi_j = Int sigma_1 w_j (sigma_1 = TM current into the
     plate short per unit source, normalised so the static half-space case
     gives -e^{-kh}), vertical via currents in closed sections via modal
     E_z^A, rings' capacitance Int eps0 k^2 y_in w~_i w~_j J0 k dk / 2 pi.
   - **Sign trap:** Z = -<f, E(J)>, so the frill enters the current rows as
     -C V; with +C the hole made |S11| drop to 0.06 and the S21 phase 18 deg
     off. Symmetric system [[Z, -C], [-C^T, -jw C_ff]].
   - **Quadrature trap:** the ring Hankel transforms must be exact on the
     real axis (closed form with Struve H0/H1); a 24-pt Gauss rule cannot
     follow J0(kr) up to k = 400/a and gave capacitances of 63 000 fF.
   - **One ring (coax log profile) is not enough** when b - a is not small
     next to the layer heights (b 0.35 > H 0.254): capacitance did not fall
     with hole size. 4 rings: 61.6 / 48.9 / 45.5 fF for 0.4 / 0.7 / 1.2 mm
     (8 rings within 2-4 %).
   - Zero-thickness plane: gap capacitance grows only like ln(1/gap) as
     b -> a, so a 2 um clearance does not short the via (|S21| 0.94 @9 GHz
     vs 0 with connect=). Plate thickness is not modelled.
   - vs openEMS, refined FDTD mesh (`fine20` files; stitched, 15 GHz):
     0.7 mm: S21 phase within 2.2 deg, |S21| 0.011, |S11| 0.28 vs 0.22.
     0.7 -> 0.4 mm: S21 -5.4 vs -6.5 deg, |S11| 0.37 vs 0.38 (the normal
     FDTD mesh gave -8.7 deg / 0.43: one cell across the 50 um gap, not
     converged; refinement moved openEMS 3.7 deg toward the model).
     0.7 -> 1.2 mm: +6.5 vs +4.1 deg, |S11| 0.17 vs 0.05 (openEMS converged
     here: normal 0.057). So large holes (b >> pad) overshoot: only the
     m = 0 aperture mode; a hole extending under a trace makes its L2
     return current detour -> cos/sin phi modes (magnetostatic coupling to
     horizontal currents) are the next step if large antipads matter.
     Note: without the antipad model the 0.7 mm S21 phase is closer (0.9
     deg) but |S11| worse (0.33 vs 0.22).
   - Refined openEMS references (`ref_plane_openems.py plane_stitch <anti>
     1e-4 20` -> `*_fine20_openems.txt`, ~35 min each, 8.5 M cells):
     `fine` (third-rule spacing) 20 um + third-rule lines at drill / pad /
     antipad radii + 10 z cells per layer. A first attempt overlaid a
     uniform 20 um box on the existing lines: 3 um cells (3.6x smaller dt)
     and, with an exact z grid, a transition that did not conduct (|S11| =
     1, energy stuck at -6.9 dB) -- deleted.
   - Speed (2026-10-04): `FrillModel.blocks` 2.6 -> 0.3 s per frequency
     (one 0.7 mm antipad, 4 rings; C, F identical to 4.5e-16). The path has
     ~22k points (real-axis tail to 400/a for the ring self terms), only
     ~1200 of them complex: complex `jv` only there, real `j0` elsewhere
     (5x cheaper). `_charges` drops points where the level -> plate
     transfer |sig| ~ exp(-k dz) < 1e-16 max (~90 % of the tail). Ring
     transforms are computed once per (frill, side) and share J0 / J1 /
     Int J0 (Struve) values at the node radii between rings (28 -> 10
     Struve evaluations of the whole tail). Antipad tests 53 -> 19 s.

23. **Lumped components** (`Layout.add_lumped`, `add_component`, `lumped.py`):
   - Geometry: a bridge Port (a -> b, width) meshed as a structured feed
     strip (trimesh `ports` list + grid lines at 0, s_gap, length); the
     delta gap at its middle is an internal port; l = -sgn * edge_len
     (current a -> b), `Mesh.lumped_vectors`.
   - Loads: Z_L across a gap adds Z_L l l^T. Solve [E | L] with one LU,
     then x = Xe - Xi D (I + L^T Xi D)^-1 L^T Xe (Woodbury variant valid for
     D = 0; D may be a full N-port Z-matrix). Open gaps: D = 1e12 ohm.
     Checked == direct (Z + L D L^T) solve to 5e-11.
   - `sweep(values={name: value})` overrides; text values parsed by
     `lumped.parse` ('R 10 + L 0.5n', 'C 1p || R 1k').
   - ODB: `odb.components` parses comp_+_top/components (CMP, PRP VALUE /
     TYPE, TOP pins with net index into eda/data NET order, UNITS line);
     `component_value` reads '3.3Kohms', '100nF', '4R7', '22p' (bare
     numbers only for R -- '70' on an inductor is ambiguous);
     `place_components` bridges 2-pin parts whose pins are on the modelled
     copper inside the window (width 0.8 x narrower pad, measured across
     the bridge). On the public sample: 500 two-pin parts, 285 with a
     readable value (most others have an empty VALUE).
   - Validation: series R (grid & tri) and shunt R via a grounded pad vs
     the ideal circuit (< 1e-3 / 0.02); 1 pF vs openEMS LumpedElement 1.5-15
     GHz: |S21| 0.014, |S11| 0.017, 1.2 deg.

24. **Compressed solver** (`hmatrix.py`, `Solver(method=)`; auto above
   `Solver.HMATRIX_N` = 7000 unknowns or when 3 dense Z do not fit in 60 %
   of free memory):
   - Partition must be symmetric: split the cluster with the larger
     (diam, id) -- with ties broken by order, (a,b) and (b,a) split
     differently on regular grid meshes and far blocks applied with their
     transpose covered parts of Z twice (LPF |dS| 0.155). Test tiles Z.
   - Admissibility also needs distance >= ADM_SIZES (5) x the largest cell:
     on long thin grid cells a diameter-only test admitted pairs 2 cell
     lengths apart, where the 3-point rule is wrong (LPF |dS| 2.4e-3 ->
     1.3e-5). ACA tolerance and centroid vs 3-point did not matter.
   - Near blocks: per leaf, test triangles of its unknowns vs source
     triangles of all neighbour-leaf unknowns with the dense tiers
     (`AssemblyGeometry(blocks=False)` skips the O(N^2) dense setup);
     entries outside (leaf rows) x (neighbour cols) are dropped.
   - Far: blocks with a side <= 64 stored dense; ACA rank cap min/3.
   - **BLAS threads:** mom25d's OPENBLAS_NUM_THREADS only works if numpy is
     not imported yet; otherwise OpenBLAS ran on 22 threads and a 400 x 200
     complex matvec took 2.7 ms instead of 21 us (far matvec 666 -> 5 ms).
     `__init__` now also calls threadpoolctl.threadpool_limits (new
     dependency). This affected every earlier benchmark that imported
     numpy first.
   - Preconditioner: LU of near couplings within PREC_SIZES (6) cell sizes:
     full near LU 6.1 s / 107 its, 6 sizes 3.7 s / 187 its, 3 sizes 0.8 s /
     3103 its (GCPW N 4419, 3 GHz). Full near LU at N 10155 took 99 s.
   - N 10155 (GCPW 40 mm, 208 vias): |dS| 3.1e-6 vs dense; per frequency
     H 274 / 164 s vs dense ~120 s (fill 90 s + LU 20 s): equal-ish speed,
     36 / 20 % of dense storage at 3 / 9 GHz. GMRES 142 its at 3 GHz, 28
     at 9 GHz (low-frequency conditioning).
   - N 25397 (GCPW 120 mm, 528 vias; dense Z alone 10.3 GB): 519 s at 3
     GHz (far 214, near 61, vias 41, LU 17 s, GMRES 334 its), 230 s at 9
     GHz (27 its); 19 / 9.4 % of dense storage, ~7 GB peak. |S21| 0.9289 /
     0.8572 = (40 mm dense result)^3 0.929 / 0.858.

25. **S-parameter blocks**: (a) `SweepResult.connect(model, [(port, k)])`:
   network algebra after the sweep (renormalise to the block's z0,
   interpolate its S, join ports with the sub-network formula; == ABCD
   cascade to 1e-10). (b) `Layout.add_sparams(name, file, [(sig, ref)...],
   width, ground_pad=)`: gaps sig -> ref per block port + `add_component`
   (Touchstone Z-matrix as the Woodbury load); optional square pad + via to
   gnd at each reference point. 6 dB attenuator between two line halves:
   |S21| 0.498-0.501 (1-10 GHz), |S11| 0.01 -> 0.11 from the pads / gaps /
   vias. `Touchstone.s(f)` gives interpolated S (data in `.s_data`).
   Trap in the test: the block's signal points must lie on copper (first
   attempt put them in the empty gap: S21 = 2e-4).
   (c) Currents with connected blocks: `Network.currents(fi, port)` solves
   for the incident 50-ohm waves at all layout / block ports (drive the
   chosen port, block S at connected pairs), converts layout waves to the
   line-referenced current waves A = (I + V / zr) / 2 (I = A + B, V = zr
   (A - B), zr = res.z0) and superposes the stored excitation solutions
   (`SweepResult.wave_currents`). Open / short at P2: I / V at the
   reference plane ~0 (fit of the feed standing wave, < 2 %).

26. **Compiled fill** (`_fast.py`, numba optional, 2026-10-04):
   - One kernel per (test block, source group) does distance + spline +
     inner and outer quadrature + RWG coefficients, writing vals[p, i, s, j]
     (parallel over source triangles: each writes its own slice); a serial
     `scatter` adds them into Z (two triangles share an RWG, so a parallel
     scatter would race). It writes into the Z[:nb, :nb] view directly -- a
     contiguous copy would cost another 16 N^2 bytes.
   - The spline is evaluated like scipy PPoly (binary-search interval,
     sum c[3-k] dx^k, end intervals extrapolate) -> Z identical to the numpy
     path to 1e-19 (single layer) / 1.5e-16 (layered); test_fast checks 1e-12.
   - `far_entries` loops over the flat (row, col) index: ACA asks for one
     row / column at a time, a prange over rows ran on one thread. `_aca`
     now keeps U, V in preallocated arrays (rank updates as matvecs); its
     Python loops cost as much as the compiled evaluator before.
   - Threads: default 8 (`MOM25D_THREADS`); 22 on the WSL box was 2x slower
     than 4-16. Threading layer OpenMP first (the installed TBB is too old,
     numba warned on every run).
   - Speed (GCPW + via fence, 5 GHz): dense N 3117 10.4 -> 1.2 s, N 7137
     43.7 -> 5.8 s; H-matrix N 4469 28 -> 3.9 s, N 7137 50.8 -> 6.1 s.
     Test suite 8 -> 4 min. Now dominant: dense LU, layered Sommerfeld
     tabulation, `potential_integrals` in the geometry setup (once per
     sweep); antipad blocks fixed separately (decision 22, last bullet).

27. **scipy struve nan** (`antipad._struve`, 2026-10-05): scipy 1.15's
   `struve(0, x)` returns nan exactly at some zeros of H0 (x = 25.7653728:
   a 0.95 mm antipad at 16 GHz). One nan in the closed-form ring spectra made
   the whole frill block nan and the sweep crashed in CubicSpline ("y must
   contain only finite values"). Isolated nans are replaced by the mean of
   the values at x -+ 1e-4 (error ~1e-9); bit-identical otherwise.

28. **Ports with an invalid line voltage** (2026-10-05). The line voltage is
   -Int E_z from the plane BELOW the strip. Where the strip's return is the
   plane ABOVE it, that is not the line voltage (path through the region
   below, A-term ~ (k d)^2): Z0 drifted 48 -> 23 ohm over 15-40 GHz and the
   V-weighted fit biased eps_eff by 3-4 %. Case: a board with open-air
   microstrip on BOTH outer layers, modelled flipped with a dummy PEC
   ground 1-4 mm under the lower outer layer (mom25d needs PEC at the
   bottom; the top side stays exact). Options (defaults unchanged ->
   bit-identical S on the reference models):
   - `Solver(v_weight=0)`: currents-only wave fit. Fine when the fit window
     is >~ lambda_g / 2 (the inverted line then matches the exact one to
     0.1-0.8 % in eps_eff); poor when the window << lambda_g.
   - `Port(z0_from="reciprocity")`: no scalar-potential prior for that port
     group; Z0 = coupling-weighted mean of Z0_j SI_ji / SI_ij over the
     "voltage" ports j (`_reciprocal_z0`, which now runs the old fit on the
     voltage ports only, `_voltage_z0`).
   - `Solver(z0_recip_min=c)`: reciprocity rows only for min(|SI_ij|,
     |SI_ji|) >= c. Needed because couplings to a badly de-embedded port
     (fit window in the near field of a stub / resistor 1.1 mm from the
     reference plane) were non-reciprocal by up to 0.28 at the band edges
     and, at weight |SI|^2 >> prior 1e-4, pulled a 79-ohm line voltage Z0
     to 90-116 ohm. A port's raw non-reciprocity is the first thing to check
     when such terms look odd: |S_ij - S_ji| per pair vs frequency.
   - Dummy-ground side effects: |S21| of a lossless 12 mm line 0.9976 vs
     1.0019 exact at 30 GHz (power into the air parallel plate), eps_eff
     insensitive to the air height 1-4 mm.

29. **DXF import** (`dxf.py`, 2026-10-05): from an HFSS layout export. Its
   header said cm for mm coordinates (pass `scale`, never trust
   $INSUNITS); planes had keyhole holes (zero-width slit -> make_valid +
   buffer(0)); the substrate came as two polygons whose shared edge did not
   union (a MultiPolygon -> code taking the largest piece silently dropped a
   third of the board from the plots): `outline(seam=)`. Vias are CIRCLEs;
   copper containing a via centre is ground (`split_by_vias`).

30. **Thick metal, Sonnet-style** (`Layout.add_thickness(layer, twin)`,
   2026-10-05): the layer's copper is meshed twice -- the twin level is a
   metal level of the Stackup one copper thickness away (thin air /
   dielectric layer), same section, nothing in between -- and joined by
   one side-wall basis per boundary edge.
   - Mesh (`_thicken`): the twin copies the layer's triangles (new nodes,
     same coordinates and vertex order; `twin_tri`, `twin_layer`), so
     ports, lumped bridges and feeds need no change: `cut_edges` takes the
     edges of both sheets (never walls), i.e. delta gaps and port currents
     span the full thickness.
   - A wall is a "bent RWG" (`Mesh.wall`, last `n_walls` bases): T+ = the
     boundary triangle, T- = its twin, same local vertex; the free edge
     slot of each triangle holds it, so charges, horizontal half-currents,
     fill, near/far tiers and current plots are the ordinary RWG code. Its
     vertical current l x (uniform over the l x h strip) couples only to
     other walls (`wall_impedance`: Grover filaments, self at the strip's
     GMD 0.2235 l, first images in the section's PEC planes, pairs within
     `WALL_CUTOFF` = 40 h; sparse) -- the via convention (no H-V mutual L).
     Loss Zs h l.
   - Chose bent RWGs over via-type segments: they keep the horizontal
     half-current in the edge triangle and stay in the compressed RWG
     block (via unknowns are dense rows in the H-matrix).
   - H-matrix far entries of walls use per-half kernels
     (`HOperator._eval_halves`; two levels per basis). Dense == H-matrix to
     2e-6 in S on the test line.
   - Loss: each sheet is one face of the copper, a slab of t/2
     (`Solver._zs`: zs per RWG basis; DC: 1/(sigma t) in parallel). On a
     50-ohm microstrip, alpha 0.415 -> 0.330 Np/m at 5 GHz (35 um).
   - Accuracy: 1.1 mm line on 0.508 mm eps 3.55, 2D FD reference (quasi-
     static, 4 um grid): t = 36 um dZ0 -1.116 ohm, d eps_eff -0.0409 (HJ
     -1.10 / -0.041); t = 0.1 mm -2.42 / -0.089. Two sheets: 89-91 % at
     35 um, 84-88 % at 0.1 mm (edge cell 0.03-0.08 mm). The rest is the
     side-wall charge between the sheets -- more sheets would close it.
   - Walls on different levels at the same x-y position (an L1 and an L2
     strip edge on one grid line of a Manhattan mesh, both levels thick) gave
     d = 0 in the filament mutual (0 * inf = NaN, NaN port z0). Fixed
     2026-10-10 in `wall_impedance`: pair distance floored at 0.2235 x the
     shorter wall length (the strip GMD the self term already uses). Tri
     meshes unchanged (|dS| 6e-8); test `test_coincident_walls_two_levels`.
   - Cost: N roughly doubles (STE block 7.3k -> 15.4k, 1121 walls). The
     thin sheet pair also slows GMRES: 300 -> 200 iterations at 15-16 GHz
     vs 80 thin (falling with f, low-frequency-type conditioning); a
     loop-star / Calderon preconditioner (roadmap 4) would help most.
   - Green's tables: the thin layer sets t_min, so `_build_path` / the rho
     table get ~3x longer; still negligible next to the fill.

31. **Derived fields and generic viewers** (`fields.py`, `view3d.py`,
   2026-10-05; first built per model for the STE block, then moved here).
   - Grids: charges / currents are deposited on a dx / 2 grid that covers
     the display grid AND the whole mesh (`_extended`); clipping points into
     the border cells instead gave a ground current 30x too large.
   - Plane return current: the static Poisson-kernel image (-P_h * K) loses
     exp(-beta h) of a travelling wave's return current (9 % on 0.5 mm at
     5 GHz; ~18 % for 0.185 mm at 30 GHz): magnetostatics leaves out the
     displacement current. Use the retarded image of a homogeneous medium,
     -T * K with T = exp(-h sqrt(k^2 - k1^2)) by FFT (`_image`): within 2-3 %
     of the strip current, ~2.6 deg phase (inhomogeneous medium).
   - Via-driven plane currents: open sections I/(2 pi rho) h/sqrt(rho^2+h^2);
     closed sections without copper the TEM radial wave -jkI/4 H1(2)(k rho).
     Checked on the STE VT: segment currents 0.12-0.13 A along the VT, the
     16-via cage returns ~half.
   - E field: formulation-C scalar potential (gV~ = (V^e + k0^2 V^h)/kr^2)
     at any height -- TL voltages inside a layer from its two interface
     values (sinh form, overflow-safe exponentials), dV/dz analytic, open top
     V_N e^{-u0 s}; leading static term c/(4 pi R) extracted (c = 2/(eps_a +
     eps_b) of the source level). Checked vs images (eps 1, < 2e-3) and phi /
     D_z continuity at a dielectric interface. -jw A and antipad apertures
     left out (~(k1 h)^2). -Int E_z dz under a strip = line voltage (gW) to
     8 %. Single-slab models are re-tabulated in the layered form.
   - Viewer traps: Plotly gl3d mixes depth order as soon as a trace in the
     scene has opacity < 1 (even hidden ones): coloured surfaces must be
     opaque; only the geometry view keeps see-through planes. Remove the
     axis spike lines (`showspikes: false`). Equal x / y scale needs fixed
     axis ranges + aspectratio from them. Plane meshes with the same holes
     are shared (`pmeshes`): 10 refined copies made a 7 MB page.
   - Field viewer: 0 dB = 99.9th percentile of |E| (point-charge peaks next
     to the copper otherwise set the scale); heights `gap` = dx/4 away from
     metal levels / planes, none inside thick copper. STE block: 193 x 211 x
     66 points, ~35 s per case, ~3.7 MB per case in the page.

32. **Series .s2p beyond the data** (`lumped.series_from_s2p`, 2026-10-05):
   vendor data often stop below the sweep (Murata GRM0225C1HR20: 30 GHz,
   STE sweep to 40 GHz); `Touchstone.s` raises outside the band. Default
   now: a series R(f) = R0 + R1 f, L, C fitted at f >= f_max / 6 (X by
   least squares on the relative error with scaled columns w/1e9, 1e9/w --
   unscaled, lstsq returned C = 1e47 pF) continues Z outside; in band the
   data are used as before. `parse` accepts a .s2p path, so `add_lumped(...,
   "part.s2p")` works. GRM0225C1HR20: C 0.2001 pF, ESL 0.149 nH, SRF 29.2
   GHz, |X| fit error < 0.1 ohm, continuous at 30 GHz.

33. **Plane apertures** (`aperture.py`, `Layout.add_aperture`, 2026-10-07):
   openings (slots, keep-outs) in solid inner planes. Closed with PEC, +M above
   / -M below on an RWG mesh of the opening (rim edges excluded), H continuity
   tested -> [[Z, B], [B^T, Y]], B = [D_up^T; -D_dn^T] (reciprocity per
   section). Kernels in the TL form of layered.py: J -> H at the plane
   G = h_a Rot - h_s Q(phi) (S0 / S2 of (I^h +- I^e)/2, static e^{-k dz} part in
   closed form); M -> H: dual of formulation C (g_e~ = y^e, g_q~ = (y^h + k0^2
   y^e)/k^2), assembled by `assemble` with gA = g_e, gV = g_q, times -eps0/mu0.
   Vertical currents: open sections static Biot-Savart + images; full-span vias
   in closed sections TEM H = k I/(4j) H1(2)(k rho).
   - Near J-M pairs (dz ~ cell size): fine rules (sub-triangles <= dz) in a
     numba kernel (`_fast.aperture_near`): 220 s -> 25 s on the test line;
     finer rules (dz/4, level 4) changed nothing.
   - **The opening needs its own fine mesh** (default min(max_cell, 2.5
     edge_cell), edge_cell / 2): with the layout's 0.3 mm cells a 0.6 mm slot
     under a CPW gap was +0.16 in eps_eff, 3 ohm low; 0.1 / 0.02 mm converged
     (0.05 / 0.01 within 0.05 ohm).
   - D is sparse: metal further than CUT_DZ (25) dz from the opening and
     entries < 1e-7 max are dropped (25 dz moved eps_eff 0.009 on the test
     line; raise it for accuracy). H-matrix keeps it as a sparse block (Zab).
   - The line voltage (gW, w.r.t. the plane below) ignores M and is not
     proportional to the true voltage over an opening: fit residual 1e-2.
     With apertures the solver defaults to v_weight = 0 (current fit; feeds
     >~ lambda_g / 2) and Z0 must come from an internal port (reciprocity,
     automatic) or Solver(z0=).
   - Validation (tests/test_aperture.py, 2-D FD tests/data/aperture_fd2d.py):
     GCPW 0.30/0.25 on 0.076 mm over a plane on 2 mm of air (slot carries the
     field): solid / slot half-width 0.2 / 0.4: MoM 36.6 / 72.5 / 109.7 ohm vs
     FD 35.6 / 71.6 / 109.8 (MoM ~1 ohm above FD already for the solid plane:
     via rows vs wall). Over a 0.079 mm PPW (L2-L3, vias connected): slot
     half-widths 0.05-0.45 track FD within the same ~1 ohm offset, except the
     slot edge under the ground edge (0.4): Z0 0.6-0.9 ohm low / eps_eff +0.02-0.03
     relative to no plane, where FD says ~0. Static via coupling gave +0.009;
     the TEM (H1) coupling is the consistent one (via_impedance is TEM too),
     FD has no PPW dynamics -- unresolved, openEMS check open.
   - Thick planes (lo, hi): one M shared by both faces collapses the opening's
     interior (E_z across the foil lost): 18 um L2 -> Z0 4 ohm low (FD: thickness
     <= 0.3 ohm). Disabled (ALLOW_SHARED_THICK). Proper model: M per face, the
     interior section (closed PPW, foil thickness) coupling them, vertical wall
     currents on the rim (E_z = 0 at the plate edge); not done.

34. **Robust runs and tuning support** (2026-10-12, `solver.py`, `morph.py`, `view3d.py`; tests
    `Engine/tests/test_robustness.py`). Motivated by a 600-evaluation MoM tuning of a hairpin filter on the
    shared laptop:
    - `Solver.sweep(..., checkpoint=path)`: an .npz (key = frequencies, N, node sums, values) written atomically
      (tmp + os.replace) after every frequency; a restart with the same key resumes, skipping done points
      (keep_currents solves everything again). Two OOM kills had cost 40 min and a 30-min sweep.
    - `Solver.memory_estimate()` (3 x 16 N^2 dense + FreqInterp anchors) and `sweep(wait_memory=s)` /
      `$MOM25D_WAIT_MEMORY`: wait for MemAvailable, then MemoryError; default 0 = warn only. The "auto" method
      choice happens at Solver() time, but other jobs grow afterwards.
    - `sweep` prints `PROGRESS k/n` after each frequency (simulation-manager ETC).
    - `Layout.mesh(kind="tri", like=old_mesh)` -> `morph.morph_mesh`: same triangles, nodes moved with the copper
      (vias rigid, edge nodes keep their segment parameter, interior nodes IDW^2 of the piece's edge nodes);
      checks orientation, half the smallest angle, free edges on the new boundary, area within 1 %; else a fresh
      mesh (`mesh.morphed`). Removes re-meshing noise (~1 dB RL between neighbouring designs) from FD / LM steps.
      Needs corresponding layouts (same polygons/vertex counts, rects, ports, vias; no pours / lumped / apertures).
      `Mesh._raw` keeps the pre-thickening triangulation for it.
    - Viewers: `theme="auto"|"dark"|"light"` (default `$MOM25D_THEME`, else auto) pins the colour scheme.
    Not done (open): per-piece mesh density (coarse ground band, fine strips — needs a per-piece edge band in
    trimesh); adjoint sensitivities (dS/dp from one transposed solve, using the morph node velocities).

35. **Internal ports: lossless dielectrics, aperture display** (2026-10-14, `solver.py`, `mesh.py`,
    `view3d.py`; tests `test_internal_ports.py::test_tand_floor_with_internal_ports`,
    `test_aperture.py::test_view_planes_show_apertures`). From
    validating gap ports on a uniform GCPW line (0.44 mm line, 0.41 mm gaps, thin sheet over a 205 um eps 3.06
    stack, 20-38 GHz) against GCPW feed ports:
    - **Internal ports depend on tan d being exactly 0.** With every dielectric exactly lossless, S11 of the gap-port line was
      -7.7 dB; tan d 1e-5 gave -15.1 dB and the real 0.004 gave -15.2 dB (continuous), and the lossless results
      were non-passive at some frequencies. Feed ports did not change. Copper loss alone (sigma, tan d 0) did not
      help, so it is the dielectric (real-axis poles of the layered kernels), not the conductor. Root cause not
      found: the Woodbury load and a direct (Z + L D L^T) solve gave identical S. NOT a universal fix: with tan d
      1e-5 `test_via_port` turned non-passive (row sums of |S|^2 up to 1.066, |S33| 1.03), so the internal-port
      result depends on the kernel path (exactly lossless or not) in both directions. `Solver(tand_floor=1e-5)`
      is therefore opt-in (default None: unchanged); `Solver.lossless` keeps the user's value for the feed fit.
      Validate internal ports on a uniform line for the stack in use. Open: find the cause.
    - Bridges that only touch the copper edges are fine: a gap port whose bridge ends exactly on the line end /
      pad edge gave the same S as one reaching 40 um into the copper (tri: identical; grid: within 0.2 dB at
      20 GHz), so no overlap is needed (a suspicion during the investigation, checked and dropped).
    - **Ground terminal of a gap port.** On the uniform line a gap port to a separate pad stitched to the ground
      plane, or to the coplanar pour across a slot at the line end with a >= 2 mm ground band, was passive and
      symmetric (|S11|-|S22| < 0.02, missing power within ~1 % of feeds). A narrow (0.95 mm) truncated ground band
      left one-via ground fragments that resonate (a 1.6 mm, one-via piece: a quarter-wave stub at 29 GHz) and
      made results look like port failures. Gap ports still carry the bridge/slot parasitics (S11 -10..-18 dB on
      a matched line): de-embed with straight reference lines (two lengths, reciprocal error box per frequency).
    - **Gap ports with an aperture ground plane did not work**: a line in an opening of an inner plane, port
      bridged to a patch stitched to the plane by short vias, gave invalid S even on a uniform line (70-85 %
      missing power lossy). Not resolved; see open items.
    - `view3d`: plane apertures are now cut out of the displayed planes (`_tri_poly(cut=...)`); before, planes were
      drawn solid and traces over an aperture looked as if they ran over metal.

36. **Vias through a plane opening** (2026-10-15, `mesh.py` `Mesh.via_open_crossings`, `solver.py`
    `via_impedance(open_seg=)`). A via that passes a plane level without connecting, at a point inside one of that
    plane's apertures, used to be split per stack section like a via through solid copper: its piece between the
    two plates got the closed-section TEM radial-wave impedance, whose real part is power lost to an infinite
    plate pair -- although the via sits in the opening. With gap ports whose stitched pads (and pad vias) lie over
    an L2 opening this gave 20-45 % missing power and gain at the top of the band on a uniform line. Now such a
    segment is one piece over its whole span (quasi-static partial inductance, image in the plane below), coupled
    only to other such segments; the opening's magnetic currents carry the coupling to the plates around it.
    Ablation on the uniform-line check (L2 plane 75 um under a GCPW sheet, 1.3 mm opening along the line and under
    the port pads, 25-33 GHz, with decision 37): without this fix 18-36 % missing, |S| > 1 at 33 GHz, 3rd-length
    prediction |dS| 0.06-0.40; with it 3-10 % missing, S11 = S22 within 0.003, |dS| 0.007-0.06 (same line without
    L2: 0.05-0.09; with the pad vias over solid L2: 0.05-0.06). `leakage.cavity_leakage` should skip these
    segments (not done: their TEM power is not physical).
37. **Triangle area bound never applied** (2026-10-15, `trimesh._fixed`). The interior pass passed its area bound
    as `a{amax:.6e}` ("a1.125000e-08"); Triangle reads only digits and '.', so the bound was 1.125 m^2 and the
    rest of the string became other switches. Every `kind="tri"` mesh in SI units had unrefined interiors (only the
    edge band and min_angle shaped them): copper pours and plane openings got triangles of 1-3 mm^2 at
    max_cell 0.15-0.2 mm. Copper results were mostly fine (pour currents are smooth; edges were refined); aperture
    magnetic currents under a 0.44 mm line were not resolved at all. Now `_fixed()` writes plain decimals and
    max_cell holds. Models get larger at the same max_cell (a P1 line section 3 647 -> 5 317 unknowns at 0.2 mm):
    choose max_cell for the pours deliberately. Full test suite (114) passes.
    - Open: an opening over a whole model (all copper + 1 mm) does NOT reproduce the same model without the plane
      (gap-port line 1.5 mm: S21 phase -26 deg vs -100 deg, |dS| up to 1.08), while a 1.3 mm opening along the
      line is consistent (eps_eff 2.52 vs 2.52 without the plane). Large openings close (75 um) to dense sheet
      metal are not yet trustworthy; check the aperture size range before using one (CUT_DZ / DROP, the
      M-M closed-section kernel, mesh grading are the suspects).

## Validation baseline (keep these numbers when changing things)

- εr=1 slab vs images: < 1e-6 (relative to 1/4πR), 1–30 GHz.
- 50 Ω line on RO4003 (εr 3.55, h 0.508 mm, W 1.137 mm): εeff within 0.1–0.2 % of
  Kirschning–Jansen 2–10 GHz; |S11| < 1e-3; |S21| = 1 ± 0.003; phase < 0.1°;
  MoM Z0 ≈ 50.4 Ω at 1 GHz (mesh 0.5 mm, edge 0.08 mm).
- Open-end ΔL 0.22 mm vs Hammerstad-Bekkadal 0.21 mm.
- 50→25 Ω step: S11 within 0.1 % of quasi-static; unitarity/reciprocity < 4e-3 to 8 GHz.
- Notch filter vs openEMS: 3.667 vs 3.671 GHz, |S21| within 0.2 dB (1–7 GHz).
  (A naive "stub from line edge + ΔL" estimate is ~2 % off — that's the
  T-junction, not a solver bug.)
- Timing (original box): ~1.5 s/freq at N=757, ~17 s/freq at N=3173.
- Shunt via (0.3 mm drill, 50 ohm line, RO4350B 0.254) vs openEMS: |S21|
  0.0746 vs 0.0777 @10.25 GHz, 0.0247 vs 0.0293 @3.75 GHz; S11 phase offset
  1.5-3.5 deg growing with f (~70 um ref-plane snapping in openEMS).
- GCPW pour (gap 0.2, 1.2 mm wide, 1 mm fence) vs openEMS: S21 phase within
  0.5-3 deg over 10 mm, 1-15 GHz; MoM Z0 47.8 ohm, eeff 2.74.
- Multilayer: air stack vs images < 1e-6; stripline (W 0.4, b 0.508, er 3.66)
  eeff 3.663, Z0 40.5 vs Cohn 40.76; top (W 1.112) -> via -> inner (W 0.45),
  2 x 0.254 mm: S21 phase within 0.4-0.9 deg of openEMS 1-15 GHz, |S11|
  < 0.06 both; MoM |S21| 0.996 vs openEMS 0.988 @15 GHz (via does not
  radiate in MoM); reciprocity |S12-S21| up to 0.013 @15 GHz (different-width
  ports, like the 50->25 ohm step). N=3789, ~20 s/freq.

- Inner plane (3 x 0.254 mm, eps 3.66, L2 at 0.508, antipad 0.7): top W 0.56
  -> via -> stripline W 0.24 vs openEMS, 1.5-15 GHz: S21 phase within
  0.1-0.8 deg, |S21| within 0.025; stripline eeff 3.664. Lost power without
  stitching MoM 0.7 -> 7.9 %, openEMS 1.2 -> 10.7 % (MoM ~25 % low); with 4
  stitching vias < 0.6 % / < 1.2 %. |S11| MoM high above 9 GHz (0.30 vs
  0.20 @15 GHz). Suspect: antipad aperture (magnetic ring current drives
  the parallel plate directly; also the barrel-to-plane C) -- next fix.

- Tri mesh: through line eeff within 0.1 % of KJ (grid 0.05 %), Z0 51.56 vs
  grid 51.28 ohm @10 GHz; notch vs openEMS as grid; shunt via vs openEMS as
  grid.
- FreqInterp at N ~ 9000 (141 vias, 57 points at 0.25 GHz): 10 anchors,
  35 min instead of ~100; per point 6-7 s interpolation + 15-17 s LU;
  max|dS| 1.1e-4 against exact points.

## openEMS cross-checks

openEMS (v0.37) with Python bindings was used as the independent reference.
To regenerate the reference: take `openEMS/python/Tutorials/MSL_NotchFilter.py`,
set a writable `Sim_Path`, strip the matplotlib plotting, and save
`np.c_[f, abs(s11), abs(s21)]`. Runs in ~30 s. Use the same approach for new
reference cases (e.g. coupled lines) before trusting new features.

## ODB++ import (Engine/mom25d/odb.py, Engine/tools/odb_import.py)

- Parsed layers are pickled to `<odb>/../cache/mom25d/<step>_<layer>_v<CACHE_VERSION>.pkl`;
  **bump `CACHE_VERSION` whenever parsing changes** (the mtime check alone
  keeps stale caches).
- `parse_features` returns one entry per feature record (None for text etc.)
  so indices match the EDA `FID C <layer> <feature>` records used for net
  selection. `LYR` order in `eda/data` defines the layer index.
- Altium surfaces list hole (H) contours NOT directly after their island:
  in a board-wide GND pour (dozens of islands, thousands of holes) a trace
  clearance came thousands of records after its island. Assigning holes
  sequentially (or `make_valid` on the huge polygon) filled the clearances
  -> GND looked solid under traces. Fix: assign each hole to the smallest
  containing island.
- Tensor grid + staircase: a 0.37 mm 45-degree trace at 0.5 mm cells became
  corner-touching squares (no RWG edge -> open circuit). Default cell for
  non-Manhattan copper is w_min/2.5; `port_connectivity` refuses disconnected
  meshes. Fine cells propagate over the whole window (tensor grid). Use
  `--mesh tri` for non-Manhattan copper (decision 19).
- Stackup (`ODB.stack`, `ODB.gap`, `build_stackup`; spec: Engine/docs/odb_spec_user.pdf,
  "ODB++ System Attributes"): standard exports have DIELECTRIC layers with
  `.layer_dielectric` (thickness in the attrlist's UNITS=INCH/MM),
  `.dielectric_constant`, `.loss_tangent`; masks the same (eps_r = 1 is a
  placeholder -> --mask-er). Altium (vendor) exports have no DIELECTRIC
  layers: `.layer_dielectric` on a copper layer = the dielectric BELOW it
  (verified on a 12-layer export: sum without the bottom layer's value =
  board thickness to 0.1 %; the bottom layer's value repeats the bottom
  mask), no eps_r -> --er required. `.copper_weight` oz x 34.8 um. Copper
  between modelled levels is added to the dielectric above it. Never run on
  default h / eps_r: a wrong h moved a line's Z0 by 20 %.
- The ODB++ sample `Models/odb_rigidflex_sample/geometry/designodb_rigidflex.tgz` (standard stack, rigid-
  flex) drives `test_sample_design_stack` (skipped when absent). Solder mask
  vs openEMS: `Engine/tests/data/ref_mask_openems.py` (mask / bare, ~10 min each).
- `Solver.sweep` (verbose) prints `[k/n] elapsed, ETC, done at HH:MM` after
  each point; the user asked for ETC on long runs -- start long sweeps with
  nohup/`PYTHONUNBUFFERED=1` in the background and report the ETC line.
- Ports must be where the window cuts a trace (not on component pads); a
  trace crossing the window edge obliquely gives a too-wide port.

## Scaling guidance: large boards, MLFMM (discussed 2026-10-04)

- No MLFMM. Large models use H-matrix/ACA (decision 24): kernel-independent,
  so it works on the exact tabulated layered kernels; MLFMM would need
  complex-image / free-space-core approximations and breaks down at low
  frequency (our 1-3 GHz range). Measured: 120 mm GCPW (2-6 lambda_g),
  N 25397 -> 19 % / 9.4 % of dense storage at 3 / 9 GHz (compresses
  better at higher f; cost there is GMRES its, not rank).
- Rule of thumb (literature / scaling, NOT measured here): ACA stays near
  N log N to ~10 lambda_g; 10-30 lambda_g measure rank growth, next step a
  nested H^2 matrix; MLFMM / directional H^2 pays off beyond ~30 lambda_g
  and N > 1e5-1e6. Offered (not done): sweep one layout 2-20 lambda_g and
  plot max block rank / storage to find mom25d's own crossover.
- Whole board, e.g. multilayer 30 x 50 cm at 31 GHz (~60 x 100 lambda_g,
  1e7-1e8 unknowns with patterned planes): do NOT solve it whole.
  Cut-and-stitch: (1) full-wave windows of 5-30 mm (1-6 lambda_g, N ~
  5-30k) around transitions, launches, breakouts, filters/couplers,
  neighbouring channels (odb_import --window, tri mesh, GCPW ports, real
  ODB stackup); (2) long uniform runs as TL models; (3) cascade with
  SweepResult.connect / Network; (4) board-level plane cavity / edge /
  PI effects with a cavity model, coarse openEMS or an SI/PI tool.
  Gaps at 31 GHz: quasi-static via barrel (k1 h ~ 0.3 per 0.254 mm layer
  OK, long through-vias not), solid planes only, no differential ports,
  no board edges (infinite planes), manual window stitching (could be
  automated: nets -> windows -> cascade). Validate one window per
  structure type vs openEMS with mesh convergence.

## Open items / next steps

**From 2026-10-14 (internal ports / aperture models, decision 35):**
- Root cause of the tan d = 0 internal-port error (kernel path for real-axis poles near the gap).
- Large plane openings (decision 37 open point) and `leakage.cavity_leakage` skipping vias through openings (36).
- A port for a line in a plane aperture (CPW over an infinite plane): e.g. a slot port (magnetic-current delta
  source across the aperture between line and plane), or allow a meshed sheet on the same level as a plane
  (today it needs a thin spacer layer).
- A coplanar-mode gap port (line against both side grounds at once, like the feed's CoplanarGround).
- `current_view` was very slow / unresponsive on a ~2 k-unknown aperture model in Chrome and its plane layers did
  not show; `fields.plane_currents` ignores aperture fields (image of the sheet currents only) -- for aperture
  models take the plane current from n x H including M.
- Thin-sheet effective width (edge expansion / Hammerstad dW) as an option, instead of hand-tuned edge offsets.
- `Solver(method="auto")` may pick the H-matrix method; `memory_estimate()` then returns None (callers that
  print it need `method="dense"`).

**Proposed 2026-10-05 (from the first DXF board model):**
- `Stackup(bottom="air")`: an open bottom section (mirror of the top one),
  so boards with microstrip on both outer layers need no dummy ground and
  the lower outer layer's ports get a valid line voltage.
- ~~Copper thickness~~ done 2026-10-05 (decision 30, two sheets + walls).
  Possible next: N sheets, a cheaper effective-width option (Hammerstad dW)
  for large boards.
- Plane currents: done (`fields.plane_currents`, decision 31) from the via
  segment currents; the antipad gap voltages only change the near field
  (Ampere: the TEM amplitude beyond ~d is set by the enclosed via current).
- ~~Generic 3D geometry / current viewers~~ done 2026-10-05 (`view3d.py`,
  decision 31; also a cut-plane E-field viewer).

**Roadmap (suggested 2026-10-04, in this order):**
1. Version control: `git init`, `.gitignore` for `.venv/`, `*.egg-info`,
   `__pycache__`, `.pytest_cache`, the unpacked ODB++ sample + caches,
   proprietary `Models/<board>/` folders, numba cache files (`*.nbi`,
   `*.nbc`), then a first commit. The code grew ~3x on 2026-10-03 and
   was reorganised into Engine/ + Models/ on 2026-10-04 without any history.
   Open decisions (ask the user): commit `results/` or not (generated; two
   7.8 MB HTML pages); local only or a private remote. Do not keep `.git`
   on the Drive mount (no symlinks, slow stat, Drive sync conflicts in
   .git/): repo on the Linux side (e.g. replace the stale ~/mom25d) with
   Drive as a copy, or `git --git-dir` on the Linux side.
2. ~~Faster fill with numba~~ done 2026-10-04 (decision 26: 7-9x, same
   Z); antipad blocks 8.6x (decision 22). LU now dominates dense runs
   above N ~ 5000.
3. Differential / coupled-line (multimode) ports: one port per pair with
   differential and common mode; today each line is a single-ended port.
4. Low-frequency stabilisation (loop-star basis or Calderon-type
   preconditioner): Z cond ~1e5 at 1-3 GHz -> GMRES 150-330 its, and
   small Z errors amplified in S (decision 21 precision trap).
5. Multi-pin S-parameter parts from ODB++ (port-to-pin map in the JSON
   table, e.g. {"U1": {"file": "amp.s2p", "ports": [["1","3"],["2","3"]]}}),
   bottom-side parts.
6. Smaller: oblique ports in `find_ports`; read `matrix/stackup.xml`;
   solder mask following the trace contours / opened over pads; cos/sin
   phi antipad modes (only for antipads >> pad).

- Inner planes (2026-10-03): solid infinite planes done, validated vs
  openEMS (see baseline). Antipads: experimental m = 0 model (decision 22),
  confirmed against the refined openEMS reference (small / medium holes
  within ~16 %, large holes overshoot -> cos/sin phi modes); patterned planes and
  voids, finite board cavity, ODB `--planes` option (plane layer -> Stackup
  plane, antipads from its holes, refuse windows with voids under traces).

- ODB GCPW imports (2026-10-03): with `--mesh tri` a board window with a
  1 mm pour band was N ~ 9000 (grid: 22k), ~100 s/freq, GCPW feeds measured
  at the window edge. Wider bands / windows still need ACA or an iterative
  solver. The feed ground width is capped by the band (band - gap).
- ODB stackup import done (2026-10-03). matrix/stackup.xml (optional in
  the spec, full material data) is not read yet.
- ODB multilayer: patterned (non-solid) planes inside the stack can only be
  modelled as a meshed metal level (huge); solid ones are `planes=`.

- Full-wave vias (layered G_A^zz, G_A^zx, formulation-C correction) instead of
  the quasi-static vertical part; matters once k1 h is no longer small.
- Tri mesh with anisotropic edge band and oblique ports done (2026-10-03).
  Next: `find_ports` could now place ports along oblique traces at the
  window edge (today such a crossing gives a too-wide axis-aligned port);
  the finer tri mesh keeps a few 1-2 deg triangles at micron CAD steps
  (snap = 0.05 edge_cell is then < 2 um).
- Coupled-line / multimode ports. Lumped components: done (decision 23);
  next: multi-pin parts from ODB (pin-to-ground ports), bottom-side parts.
- Compressed / iterative solver done (decision 24). Next: a low-frequency-stable
  preconditioner (loop-star / Calderon) to cut GMRES iterations at 1-3 GHz,
  FreqInterp for compressed blocks.
- Frequency interpolation of Z: done (decision 21).
