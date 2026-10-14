# Models/ — notes for running and post-processing

Each model: `geometry/<model>.py` (layout(), substrate() / stackup()),
`analysis/*.py` (run from anywhere; they add `../geometry` and `Engine/`
to sys.path), `results/` (outputs). New model = copy that pattern.

## Running on this machine (WSL, repo on a drvfs mount)

- A venv cannot be created inside the Drive mount (`Operation not permitted:
  'lib' -> '.venv/lib64'`, no symlinks). The venv in use is `~/venvs/mom25d`
  (see the top CLAUDE.md: deps + a .pth to Engine/). Original recipe:
  ```bash
  python3 -m venv --without-pip ~/venvs/mom25d        # python3.14 here has no ensurepip
  curl -s https://bootstrap.pypa.io/get-pip.py | ~/venvs/mom25d/bin/python
  ~/venvs/mom25d/bin/pip install numpy scipy matplotlib shapely pytest
  cd "$MOM25D_PATH"
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg ~/venvs/mom25d/bin/python Models/<model>/analysis/<script>.py
  ```
- Nothing is cached between runs: every script re-fills and re-solves every
  frequency (~20 s per point at N = 3173). Only sweep the frequencies you
  plot; `stepped_impedance_lpf/analysis/sweep.py` (25 points) takes ~8.5 min.

## Current data layout (easy to get wrong)

- `Solver.current_density(x)` returns one value per **triangle**
  (`(Nt, 2, Nexc)`, at centroids; Nt = 2226 for the LPF), not per RWG edge
  (`mesh.edge_mid`, N = 3173). Mixing the two index sets caused an
  out-of-bounds bug in an earlier 3D plot script.
- Inside a triangle the RWG current is linear:
  `K(r) = sum_i cA_i x_i (r - v_i)`, `cA = tri_sign * edge_len / (2 area)`;
  `ground_plane_current.py` evaluates it at the 7 Dunavant points.

## Ground-plane return current (`ground_plane_current.py`)

The solver never discretises the ground (it is inside the Green's function),
but its current follows directly from the trace current. With mu = mu0
everywhere and a PEC ground, the tangential H at z = 0+ is twice the
Biot-Savart field of K (image), and J_gnd = z x H:

    J_gnd(r) = -(h / 2 pi) Int K(r') / (|r - r'|^2 + h^2)^(3/2) dA'

- Quasi-static: valid for k1 h << 1 (0.18 at 9 GHz for h = 0.508 mm,
  eps_r 3.55). eps_r does not enter; the kernel does not depend on frequency.
  Frequency only enters through K.
- The ground current is the trace current smeared by a kernel of width ~h and
  of opposite sign. It is **not** a sharp mirror image: under a narrow line
  (W = 0.29 mm < h) it spreads to +-1.5 mm, and the tail falls off only as
  1/y^2. A window of +-Y keeps a fraction (2/pi) atan(Y/h) of the current
  (91.9 % for Y = 4 mm). This matches the printed check: the slab-averaged
  I_ground/I_trace is -0.90 at 1.5 GHz.
- An earlier explanation claimed fringing shrinks with frequency and that
  reflections "increase the return current". Both are wrong: locally the
  ground always carries minus the trace current, spread over ~h.

## Visualisation notes

- matplotlib smooth shading (area-weighted nodal average + `tripcolor(...,
  shading="gouraud")`, as in `current_density.py`) renders quickly. The
  timeouts seen earlier came from re-running full sweeps, not from plotting.
- three.js / Babylon.js pages rendered black in Firefox under WSL (no WebGL).
  Plotly `Mesh3d`/`Surface` HTML worked when served via
  `python3 -m http.server` (file:// URLs to `/tmp` paths failed).
- Plotly 3D: `aspectmode="data"` gives true scale, which makes h = 0.5 mm
  almost invisible next to a 28 mm filter. Exaggerate z explicitly and label it.

## 3D HTML view (`lpf_3d_view.py` -> `lpf_3d_view.html`)

- Plotly, true scale (`aspectmode="data"`, mm): mesh triangles with nodal
  |J| (dB rel. max, -40..0) on the trace, the P_h ground current on a 0.1 mm
  grid at z = 0, checkboxes (trace / ground currents) + frequency select via
  `write_html(post_script=...)` (`{plot_id}` is substituted by Plotly).
  3 frequencies, ~1.5 min; the HTML is ~7.6 MB (plotly.js embedded).
- **Checking what the user sees:** headless Firefox screenshots
  (`firefox --headless -no-remote -profile ~/ffshot/prof --screenshot
  ~/ffshot/x.png URL`) show HTML/DOM fine but render every Plotly 3D scene
  blank, even a trivial one (WebGL itself reports OK with
  `webgl.force-enabled`). There is no screenshot tool under WSLg. Ask the
  user for a screenshot: they save them to a Windows temp folder
  (reachable under `/mnt/c/...`).
- Firefox (snap-confined) cannot read or write the Claude scratchpad under
  `/tmp`: file:// URLs there fail and `--screenshot` into `/tmp` silently
  writes nothing. Serve pages with `python3 -m http.server` from the repo and
  write screenshots under `~`.
- "I don't see changes" was stale content: an old `http.server` on another
  port and old tabs still showed earlier (wrong) pages. Kill old servers, open
  with `firefox --new-window "URL?v=$(date +%s)"`.

## Mesh statistics (LPF, max_cell 0.6 mm, edge_cell 0.06 mm)

- N = 3173 RWG, 2226 triangles, 1280 nodes, dense Z 154 MB; cells
  0.06-0.59 mm, worst aspect 9.8 (edge lines run the whole length), largest
  cell ~ lambda_g/32 at 9 GHz.
- The two 16 mm feeds hold 1594 of the 3173 unknowns. The fit window needs
  only ~max(1.5 W, 4 h) margins, so ~8 mm feeds could roughly halve N
  (~3-5x faster). Not yet tried: verify S is unchanged before shortening.
- High-Z lines (W = 0.286 mm) get only 3 cells across (0.06/0.17/0.06 mm).

## Workflow pitfalls (hit in practice)

- **Which package gets imported:** `python some/dir/script.py` puts
  `some/dir` first on `sys.path`. A script in a directory that also holds a
  copy of `mom25d/` imports that copy, not the repo. Model scripts put
  `Engine/` first on sys.path; check `mom25d.__file__` when in doubt.
- **Waiting for a background run:** `while pgrep -f name.py; do sleep 5; done`
  never ends when launched via `bash -c "..."`, because pgrep matches the
  wrapper's own command line. Wait on the PID (`wait $pid`, or
  `while kill -0 $pid`) or tail the log instead.
- `pkill -f "<pattern>"` / `for p in $(pgrep -f "<pattern>"); do kill $p` run
  from a shell whose own command line contains the pattern kill that shell
  too (exit 144; hit twice). Find PIDs with `ps -eo pid,cmd | grep "[r]ef_plane"`
  (the bracket keeps grep and the shell from matching) and kill those -- but
  only if the same command line doesn't also contain the plain name (e.g. a
  restart of the same script in the same command): kill and restart in two
  separate commands.
- **Long runs:** start with `nohup` + `PYTHONUNBUFFERED=1` and a log file,
  then report the solver's `[k/n] elapsed, ETC, done at HH:MM` line when the
  user asks "etc". Without unbuffered output the log stays empty until exit.
- **Before optimising:** record `assemble` Z and sweep S from the unmodified
  package and compare after the change (bit-identical Z for pure refactors);
  the tests alone only check accuracy to ~1e-3.
- **Before claiming physics:** derive it, or compute it from solver output
  with a numerical check (e.g. the return-current fraction printed by
  `ground_plane_current.py`). Never present synthetic or formula-generated
  data as solver results, even as a placeholder.

## Multilayer 3D view (`transition_3d_view.py`)

- No plotly in the Python venv: the script only writes JSON; the page uses
  Plotly.js from a CDN with the JSON inlined (replace `__DATA__` in
  `transition_3d_template.html`).
- plotly.js 2.35.2 is NOT on cdnjs (404 -> "Plotly could not load"). Use
  `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js`;
  check a CDN URL with curl before publishing.
- Published as claude.ai artifact H9PfVwKBX5a84Y5tvEfuui (private).

## Lessons from the first DXF board model (2026-10-05, kept outside this tree)

- **Model folders outside `Models/`** (proprietary boards): the scripts take
  the engine from `MOM25D_PATH` (the mom25d folder) and put `../geometry`
  and `$MOM25D_PATH/Engine` on sys.path; never write the engine location
  into the board's files.
- **Open air on both outer layers**: flip the board in z (same S), keep the
  long / important circuit on the exact top side, dummy PEC ground ~2 mm
  under the other outer layer; its ports need `z0_from="reciprocity"` and
  the solver `v_weight=0` (CLAUDE.md decision 28). Check with a straight
  line on both sides (eps_eff, |S21|) before trusting the board.
- **Frequency interpolation does not run with the H-matrix solver** (auto
  above 7000 unknowns): every point is a full fill + GMRES. It also stays
  off unless the sweep has >= 2x as many points as anchors (anchor step
  ~0.35 c / (k_ratio D): ~1.4 GHz for a 45 mm layout -> 19 anchors for 26
  points = off; ~2.5 GHz at 25 mm -> 10 anchors = on).
- **Shrinking a board model** (N 12904 -> 7294, 75-110 -> ~20 s/point):
  clip ground pours to a band around the signal copper whose edge runs
  BETWEEN via rows (a band edge through via pads made N larger: 1.6 mm band
  12943 vs 2.0 mm 10929), keep the full via cage around vertical
  transitions, shorter feeds / narrower feed grounds, coarser edge cells.
  Compare against the full model: |dS| 0.016-0.055 (20-35 GHz) here, about
  the full model's own mesh sensitivity.
- **Short feeds**: 4.5 mm feeds (1 mm margins) left a ~2 mm fit window:
  fitted line loss noisy above ~29 GHz, and with v_weight=0 the window must
  stay >~ lambda_g / 2. A port 1.1 mm from a stub / resistor had couplings
  non-reciprocal by up to 0.28 at the band edges: move such reference
  planes further out (add line) and check |S_ij - S_ji|.
- **Cache solutions for viewers**: save the port-wave current vectors (+ S)
  to an npz; redrawing a 3D current page then takes seconds (`--replot`).
- **Copper outlines from shapely**: never take "the largest piece" of a
  union without checking for MultiPolygon seams (see decision 29).
