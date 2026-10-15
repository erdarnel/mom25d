# mom25d — modelling assumptions

What every mom25d result assumes, always, even when nothing goes wrong. Read it before comparing with measurement or
another solver: a difference may be one of these, not an error. Things that fail or are not trustworthy are in
[LIMITATIONS.md](LIMITATIONS.md); the evidence behind both is in the numbered decisions of [CLAUDE.md](CLAUDE.md) (Dn).

## Geometry and stack

- **Laterally infinite layers.** Every dielectric layer, the ground plane and every inner plane extend to infinity.
  There are no board edges, no side walls and no enclosure: power that leaves the modelled copper (surface waves,
  parallel-plate waves between planes, radiation) never comes back. Board-edge and cavity resonances do not exist (D18).
- **PEC ground at the bottom** of the stack (z = 0). Inner planes are PEC sheets of zero thickness (a foil is one
  plane at its top face; D33); the top is air (open) or a PEC cover.
- **Copper is a zero-thickness sheet** unless `add_thickness` is used (two sheets + side walls, D30). With a sheet,
  the thickness enters only through the surface impedance; a thin-sheet edge offset (e.g. `EDGE_DELTA`) may be used
  to give the sheet line the thick line's Z0 — that is a fit, not physics.
- **Planar layout**: copper lies on discrete metal levels; vertical current flows only in vias and thick-metal walls.
- **Materials**: homogeneous, isotropic layers with a frequency-independent eps_r and tan d (no Djordjevic-Sarkar
  dispersion unless the user changes eps per frequency); copper conductivity sigma, smooth (no roughness model).
- Plane openings are closed with PEC and replaced by magnetic currents on their own mesh (equivalence principle, D33).

## Fields and currents

- **Time convention** e^{jwt}; SI units (metres, Hz).
- **Mixed-potential integral equation** for horizontal surface currents with exact layered Green's functions
  (Sommerfeld integrals, surface-wave poles handled exactly, no complex-image approximation).
- **Conductor loss**: skin-effect surface impedance Z_s = k_c coth(k_c t) / sigma on the sheet; ground-plane loss
  from the image return current (valid for k1 h << 1, D13).
- **Dielectric loss**: complex eps_r = eps_r (1 - j tan d).
- **Vias are quasi-static** in open sections (partial inductance of the barrel + image); in sections closed by two
  PEC planes they carry the parallel-plate TEM + TM_n modes, whose TEM real part is power lost to the infinite plate
  pair (D15, D18). No mutual inductance between vertical and horizontal currents (PEEC, D15). The barrel current is a
  filament of the drill radius; a via attaches to the copper inside its hole.
- **Vias through a plane opening** (not connected): one quasi-static piece over their span, image in the plane below
  (D36).
- **Antipads** (when given) are modelled with a ring-expanded frill field; otherwise a via passing a plane without
  `connect` and outside any opening is taken as passing a negligibly small hole (D22).

## Ports and networks

- **Feed ports**: a delta-gap source on a semi-infinite-like feed strip; S is de-embedded to the reference plane by
  fitting forward/backward waves to the sampled current (and voltage) along the feed. Assumes a single quasi-TEM mode
  on the feed and that the fit window is outside the near field of any discontinuity (D4, D7, D28).
- **Line voltage** = -∫E_z dz from the plane below the strip; Z0 from it (or from reciprocity, D5, D28).
- **Gap (internal) ports and lumped components**: a delta gap across a short bridge; the port voltage is the gap
  voltage, the reference impedance z_ref (50 ohm). The bridge and its slot are part of the port: S includes their
  parasitics until de-embedded (D35).
- **S-parameter blocks** (Touchstone) are connected as ideal networks at gap ports: their reference planes are the
  gap ports, with no extra coupling to the layout.
- **Reciprocal, linear, passive** structure (except connected active S-parameter blocks).

## Numerics

- **RWG basis on triangles**, Galerkin testing; singular integrals analytic, smooth parts by Dunavant rules (D1).
- **Mesh**: edge band of `edge_cell` along every copper edge, interior refined to `min_angle` and bounded by
  `max_cell` (bounded since D37, 2026-10-15; earlier tri meshes had unbounded interiors).
- **H-matrix** (when chosen): ACA far blocks to a 1e-4 tolerance, GMRES to 1e-6 (accepts small misses with a
  warning); results equal dense to that tolerance (D24).
- **Frequency interpolation** (FreqInterp) of the impedance matrix between anchors when enabled (D21).
