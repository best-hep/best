# Changelog

### v1.3.1
- CP violation: the reverse direction of a process no longer has to carry the
  forward |M|^2. `add_process(..., mesq_backward=func)` gives
  the reverse |M|^2 directly (a process whose reverse is its own CP conjugate, or
  a subtracted remainder); `add_process(..., cp_conjugate='name')` links a process
  to its CP conjugate: the two are integrated on the same Vegas samples slot by
  slot, so that the difference of a particle's and its antiparticle's
  distributions is free of the noise they share, and without a reverse |M|^2 of
  their own each one's reverse is the other's forward |M|^2 (CPT). The backward
  fallback and the quality control of a grid point treat a pair as a whole, its
  iterations are combined with one weight per iteration (as for every process,
  see Fixed below), and in `exprb_seq` a pair is one substep.
  `initialize_species(..., antiparticle=)` declares antiparticles (a
  self-conjugate species names itself); the solver
  checks a pair slot by slot (mass, statistics, dof, grid, identical particles,
  declared antiparticles, same neval/nitn/alpha, species initialized before, a
  reverse |M|^2 for both members or neither), not the physics (CPT consistency,
  unitarity and the real-intermediate-state subtraction are the user's; see the
  README). Jacobian, ledgers and checkpoints work as for any process
  (`matrix_elements_squared={'name': (func, func_backward)}`; a checkpoint
  restores what was there and refuses a reverse |M|^2 for a process saved without
  one). `scripts/check_conjugate.py`.
- Custom collision terms: `add_custom_term(name, func, jacobian=None)` with
  `func(solver) -> {species: (gain, loss)}` on the grid. They take part in every
  rate pass, in the exponential steppers, in the seeding of empty species and
  in the per-process gain/loss ledgers like a process; checkpoints store them by
  function name (`load_checkpoint(..., custom_terms=...)`).
- `solver.step_hook(solver)` is called at the start of every step (medium
  quantities such as thermal or Debye masses computed once per step; mass
  functions and matrix elements may read the solver).
- `solver.compute_rates()` runs a rate pass without stepping (rates and the
  per-process gain/loss ledgers at the current state).
- New method `exprb2` (two-stage exponential Rosenbrock, ETD2RK; `exprb2_update`):
  the `exprb` predictor, a second rate pass at the predicted state and a φ₂
  correction. Second order; a stiff species follows its moving equilibrium with
  the physical lag 1/Γ, where `exprb` lags by one step, so quantities sourced by
  that departure (a CP asymmetry in strong wash-out) no longer need dt ≪ 1/Γ.
- `examples/Mdecay_cp.py`: the CP asymmetry of N ↔ Φ l as a result of the run,
  with the decay and its CP conjugate as a pair (each with its forward |M|^2 as
  the reverse one), l and lbar as species, the plasma coupling of the leptons as
  a custom term with its Jacobian and `exprb_full`, next to the integrated
  reference D1 of Hahn-Woernle, Plümacher & Wong 2009 (README, CP violation);
  `scripts/plot_cp.py` plots its output. `scripts/check_asymmetry.py`:
  the same asymmetry from the linear asymmetry equation (HPW's formulation) with
  the momentum-resolved N, leptons in kinetic equilibrium, `exprb2`, against an
  exact solution of the mode equations. `scripts/check_custom_term.py`: the
  decay written as one-dimensional kernels against the Vegas process (rates,
  Jacobian, evolution, seeding, checkpoint).
- Fixed: a second `BEST` instance in one MPI program took the communicator the
  first one had repointed `MPI.COMM_WORLD` at for Vegas as its world, so under
  `mpirun` every rank ran it alone (and printed); the world communicator is now
  taken once at import. Single-instance scripts are unchanged.
- Fixed: the rates of every process are taken from Vegas' per-iteration
  estimates combined with one weight per iteration (the inverse variance of the
  sum column), the combination the CP pairs use. Vegas' own weighted average of
  the joint columns [FW+BW, FW, BW] inverts a singular covariance through an SVD
  cut, which moved the loss and the gain off their common-sample estimate: in
  the Mdecay runs this was the drift of the conserved numbers and of the
  comoving energy and the step-to-step noise of the soft occupations. Elastic
  conservation is now at the quadrature level of the grid. Unpaired processes
  no longer reproduce 1.3.0 bit for bit, and the quality-control counts can
  differ from 1.3.0, since the σ they act on is now that of the scalar
  combination.
- Fixed: `load_checkpoint` left a species that the resuming script had
  initialized empty marked for seeding, so the first step after the restore
  overwrote its restored distribution with the seed spectrum; a restored
  species is now seeded only if it is still at the floor. `step_hook` is now
  called before the seeding of empty species, so the seed sees the medium of
  the step. An antiparticle declaration that contradicts an earlier one is
  refused before anything is recorded. A custom term's entry for a species
  outside `species_list` is skipped, as a process's is.
- `BEST.J_SIG` (default 1.0): in `exprb_full` an entry of the Monte Carlo
  Jacobian is used only if |J| ≥ J_SIG·σ, σ from the sum of squares of its
  sample contributions (the rate quality-control rule applied to J); the
  verbose line of the step reports the counts zeroed/kept. Unresolved entries
  of a row whose true value is ~0 (tail nodes) otherwise pair with the large
  exact couplings into that node and give the matrix exponential spurious
  eigenvalues, which blew up the self-consistent-bath runs. 0 disables.

### v1.3.0
- Exact kinematics for every process: the last two free legs are generated by
  two-body kinematics in the rest frame of their total four-momentum and
  boosted back (PDG, Kinematics review), so energy and momentum are conserved
  sample by sample. The on-shell projection, `delta_width`, `adapt_width` and
  the `max_rel_err`/`min_rel_err` thresholds are gone; the integration has one
  dimension fewer and a much smaller variance, and the default `neval` is now
  1e5. Checkpoints written by earlier versions are not compatible (different
  integrator maps).
- New method `exprb_full`: exponential Rosenbrock–Euler with the full Jacobian
  of the collision operator (assembled on the Vegas samples), for a bath that a
  slow process feeds or drains while stiff processes keep it thermal.
- `examples/Mdecay.py`: Majorana decay N ↔ Φ l with a self-consistent bath
  (decays, fermions, thermal masses, expansion); `scripts/plot_decay.py`.
- A note is printed when a massless species is initialized (its rates grow as
  1/E in the infrared; a thermal mass m ∝ 1/a caps the stiffness).
- The initialization log prints the number density n with the normalization of
  `compute_moments` (g/(2π²)); it used to print 4π∫f q² dq. Species named
  `times` or `a` are refused (keys of the history dict).

### v1.2.10
- 1↔2 processes (decays and inverse decays): energy conservation is solved
  exactly by the two-body kinematics, one log-sampled variable on the
  closed-form kinematic window; no on-shell projection, `delta_width` unused.

### v1.2.9
- Internal degrees of freedom: `matrix_element_squared` is summed over the
  internal states of every particle, initial and final; the solver divides by
  the observed species' dof (Vegas and analytical paths). Previously the
  solver multiplied by the dof of the other legs, which required an |M|^2
  averaged over all states. Species with dof = 1 are unchanged; user
  matrix elements for fermions or other multi-state species give different
  numbers than in v1.2.8. `initialize_species(..., dof=...)` documented.
- Integrand: samples with a massless leg projected to zero energy no longer
  produce inf (the phase-space factor is evaluated on valid samples only), and
  rate-density values below `rate_flush` = 1e-150 are set to zero so that
  Vegas never divides by an underflowed variance. Both affected only 1→2
  decays and massless legs at large q; other results are unchanged.

### v1.2.8
- Matrix elements: `matrix_element_squared(momenta)` returns the bare |M|^2 with
  the couplings inside; the `coupling` argument of `add_process` is gone. The
  symmetry factor for identical particles, 1/(prod n_in,s! prod n_out,s!), is applied by the
  solver (`symmetry_factor='auto'`, the default; a number overrides it). This
  is the standard convention of particle physics: |M|^2 is the squared Feynman
  amplitude with the couplings of the Lagrangian (lam^2 for
  L = -(lam/4!) phi^4), and the symmetry factors belong to the phase
  space, where the solver puts them. Not compatible with earlier
  matrix-element functions or checkpoints;
  `load_checkpoint(..., matrix_elements_squared={process name: func})`.
  The examples use lam = 1 (lam5 = 1); the paper's figures correspond to
  lam = 2 and lam5 = sqrt(12) in this convention.
- `initialize_species` accepts a tabulated spectrum, `(q, f)` arrays or a
  two-column text file; a species initialized at zero is seeded from its
  production spectrum at the first step (`f_seed`), so freeze-in runs start
  from an empty species.
- Rate quality control: a grid point is skipped only when both directions
  vanish; an empty species in the input slot previously lost its gain.
- Checkpoint: integrator maps restored but not yet re-wrapped were saved as
  their `map()` method (crash on the next resume when a backward fallback
  fired); fixed. A process that cannot be restored is an error, not a warning.
- `n_r_parallel` must divide the number of MPI ranks (was an IndexError later).

### v1.2.7
- History records each species' mass per snapshot (time-dependent masses).

### v1.2.6
- Sampling frame aligned with the observed momentum (target on the polar
  axis) and one azimuth integrated out by rotational symmetry: results
  identical, one fewer integration dimension.
- `adapt_width`: widening capped at `delta_width` = 0.05.

### v1.2.5
- Energy conservation by on-shell projection: each sample is projected onto
  the exact E_in = E_out shell (kinetic energies rescaled by (1 ∓ λ) on the
  initial/final side, λ = ΔE/K_tot, observed leg fixed) and f is read there;
  the Gaussian only weights how far off-shell the sample was. Equilibrium is
  therefore an exact fixed point (gain − loss vanishes sample by sample), and
  the width `delta_width` × K_tot keeps tracking the shell as T drops.

### v1.2.4
- Checkpoints store vegas AdaptiveMaps instead of full Integrators;
  fixes the MPI gather overflow at large neval.

### v1.2.3

- Backward (gain) rates: single joint Vegas evaluation of (FW+BW, FW, BW)
  on common sample points; the net is the correlated difference. Deep-tail
  gains become measurable. Supersedes the v1.2.1 reconstruction gate
  (removed, with `bw_recon_nsigma` and the per-direction width entry).
- BW fallback to its own adaptive map when the joint error is poor;
  unmeasurable modes are held or zeroed (subdominant only). Counts and
  min/max rel. error in the `verbose` step summary.
- `exprb_seq`: stiffness ordering now uses the f-weighted bulk loss rate
  per process (previously the per-mode maximum); mid-step interpolator
  rebuilds use a(t), the final one a(t+dt).
- ~~`2to2m1.py`: docstring documents the `coupling`/`matrix_element` convention
  (symmetry factors assumed included in |M|²).~~ *(superseded in v1.2.8: the
  solver applies the symmetry factor itself)*

### v1.2.2

- Tail extrapolation fits in the comoving energy (scale factor passed to the
  interpolator); removes a slope bias growing with expansion.
- High-side extrapolation slope: Theil–Sen fit, robust against disturbed
  boundary modes; separate `n_high` window.
- Vegas sampling domain extended past the grid top (`domain_extension`,
  default 1.5×); restores the down-scattering resupply of the top modes,
  which was truncated at `q_max` and bent the boundary band after freeze-out.
- `2to2m1.py`: accuracy defaults tightened (neval 10⁵ → 10⁶, fixed narrow
  energy-conservation width).

### v1.2.1

- New `exprb_seq` time integrator: sequential (Gauss–Seidel) exponential
  splitting over processes.
- ~~Near-equilibrium backward rates: direct net-rate integrator with a
  significance-gated reconstruction (BW = FW + net), removing the
  near-cancellation noise of subtracting two gross rates.~~
  *(superseded in v1.2.3)*
- Interior interpolation of log(1/f + η) switched to a monotone (PCHIP)
  cubic — no spline ringing across populated/empty boundaries.
- Representation floors extended and made consistent (f resolved down to
  1e-300 through the interpolator and the rate assembly).
- Checkpoint resume now restores each MPI group's adaptive integration
  widths correctly (previously they reverted to stale values on the next
  save after a resume).
- `evolve_step` rejects unknown `method` strings instead of silently
  skipping the update; overflow guard in the adaptive-dt controller for
  strongly driven Bose-enhanced modes.
- `solver.verbose = True` exposes estimator internals (BW estimator
  selection counts, rel_err statistics).
- Sub-threshold freeze-out example rewritten as the constant-dof protocol
  run (`ann` + elastic, `exprb_seq`), with matching spectra and yield plot
  scripts.
