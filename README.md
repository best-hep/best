# BEST — Boltzmann Equation Solver for Thermalization

[Comput. Phys. Commun. 327 (2026) 110295](https://doi.org/10.1016/j.cpc.2026.110295) · [arXiv:2603.28848](https://arxiv.org/abs/2603.28848)

[Talk (Summer Institute 2026)](https://best-hep.github.io/talks/20260812_SI2026_YOON.pdf)

A Python framework for solving the momentum-resolved Boltzmann equation for arbitrary *n* → *m* scattering processes using adaptive Monte Carlo integration.

## Overview

BEST evaluates the collision integral directly in 3(*n*_total − 2) dimensions using the [Vegas](https://vegas.readthedocs.io/) adaptive Monte Carlo algorithm. It is designed for cosmological applications where the standard number-density (integrated) Boltzmann equation is insufficient and the full phase-space distribution must be tracked.

Key features:

- **Arbitrary *n* → *m* processes** — 2→2, 2→3, 3→2, and higher multiplicities, with integration dimensionality determined automatically
- **Identical-particle decomposition** — Correct treatment of processes with unequal multiplicities on each side (e.g. ϕϕ ↔ ϕϕϕ), essential for energy conservation
- **Full quantum statistics** — Bose enhancement and Pauli blocking without approximation
- **Massive particles** — Arbitrary masses, including time-dependent masses for phase transitions
- **Multiple coupled species** — Simultaneous evolution of several interacting species
- **Cosmological expansion** — Comoving momenta with built-in radiation domination
- **Exponential time integrator (`exprb`)** — Hubble-paced time steps for expansion runs to freeze-out, where explicit steppers become impractically expensive
- **Sequential exponential integrator (`exprb_seq`)** — Gauss–Seidel splitting over processes for runs where a stiff number-conserving process (elastic scattering) coexists with slow number-changing chemistry; a single summed `exprb` step damps the slow chemistry by the stiff rate, the sequential form does not
- **Semi-analytical 2→2 benchmark** — Exact energy conservation following [Ala-Mattinen et al. (2022)](https://arxiv.org/abs/2201.06456)
- **MPI parallelization** — Near-linear scaling to hundreds of cores

## Installation

```bash
git clone https://github.com/best-hep/best.git
cd best
```

No installation required. Clone the repository and ensure the following dependencies are available:

```
numpy
scipy
mpi4py
vegas
```
## Repository Structure
```
besthep.py            # Main solver
dof_Drees_etal.dat    # SM relativistic degrees of freedom table (Drees et al.)
examples/
  2to2m1.py           # 2→2 massive thermalization
  2to3m1.py           # 2→3 cannibal process
  propagator.py       # momentum-dependent matrix element (s-/t-channel)
  subthreshold_freezeout.py  # sub-threshold freeze-out to relic abundance
                             # (constant-dof protocol, ann + el, exprb_seq)
scripts/
  plot.py             # Plot evolution from checkpoint
  compare_rates.py    # Vegas vs analytical benchmark
  plot_spectra_stfo.py  # f(q) snapshots + BE overlay for the freeze-out run
                        # (fits T from its prescribed bath species)
  plot_yield_stfo.py    # Y = n/s vs x = m1/T, with H vs net-rate panel
requirements.txt
LICENSE
```

## Quick Start

### 2→2 elastic scattering

```python
import numpy as np
import os
from besthep import BEST


# ======================================================================
# Matrix element
# ======================================================================
def matrix_element(momenta, coupling):
    """Constant |M|^2. Symmetry factors included in coupling."""
    return np.full(momenta.shape[2], coupling**2)


# ======================================================================
# Initial condition
# ======================================================================
def init_f(r, r0=3.0, width=2.0):
    """Non-thermal sigmoid distribution."""
    return 1.0 / (1 + np.exp((r - r0) / width))


# ======================================================================
# Parameters
# ======================================================================
q_min    = 0.1
q_max    = 50.0
n_grid   = 40
mass     = 1.0
coupling = 1.0
neval    = int(1e6)
dt       = 1e2
n_steps  = 20
checkpoint_file = "checkpoint.pkl"


# ======================================================================
# Setup
# ======================================================================
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid)

resume = os.path.exists(checkpoint_file) and solver.world_rank == 0
resume = solver.world_comm.bcast(resume, root=0)

if resume:
    history = solver.load_checkpoint(
        checkpoint_file,
        matrix_elements={'matrix_element': matrix_element})
else:
    solver.initialize_species('phi', init_f, stat='boson', mass=mass)
    solver.add_process('2to2',
                       ['phi', 'phi'], ['phi', 'phi'],
                       matrix_element, coupling=coupling,
                       neval=neval)

    history = solver.init_history()


# ======================================================================
# Evolution
# ======================================================================
for step in range(n_steps):
    solver.evolve_step(dt=dt)

    m = solver.record(history)

    if solver.world_rank == 0:
        N0, E0 = history['phi']['n'][0], history['phi']['e'][0]
        print(f"  N/N0={m['phi']['n']/N0:.6f}  "
              f"E/E0={m['phi']['e']/E0:.6f}")

    solver.save_checkpoint(checkpoint_file, history=history)
```

Run with MPI:

```bash
mpirun -np 8 python3 examples/2to2m1.py
```
### Plotting the results

```bash
python scripts/plot.py checkpoint.pkl
```

writes one figure per species: the f(q) snapshot fan plus the N/N₀, E/E₀
conservation history. The freeze-out example has its own scripts,
`plot_spectra_stfo.py` and `plot_yield_stfo.py` (run next to `checkpoint.pkl`,
no arguments). The checkpoint is a plain pickle — `state['history']` holds
per-step f and moments for custom analysis.

### 2→3 number-changing process

```python
solver.add_process('cannibal',
    ['phi', 'phi'], ['phi', 'phi', 'phi'],
    matrix_element, coupling=1.0, neval=int(1e7), delta_width=0.01)
```

The identical-particle decomposition (*C* = 2*C*₂ + 3*C*₃) is handled automatically.

### Cosmological expansion

```python
solver.current_time = 100.0
solver.set_radiation_dominated(a0=1.0, t0=solver.current_time)
```

### Choosing a time integrator

- `heun` (default) and `euler` are explicit: adequate for short relaxation /
  thermalization problems.
- `exprb` (diagonal exponential Rosenbrock–Euler): **use this for expansion
  runs that track equilibrium over many Hubble times (e.g. freeze-out to a
  relic abundance).** There the collision rates exceed *H* by orders of
  magnitude, so explicit steppers need dt ~ 1/rate and become impractically
  expensive, while `exprb` runs Hubble-paced dt at one Vegas pass per step.
  First order, positivity-preserving; it does not remove elastic
  shape-relaxation stiffness.
- `exprb_seq` (sequential / Gauss–Seidel exponential splitting): **use this
  when a stiff number-conserving process (elastic scattering) runs alongside
  slow number-changing chemistry** (e.g. the sub-threshold freeze-out example,
  `ann` + `el`). A single summed `exprb` step damps the slow net rate by the
  stiff Γ, suppressing the chemistry by ~Γ_stiff·dt; `exprb_seq` instead
  advances each process over the full dt with its own exponential substep,
  stiffest first, re-measuring the later processes' rates on the updated f.
  Costs 2N−1 rate passes per step for N processes; identical to `exprb` for
  a single process. First-order splitting; run with `adapt_dt=False` in
  stiff regimes.

```python
solver.evolve_step(dt, method='exprb')
solver.evolve_step(dt, method='exprb_seq', adapt_dt=False)
```

### Multiple species

```python
solver.initialize_species('chi', init_chi, stat='fermion', mass=5.0)
solver.initialize_species('phi', init_phi, stat='boson', mass=1.0)
solver.add_process('annihilation',
    ['chi', 'chi'], ['phi', 'phi'],
    matrix_element_ann, coupling=0.1, neval=int(1e6))
```

### Time-dependent masses

```python
solver.set_mass_func('phi', lambda t: 1.0 if t > 20 else 0.0)
```

### Checkpointing

```python
solver.save_checkpoint('checkpoint.pkl', history=history)
history = solver.load_checkpoint('checkpoint.pkl',
    matrix_elements={'matrix_element': matrix_element})
```

`save_checkpoint` is **collective**: call it from all MPI ranks (as in the
examples above). The checkpoint stores the adapted Vegas integrator state of
every MPI group, so resumed runs continue seamlessly; resuming with a
different number of momentum groups discards the integrator maps with a
warning and re-adapts.

## Changelog

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
- Near-equilibrium backward rates: direct net-rate integrator with a
  significance-gated reconstruction (BW = FW + net), removing the
  near-cancellation noise of subtracting two gross rates.
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

## Citation

If you use BEST in your work, please cite:

BibTeX:

```bibtex
@article{Yoon:2026rce,
    author = "Yoon, Jong-Hyun",
    title = "{Boltzmann Equation Solver for Thermalization}",
    eprint = "2603.28848",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1016/j.cpc.2026.110295",
    journal = "Comput. Phys. Commun.",
    volume = "327",
    pages = "110295",
    year = "2026"
}
```

LaTeX:

```tex
%\cite{Yoon:2026rce}
\bibitem{Yoon:2026rce}
J.~H.~Yoon,
%``Boltzmann Equation Solver for Thermalization,''
Comput. Phys. Commun. \textbf{327}, 110295 (2026)
doi:10.1016/j.cpc.2026.110295
[arXiv:2603.28848 [hep-ph]].
```

## License

MIT
