# BEST — Boltzmann Equation Solver for Thermalization

[Comput. Phys. Commun. 327 (2026) 110295](https://doi.org/10.1016/j.cpc.2026.110295) · [arXiv:2603.28848](https://arxiv.org/abs/2603.28848)

[Talk (Summer Institute 2026)](https://best-hep.github.io/talks/20260812_SI2026_YOON.pdf)

A Python framework for solving the momentum-resolved Boltzmann equation for arbitrary *n* → *m* scattering processes using adaptive Monte Carlo integration.

## Overview

BEST evaluates the collision integral directly in 3(*n*_total − 2) dimensions (one azimuth integrated out analytically since v1.2.6) using the [Vegas](https://vegas.readthedocs.io/) adaptive Monte Carlo algorithm. It is designed for cosmological applications where the standard number-density (integrated) Boltzmann equation is insufficient and the full phase-space distribution must be tracked.

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

No installation of BEST itself is needed: clone the repository and set up the
dependencies. The route used for development is a conda environment with
`mpi4py` from conda (it brings its own MPI library) and `vegas` from pip
(it installs `gvar` with it):

```
conda create -n best python=3.13 numpy mpi4py
conda activate best
pip install scipy vegas
```

Check MPI before anything larger — this should print 0 and 1:

```
mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

On a cluster, build `mpi4py` against the system MPI instead: load the MPI
module, then `pip install mpi4py`; `python -c "from mpi4py import MPI;
print(MPI.Get_library_version())"` should name that MPI. `vegas` comes from
pip as above.

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
CHANGELOG.md
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
lam = 1.0   # L = -(lam/4!) phi^4

def matrix_element_squared(momenta):
    """Bare |M|^2 = lam^2; the symmetry factor for identical particles is applied by the solver."""
    return np.full(momenta.shape[2], lam**2)


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
        matrix_elements_squared={'2to2': matrix_element_squared})
else:
    solver.initialize_species('phi', init_f, stat='boson', mass=mass)
    solver.add_process('2to2',
                       ['phi', 'phi'], ['phi', 'phi'],
                       matrix_element_squared, neval=neval)

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

`matrix_element_squared(momenta)` returns the bare |M|² of the Feynman rules,
couplings included, one value per Vegas batch point (`momenta` has shape
(n_particles, 3, N); see `examples/propagator.py` for momentum-dependent
amplitudes). |M|² is the squared amplitude with the couplings of the
Lagrangian; the symmetry factors for identical particles belong to the phase
space, as usual. The solver's convention for internal states is the
unaveraged sum over every leg, initial and final: a spin-averaged
$\overline{|M|^2}$ of the textbooks is converted by multiplying with the
initial-state degrees of freedom, $\sum|M|^2 = g_1 g_2\,\overline{|M|^2}$.
The solver divides by the degrees of freedom of the observed species
(`initialize_species(..., dof=...)`, default 2 for fermions and 1 for bosons)
and applies the symmetry factor 1/(∏ n_in,s! ∏ n_out,s!) and the leg
multiplicities itself (`add_process(..., symmetry_factor='auto')`, the
default); pass `symmetry_factor=1.0` if your |M|² already contains the factor.

`initialize_species` also accepts a `(q, f)` pair of arrays or a two-column
text file (`q f`, comoving q at a₀). A species initialized at zero is seeded
from its production spectrum at the first step (`solver.f_seed`, default
1e-10), so freeze-in runs can start from an empty species.

### 2→3 number-changing process

```python
solver.add_process('cannibal',
    ['phi', 'phi'], ['phi', 'phi', 'phi'],
    matrix_element_squared, neval=int(1e7), delta_width=0.01)
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
    matrix_element_ann, neval=int(1e6))
```

### Time-dependent masses

```python
solver.set_mass_func('phi', lambda t: 1.0 if t > 20 else 0.0)
```

### Checkpointing

```python
solver.save_checkpoint('checkpoint.pkl', history=history)
history = solver.load_checkpoint('checkpoint.pkl',
    matrix_elements_squared={'2to2': matrix_element_squared})
```

`save_checkpoint` is **collective**: call it from all MPI ranks (as in the
examples above). The checkpoint stores the adapted Vegas integrator state of
every MPI group, so resumed runs continue seamlessly; resuming with a
different number of momentum groups discards the integrator maps with a
warning and re-adapts. The keys of `matrix_elements_squared` are process
names (or function names); lambdas and closures are restored by process name.

## Plotting the results

```bash
python scripts/plot.py checkpoint.pkl
```

writes one figure per species: the f(q) snapshot fan plus the N/N₀, E/E₀
conservation history. The freeze-out example has its own scripts,
`plot_spectra_stfo.py` and `plot_yield_stfo.py` (run next to `checkpoint.pkl`,
no arguments). The checkpoint is a plain pickle — `state['history']` holds
per-step f and moments for custom analysis.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

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