# BEST — Boltzmann Equation Solver for Thermalization

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
besthep.py          # Main solver
examples/
  2to2m1.py         # 2→2 massive thermalization
  2to3m1.py         # 2→3 cannibal process
scripts/
  plot.py           # Plot evolution from checkpoint
  compare_rates.py  # Vegas vs analytical benchmark
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
n_grid   = 128
mass     = 1.0
coupling = 1.0
neval    = int(1e6)
dt       = 1e2
n_steps  = 1000
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
mpirun -np 8 python3 main.py
```
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
    month = "3",
    year = "2026"
}
```

LaTeX:

```tex
%\cite{Yoon:2026rce}
\bibitem{Yoon:2026rce}
J.~H.~Yoon,
%``Boltzmann Equation Solver for Thermalization,''
[arXiv:2603.28848 [hep-ph]].
```

## License

MIT

