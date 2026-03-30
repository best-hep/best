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

No installation required. Clone the repository and ensure the following dependencies are available:

```
numpy
scipy
mpi4py
vegas
```

## Quick Start

### 2→2 elastic scattering

```python
import numpy as np
from besthep import BEST

def matrix_element(momenta, coupling):
    return np.full(momenta.shape[2], coupling**2)

def init_f(r):
    return 1.0 / (1 + np.exp((r - 3) / 2.0))

solver = BEST(q_min=0.1, q_max=20.0, n_grid=64)
solver.initialize_species('phi', init_f, stat='boson', mass=1.0)
solver.add_process('elastic',
    ['phi', 'phi'], ['phi', 'phi'],
    matrix_element, coupling=1.0, neval=int(1e6), delta_width=0.01)

for step in range(100):
    solver.evolve_step(dt=1.0, method='heun')
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
solver.initialize_species('phi', init_phi, stat='boson', mass=0.0)
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
@article{Yoon:2026best,
    author = "Yoon, Jong-Hyun",
    title = "{Boltzmann Equation Solver for Thermalization}",
    year = "2026",
    eprint = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph"
}
```

LaTeX:

```tex
%\cite{Yoon:2026best}
\bibitem{Yoon:2026best}
J.~H.~Yoon,
``Boltzmann Equation Solver for Thermalization,''
[arXiv:XXXX.XXXXX [hep-ph]].
```

## License

MIT

