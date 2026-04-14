"""
BESThep example: thermalization via 2<->2 elastic scattering (massive).
Run: mpirun -np 272 python run_2to2_massive.py
"""
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
n_grid   = 68*4
mass     = 1.0
coupling = 1.0
neval    = int(1e6)
dt       = 1e5
n_steps  = 3000
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
