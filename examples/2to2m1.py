"""
BEST-hep example: thermalization via 2<->2 elastic scattering (massive).

Run: mpirun -np 8 python examples/2to2m1.py

Matrix-element convention
-------------------------
`matrix_element_squared(momenta)` returns the bare |M|^2 of the Feynman rules,
couplings included, without symmetry factors: |M|^2 = lam^2 for
L = -(lam/4!) phi^4. The solver applies the symmetry factor for identical
particles, 1/(prod_s n_in,s! prod_s n_out,s!) (1/4 here), and the leg multiplicities itself:
add_process(..., symmetry_factor='auto') is the default. If your |M|^2 already
contains the factor, pass symmetry_factor=1.0 instead.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST


# ======================================================================
# Matrix element
# ======================================================================
lam = 1.0   # quartic coupling, L = -(lam/4!) phi^4

def matrix_element_squared(momenta):
    """Constant bare |M|^2 = lam^2, one value per batch point."""
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
q_min    = 0.1      # momentum grid lower bound
q_max    = 50.0     # momentum grid upper bound
n_grid   = 40       # number of momentum grid points
mass     = 1.0      # phi mass
neval    = int(1e6) # Vegas evaluations
dt       = 1e2      # base time step
n_steps  = 20       # number of evolution steps

checkpoint_file = "checkpoint.pkl"  # saved state (delete to start fresh)

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
