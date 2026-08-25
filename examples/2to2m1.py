"""
BEST-hep example: thermalization via 2<->2 elastic scattering (massive).
Run: mpirun -np 8 python examples/2to2m1.py

Convention for `coupling` and `matrix_element`
----------------------------------------------
BEST knows nothing about Lagrangians. The solver sees |M|^2 only through the
user-supplied `matrix_element` function and assumes that |M|^2 ALREADY
INCLUDES all identical-particle symmetry factors (Kolb & Turner convention).
The leg multiplicities are applied by the code itself,
    C = n_alpha * C_alpha + n_beta * C_beta   (e.g. C = 2*C_2 + 3*C_3 for 2<->3).

`coupling` is just a number handed to `matrix_element`; its physical meaning is
fixed by how you write that function. In this example `matrix_element` returns
coupling**2 with no symmetry factor, so `coupling` is an effective parameter:
for L = -(lam/4!) phi^4 it corresponds to lam/2, and coupling = 1.0 (as used
for the figures in the paper) means lam = 2.

If you want `coupling` to be the Lagrangian coupling itself, put the symmetry
factor inside `matrix_element`: divide the Feynman-rule |M|^2 by k! for every
species appearing k times on a given side of the reaction, on BOTH sides:

    L = -(lam /4!) phi^4,  phi phi <-> phi phi      :  return coupling**2 / (2*2)   # 2!*2!
    L = -(lam5/5!) phi^5,  phi phi <-> phi phi phi  :  return coupling**2 / (2*6)   # 2!*3!

The factor is symmetric between the two sides, which is why one `matrix_element`
serves the gain and loss terms and both C_2 and C_3.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST


# ======================================================================
# Matrix element
# ======================================================================
def matrix_element(momenta, coupling):
    """Constant |M|^2 = coupling**2; symmetry factors absorbed into coupling (see module docstring)."""
    # If coupling is the Lagrangian coupling lam of L = -(lam/4!) phi^4, use instead:
    # return np.full(momenta.shape[2], coupling**2 / (2*2))   # 2!*2!
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
q_min    = 0.1      # momentum grid lower bound
q_max    = 50.0     # momentum grid upper bound
n_grid   = 40       # number of momentum grid points
mass     = 1.0      # phi mass
coupling = 1.0      # see docstring
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