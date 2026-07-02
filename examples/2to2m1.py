"""
BESThep example: thermalization via 2<->2 elastic scattering (massive).
Run: mpirun -np 4 python examples/2to2m1.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# --- commonly adjusted ---
q_min    = 0.1      # momentum grid lower bound
q_max    = 50.0     # momentum grid upper bound
n_grid   = 40       # number of momentum grid points
mass     = 1.0      # phi mass
coupling = 1.0      # |M|^2 = coupling^2
neval    = int(1e5) # Vegas evaluations; raise until rel_err stays below max_rel_err
dt       = 1e2      # base time step
n_steps  = 20       # number of evolution steps

# --- advanced (defaults usually fine) ---
adapt_width    = True  # auto-adapt width from rel_err; False fixes it at delta_width (debugging/scans)
delta_width    = 0.1   # Gaussian energy-conservation width (initial value if adapt_width, else fixed)
max_rel_err    = 0.1   # widen width above this rel_err (integral too noisy; raise neval)
min_rel_err    = 0.01  # narrow width below this; raise if evolution stalls, lower as neval grows
max_rel_change = 0.3   # max relative change of f per step (adaptive dt control)

checkpoint_file = "checkpoint.pkl"  # saved state (delete to start fresh)

# ======================================================================
# Setup
# ======================================================================
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid, max_rel_change=max_rel_change, adapt_width=adapt_width, max_rel_err=max_rel_err, min_rel_err=min_rel_err)

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
                       neval=neval, delta_width=delta_width)

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
