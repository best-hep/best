"""
BESThep example: momentum-dependent matrix elements.

This example shows how to write a |M|^2 that depends on the particle
momenta, using s-channel and t-channel Breit-Wigner propagators as
illustrations. It is meant to document the `momenta` array interface,
NOT to model a specific physical process.

----------------------------------------------------------------------
The `momenta` argument passed to a matrix-element function
----------------------------------------------------------------------
A matrix-element function has the signature

    def matrix_element(momenta, coupling) -> array of shape (N,)

where `momenta` is a NumPy array of shape

    (n_total, 3, N)
       |       |   |
       |       |   +--  Vegas batch index (N sample points)
       |       +------  momentum component: 0 = px, 1 = py, 2 = pz
       +--------------  particle index

The particle index follows the order given to add_process:
input species first, then output species. For

    add_process('scatter', ['phi', 'phi'], ['phi', 'phi'], ...)

the mapping is

    momenta[0] -> incoming particle 1   (initial)
    momenta[1] -> incoming particle 2   (initial)
    momenta[2] -> outgoing particle 1   (final)
    momenta[3] -> outgoing particle 2   (final)

Each momenta[i] is the 3-momentum (px, py, pz) as arrays over the batch.
Only 3-momenta are provided; the energy is reconstructed on-shell as
E_i = sqrt(|p_i|^2 + m_i^2) using the known particle mass. The return
value must be an array of length N (the squared amplitude at each
sample point).

Run: mpirun -np 4 python propagator.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from besthep import BEST


# ======================================================================
# Model parameters (illustrative only)
# ======================================================================
M_PHI = 1.0     # mass of the external phi
M_MED = 5.0     # mass of the (virtual) mediator in the propagator
GAMMA = 1.0     # mediator width (regularizes the resonance)


# ======================================================================
# Matrix elements
# ======================================================================
def matrix_element_schannel(momenta, coupling):
    """s-channel Breit-Wigner: |M|^2 = g^2 / ((s - M^2)^2 + (Gamma M)^2).

    s = (p1 + p2)^2 is built from the two INITIAL-state momenta,
    momenta[0] and momenta[1].
    """
    p1 = momenta[0]                      # initial particle 1, shape (3, N)
    p2 = momenta[1]                      # initial particle 2

    E1 = np.sqrt(np.sum(p1**2, axis=0) + M_PHI**2)
    E2 = np.sqrt(np.sum(p2**2, axis=0) + M_PHI**2)

    # Mandelstam s = (E1 + E2)^2 - |p1 + p2|^2
    s = (E1 + E2)**2 - np.sum((p1 + p2)**2, axis=0)

    return coupling**2 / ((s - M_MED**2)**2 + (GAMMA * M_MED)**2)


def matrix_element_tchannel(momenta, coupling):
    """t-channel Breit-Wigner: |M|^2 = g^2 / ((t - M^2)^2 + (Gamma M)^2).

    t = (p1 - p3)^2 is built from one INITIAL (momenta[0]) and one
    FINAL (momenta[2]) momentum. Because t mixes an initial and a final
    leg, the particle ordering in `momenta` matters here.
    """
    p1 = momenta[0]                      # initial particle 1
    p3 = momenta[2]                      # final particle 1

    E1 = np.sqrt(np.sum(p1**2, axis=0) + M_PHI**2)
    E3 = np.sqrt(np.sum(p3**2, axis=0) + M_PHI**2)

    # Mandelstam t = (E1 - E3)^2 - |p1 - p3|^2
    t = (E1 - E3)**2 - np.sum((p1 - p3)**2, axis=0)

    return coupling**2 / ((t - M_MED**2)**2 + (GAMMA * M_MED)**2)


# ======================================================================
# Initial condition
# ======================================================================
def init_f(r, r0=3.0, width=2.0):
    """Non-thermal sigmoid distribution."""
    return 1.0 / (1.0 + np.exp((r - r0) / width))


# ======================================================================
# Setup
# ======================================================================
# Choose which propagator to use by swapping the function below.
# matrix_element = matrix_element_schannel
matrix_element = matrix_element_tchannel

solver = BEST(q_min=0.1, q_max=50.0, n_grid=40)
solver.initialize_species('phi', init_f, stat='boson', mass=M_PHI)
solver.add_process('scatter',
                   ['phi', 'phi'], ['phi', 'phi'],
                   matrix_element, coupling=1.0,
                   neval=int(1e5))

history = solver.init_history()


# ======================================================================
# Evolution (short run: demonstrates the interface, not equilibration)
# ======================================================================
n_steps = 10
dt = 1e5

for step in range(n_steps):
    solver.evolve_step(dt=dt)
    m = solver.record(history)
    if solver.world_rank == 0:
        N0, E0 = history['phi']['n'][0], history['phi']['e'][0]
        print(f"  N/N0 = {m['phi']['n'] / N0:.6f}   "
              f"E/E0 = {m['phi']['e'] / E0:.6f}")
        solver.save_checkpoint('checkpoint.pkl', history=history)
