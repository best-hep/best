"""
DRAKE 4.3 sub-threshold benchmark -- PROTOCOL run (comparison vs reference).

Const-dof pipeline (DRAKE dofimport["const"]): heff = geff = 10.2^2 = 104.04
everywhere -- cosmology and entropy. No dof table anywhere in this pipeline;
post-processing must use the SAME constant (do NOT run the Drees-table plot
script on this checkpoint).

Channels (same contact vertex, lambda = 1):
    ann  phi1 phi1 -> phi2 phi2 : |M|^2 = lambda^2/4   (1/(2!*2!) folded in
                                  here; besthep supplies slot multiplicity)
    el   phi1 phi2 -> phi1 phi2 : |M|^2 = lambda^2     (no identical legs)


All outputs live in the checkpoint (history included); post-processing reads
the checkpoint. Delete any old checkpoint.pkl before a fresh run (resume
fires on file existence).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST

# --------------------------- scenario --------------------------------------
m1, r    = 1.0, 1.1
m2       = r * m1
x_init   = 15.0                # reference IC snapshot;
T0_bath  = m1 / x_init
coupling = 1.0                 # = lambda

# --------------------------- cosmology (CONST dof) --------------------------
m1_phys, M_Pl_red_phys = 100.0, 2.435e18       # GeV
a0      = 1.0
x_stop  = 40.0              
HEFF    = 10.2**2              # = geff = 104.04 ;
SQGEFF  = 10.2

M_Pl = M_Pl_red_phys / m1_phys                  # code units
T0c  = 1.0 / x_init
H0   = (np.pi / np.sqrt(90.0)) * SQGEFF * T0c**2 / M_Pl
t0   = 1.0 / (2.0 * H0)

def scale_factor(t):
    out = a0 * np.sqrt(np.asarray(t, float) / t0)
    return float(out) if out.ndim == 0 else out

def T_of_a(a):
    out = T0c * a0 / np.asarray(a, float)
    return float(out) if out.ndim == 0 else out

# --------------------------- numerics --------------------------------------
q_min, q_max, n_grid = 0.1, 4, 40
neval, dt_frac, n_steps = int(1e6), 0.01, 500
delta_width, adapt_width = 0.001, False
max_rel_err, min_rel_err, max_rel_change = 0.01, 0.001, 0.3
checkpoint_file = "checkpoint.pkl"

# --------------------------- physics ---------------------------------------
def matrix_element(momenta, coupling):
    # ann: |M|^2 = lambda^2 * 1/(2!*2!) -- identical pairs on both sides
    return np.full(momenta.shape[2], coupling**2 / 4.0)

def matrix_element_el(momenta, coupling):
    # el: |M|^2 = lambda^2 -- no identical legs, NO /4
    return np.full(momenta.shape[2], coupling**2)

def be_on_grid(q_grid, mass, T, a, mu=0.0):
    p = np.asarray(q_grid, float) / a
    E = np.sqrt(p**2 + mass**2)
    z = np.clip((E - mu) / T, -700, 700)
    return 1.0 / np.expm1(z)

class PhiTwoBath:
    def __init__(self, mass, mu=0.0):
        self.mass, self.mu, self.a, self.T = mass, mu, 1.0, 1.0
    def set_state(self, a, T):
        self.a, self.T = a, T
    def __call__(self, q):
        p = np.asarray(q, float) / self.a
        E = np.sqrt(p**2 + self.mass**2)
        z = np.clip((E - self.mu) / self.T, -700, 700)
        return 1.0 / np.expm1(z)

def init_phi1(q): return be_on_grid(q, m1, T0_bath, a0)
def init_phi2(q): return be_on_grid(q, m2, T0_bath, a0)

# --------------------------- setup -----------------------------------------
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid,
              max_rel_change=max_rel_change, adapt_width=adapt_width,
              max_rel_err=max_rel_err, min_rel_err=min_rel_err)
solver.verbose = False
solver.scale_factor = scale_factor
phi2_bath = PhiTwoBath(m2)

def prescribe_phi2():
    a = solver.scale_factor(solver.current_time)
    phi2_bath.set_state(a, T_of_a(a))
    solver.interpolators['phi2'] = phi2_bath
    solver.distributions_1d['phi2'] = phi2_bath(solver.r_grids['phi2'])

resume = os.path.exists(checkpoint_file) if solver.world_rank == 0 else None
resume = solver.world_comm.bcast(resume, root=0)
if resume:
    history = solver.load_checkpoint(
        checkpoint_file,
        matrix_elements={'matrix_element': matrix_element,
                         'matrix_element_el': matrix_element_el})
    solver.scale_factor = scale_factor      # not checkpointed
else:
    solver.initialize_species('phi1', init_phi1, stat='boson', mass=m1)
    solver.initialize_species('phi2', init_phi2, stat='boson', mass=m2)
    solver.add_process('el', ['phi1', 'phi2'], ['phi1', 'phi2'],
                       matrix_element_el, coupling=coupling,
                       neval=neval, delta_width=delta_width, nitn=2)
    solver.add_process('ann', ['phi1', 'phi1'], ['phi2', 'phi2'],
                       matrix_element, coupling=coupling,
                       neval=neval, delta_width=delta_width, nitn=2)
    solver.current_time = t0
    history = solver.init_history()

# --------------------------- evolution -------------------------------------
prescribe_phi2()

if solver.world_rank == 0:
    print(f"\nprotocol run: const dof ({HEFF}), ann + el, exprb_seq, "
          f"x {x_init} -> {x_stop}")

for step in range(n_steps):
    if m1 / T_of_a(solver.scale_factor(solver.current_time)) > x_stop:
        break
    dt = dt_frac * solver.current_time
    solver.species_list = ['phi1']          # phi2 prescribed: skip its rates
    solver.evolve_step(dt, method='exprb_seq', adapt_dt=False)
    solver.species_list = ['phi1', 'phi2']
    prescribe_phi2()
    m = solver.record(history)
    if solver.world_rank == 0 and solver.step_count % 5 == 0:
        a = solver.scale_factor(solver.current_time)
        print(f"{solver.step_count:>4} | x = {m1 / T_of_a(a):7.3f} | "
              f"n1_com = {m['phi1']['n']:.4e}")
    solver.save_checkpoint(checkpoint_file, history=history)
