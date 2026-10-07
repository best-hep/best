"""
Final-stage test (besthep >= 1.3.0): decay, fermions, a self-consistent thermal
bath and expansion in one run.

    dec  N  <-> Phi l        Majorana N (g = 2) decays into a boson Phi and a fermion l
    el   Phi l -> Phi l      contact scattering that keeps the bath thermal (stiff)

All three species evolve; nothing is prescribed. N starts empty (seeded from its
production spectrum = inverse decays), fills up to equilibrium while Gamma/H > 1,
and decays out of equilibrium for z = M/T > 1. The bath carries large dof
(G_BATH) so that the energy released by the decays leaves T proportional to 1/a,
which makes the N(z) curve comparable to the prescribed-bath benchmark of
Hahn-Woernle, Pluemacher & Wong, JCAP08(2009)028 (case D4, epsilon = 0).

The bath species carry a thermal mass m = M_BATH_OVER_T * T, i.e. m proportional
to 1/a (well below M, so the decay stays open): an interacting bath has one, and
without it the rates grow as 1/E towards the infrared and the softest modes
become arbitrarily stiff, which no time stepper resolves.

Checks: N(z) against the prescribed-bath run / the benchmark; Phi and l staying
Bose-Einstein / Fermi-Dirac at T = T0/a; conservation of the total comoving energy
(up to the redshift of the mass terms). Cosmology: RD, constant g_* = 106.75.

Matrix-element convention: |M|^2 summed over the internal states of
every leg; the solver divides by the observed species' dof. For a spin-blind
contact amplitude the sum over states is the product of all dof times lambda^2.

Delete any old checkpoint.pkl before a fresh run (resume fires on file existence).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST

# --------------------------- scenario --------------------------------------
M       = 1.0                  # N mass, code unit
G_N     = 2                    # Majorana: two helicity states
G_BATH  = 50                   # dof of Phi and of l: large, so the bath is a heat reservoir
K       = 1.0                  # Gamma_rf / H(T = M): decay parameter
z_init, z_stop = 0.1, 10.0     # z = M/T
GAMMA_EL_OVER_H = 1.0e3        # target elastic rate / H at T = M (order of magnitude)
M_BATH_OVER_T   = 0.02         # thermal mass of the bath species, m = 0.02 T (m proportional to 1/a; must stay well below M)

# --------------------------- cosmology (RD, const dof) ----------------------
M_phys, M_Pl_red_phys = 1.0e10, 2.435e18          # GeV
GSTAR  = 106.75
M_Pl   = M_Pl_red_phys / M_phys                   # code units
a0     = 1.0
T0     = M / z_init
H_of_T = lambda T: (np.pi / np.sqrt(90.0)) * np.sqrt(GSTAR) * T**2 / M_Pl
t0     = 1.0 / (2.0 * H_of_T(T0))

def scale_factor(t):
    out = a0 * np.sqrt(np.asarray(t, float) / t0)
    return float(out) if out.ndim == 0 else out

def T_of_a(a):
    return T0 * a0 / a

m_bath0 = M_BATH_OVER_T * T0                      # bath mass at a0; m(t) = m_bath0 / a(t)
def m_bath(t):
    return m_bath0 / scale_factor(t)

# --------------------------- couplings -------------------------------------
Gamma_rf = K * H_of_T(M)                          # rest-frame decay width
M2_DEC   = 16.0 * np.pi * M * Gamma_rf * G_N      # sum over all states, 2-body massless phase space
# elastic: per-particle rate ~ (M2_EL/G_BATH) T/2e3 for massless partners (rough);
# M2_EL below gives Gamma_el/H ~ GAMMA_EL_OVER_H at T = M
M2_EL    = GAMMA_EL_OVER_H * H_of_T(M) * G_BATH * 2.0e3 / M

def matrix_element_dec(momenta):
    return np.full(momenta.shape[2], M2_DEC)

def matrix_element_el(momenta):
    return np.full(momenta.shape[2], M2_EL)

# --------------------------- initial spectra (comoving q, a0 = 1) ------------
def init_N(q):   return 0.0                              # empty: seeded at the first step
def init_Phi(q): return 1.0 / np.expm1(np.clip(np.sqrt(q**2 + m_bath0**2) / T0, 1e-12, 700.0))
def init_l(q):   return 1.0 / (np.exp(np.clip(np.sqrt(q**2 + m_bath0**2) / T0, -700.0, 700.0)) + 1.0)

# --------------------------- numerics --------------------------------------
q_min, q_max, n_grid = 0.005 * T0, 50.0 * T0, 80
neval, n_steps = int(1e4), 1000
DT_FRAC_MAX = 0.03             # dt/t while N does not change
DN_MAX      = 0.05             # allowed change of the comoving N number per step
checkpoint_file = "checkpoint.pkl"

# --------------------------- setup -----------------------------------------
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid)
solver.verbose = True
solver.scale_factor = scale_factor

resume = os.path.exists(checkpoint_file) if solver.world_rank == 0 else None
resume = solver.world_comm.bcast(resume, root=0)
if resume:
    history = solver.load_checkpoint(
        checkpoint_file,
        matrix_elements_squared={'dec': matrix_element_dec, 'el': matrix_element_el})
    solver.scale_factor = scale_factor      # not checkpointed
else:
    solver.initialize_species('N',   init_N,   stat='fermion', mass=M,       dof=G_N)
    solver.initialize_species('Phi', init_Phi, stat='boson',   mass=m_bath0, dof=G_BATH)
    solver.initialize_species('l',   init_l,   stat='fermion', mass=m_bath0, dof=G_BATH)
    solver.add_process('dec', ['N'], ['Phi', 'l'], matrix_element_dec, neval=neval, nitn=2)
    solver.add_process('el', ['Phi', 'l'], ['Phi', 'l'], matrix_element_el, neval=neval, nitn=2)
    solver.current_time = t0
    history = solver.init_history()
solver.set_mass_func('Phi', m_bath)             # not checkpointed: set on both paths
solver.set_mass_func('l', m_bath)

# --------------------------- evolution -------------------------------------
if solver.world_rank == 0:
    print(f"\nMajorana decay run: K = {K}, Gamma_rf = {Gamma_rf:.3e}, H(M) = {H_of_T(M):.3e}, "
          f"|M|^2_dec = {M2_DEC:.3e}, |M|^2_el = {M2_EL:.3e}, m_bath/T = {M_BATH_OVER_T}, "
          f"z {z_init} -> {z_stop}")

rate = DN_MAX / 0.05                       # prior: dt/t = 0.05 until the number change has been measured
n_prev, t_prev = None, None
for step in range(n_steps):
    a = solver.scale_factor(solver.current_time)
    if M / T_of_a(a) > z_stop:
        break
    t_now, n_now = solver.current_time, history['N']['n'][-1]
    dt = min(DT_FRAC_MAX, DN_MAX / rate) * t_now          # dt/t = min(ceiling, DN_MAX / |d ln N/d ln t|)
    solver.evolve_step(dt, method='exprb_full', adapt_dt=False)
    m = solver.record(history)
    if n_prev is not None and n_prev > 1e-100:             # skip the empty initial record
        rate = max(abs(np.log(n_now / n_prev)) / np.log(t_now / t_prev), 1e-300)
    n_prev, t_prev = n_now, t_now
    if solver.world_rank == 0 and solver.step_count % 10 == 0:
        a = solver.scale_factor(solver.current_time)
        e_tot = m['N']['e'] + m['Phi']['e'] + m['l']['e']
        print(f"{solver.step_count:>4} | z = {M / T_of_a(a):7.3f} | dt/t = {dt / t_now:.4f} | "
              f"N_com = {m['N']['n']:.4e} | Phi_com = {m['Phi']['n']:.4e} | l_com = {m['l']['n']:.4e} | "
              f"E_com = {e_tot:.6e}")
    if solver.step_count % 10 == 0:
        solver.save_checkpoint(checkpoint_file, history=history)
