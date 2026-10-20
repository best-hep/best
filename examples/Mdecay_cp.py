"""
CP asymmetry from the decay N <-> Phi l as a result of the run (besthep >= 1.3.1):
the decay and its CP conjugate as two processes with their own |M|^2, l and lbar
as species, the asymmetry n_l - n_lbar read off the distributions.

    decA   N <-> Phi l          (1 + eps) |M_0|^2
    decB   N <-> Phibar lbar    (1 - eps) |M_0|^2,  the CP conjugate of decA
    bath   relaxation of l and lbar to the plasma (custom term with its Jacobian)

N (Majorana, g = 2) is momentum resolved and starts empty; Phi, Phibar are the
prescribed thermal bath of Hahn-Woernle, Pluemacher & Wong (HPW), JCAP08(2009)028.
The two decays are integrated on the same Vegas samples, so f_l - f_lbar carries
the noise of order eps times the rate, not that of two independent rates; the
reverse |M|^2 of each is its own forward one (the narrow-width effective set,
HPW eqs. 3.9 to 3.11). HPW keep the leptons thermal and in kinetic equilibrium;
here that is a relaxation term, C_l = -Gamma_g (f_l - f_eq (1 + mu)) and its
conjugate with f_eq (1 - mu), mu fixed by n_l - n_lbar, so the lepton number is
conserved exactly and the pair relaxes to the bath. The step is exprb_full.

Printed: kappa_f = (4/3) (n_l - n_lbar) / (eps n_gamma) against the integrated
reference D1 (HPW eqs. 3.16 and 3.23) solved alongside, the departure of the
pair from its equilibrium number and the part of delta f outside the thermal
shape. With the defaults, kappa_f / D1 = 0.967 at K = 10 and 0.990 at K = 0.1
against 1.00 and 0.97 from the mode equations solved exactly with the leptons
in kinetic equilibrium (scripts/check_asymmetry.py). scripts/plot_cp.py plots
the saved npz.

Usage: python3 examples/Mdecay_cp.py [K] [mb|qs] [DT_FRAC_MAX] [z_stop] [eps]     (defaults 0.1, mb, 0.03, 30, 1e-6)
"""
import os, sys
import numpy as np
from scipy.special import kv, kve, zeta
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST

# --------------------------- scenario --------------------------------------
M, G_N, G_BATH = 1.0, 2, 2                 # N mass (code unit); dof of N, of Phi and of l (doublets)
K       = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
MB      = not (len(sys.argv) > 2 and sys.argv[2] == 'qs')   # Maxwell-Boltzmann statistics: HPW case D2; 'qs': case D4
DT_FRAC_MAX = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
z_init, z_stop = 0.1, (float(sys.argv[4]) if len(sys.argv) > 4 else 30.0)
EPS     = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0e-6   # CP asymmetry of the decay; kappa does not depend on it
GAMMA_G_OVER_H = 1.0e3                     # relaxation rate of the leptons to the plasma, over H at T = M (scales as T)

# --------------------------- cosmology (RD, const dof) ----------------------
M_phys, M_Pl_red_phys = 1.0e10, 2.435e18
GSTAR  = 106.75
M_Pl   = M_Pl_red_phys / M_phys
T0     = M / z_init
H_of_T = lambda T: (np.pi / np.sqrt(90.0)) * np.sqrt(GSTAR) * T**2 / M_Pl
t0     = 1.0 / (2.0 * H_of_T(T0))

def scale_factor(t):
    out = np.sqrt(np.asarray(t, float) / t0)
    return float(out) if out.ndim == 0 else out

def z_of_t(t):
    return M * scale_factor(t) / T0

Gamma_rf = K * H_of_T(M)
M2_DEC   = 16.0 * np.pi * M * Gamma_rf * G_N       # both channels, summed over all internal states, massless two-body phase space

def mesq_A(momenta):  return np.full(momenta.shape[2], 0.5 * (1.0 + EPS) * M2_DEC)
def mesq_B(momenta):  return np.full(momenta.shape[2], 0.5 * (1.0 - EPS) * M2_DEC)

def init_N(q):   return 0.0
def init_Phi(q): return np.exp(-q / T0) if MB else 1.0 / np.expm1(q / T0)
def init_l(q):   return np.exp(-q / T0) if MB else 1.0 / (np.exp(q / T0) + 1.0)
STAT = {'N': 'mb', 'Phi': 'mb', 'l': 'mb'} if MB else {'N': 'fermion', 'Phi': 'boson', 'l': 'fermion'}

# --------------------------- numerics --------------------------------------
q_min, q_max, n_grid = 0.005 * T0, 50.0 * T0, 80
neval, n_steps = int(1e4), 2000
DN_MAX = 0.05                              # allowed change of the comoving N number per step

# --------------------------- setup -----------------------------------------
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid)
solver.scale_factor = scale_factor
solver.initialize_species('N', init_N, stat=STAT['N'], mass=M, dof=G_N, antiparticle='N')
for sp, anti in (('Phi', 'Phibar'), ('Phibar', 'Phi')):
    solver.initialize_species(sp, init_Phi, stat=STAT['Phi'], mass=0.0, dof=G_BATH, antiparticle=anti)
for sp, anti in (('l', 'lbar'), ('lbar', 'l')):
    solver.initialize_species(sp, init_l, stat=STAT['l'], mass=0.0, dof=G_BATH, antiparticle=anti)
solver.add_process('decA', ['N'], ['Phi', 'l'], mesq_A, neval=neval, nitn=2, mesq_backward=mesq_A)
solver.add_process('decB', ['N'], ['Phibar', 'lbar'], mesq_B, neval=neval, nitn=2, mesq_backward=mesq_B, cp_conjugate='decA')

# relaxation of the leptons to the plasma: the targets f_eq (1 +- mu) share the
# lepton number of the pair, so the term conserves it exactly (same trapezoid as
# the numbers below) and restores the number of the pair and the shape of each
q_l   = solver.r_grids['l']
w_l   = np.zeros_like(q_l); w_l[:-1] += np.diff(q_l) / 2; w_l[1:] += np.diff(q_l) / 2; w_l *= q_l * q_l   # trapezoid weights
f_eq  = solver.distributions_1d['l'].copy()
g_eq  = f_eq * (1.0 - (0.0 if MB else f_eq))   # shape of the asymmetry in kinetic equilibrium: df/dmu of the thermal distribution
D_eq  = 2.0 * w_l @ g_eq

def Gamma_g():
    return GAMMA_G_OVER_H * H_of_T(M) / scale_factor(solver.current_time) / z_init   # Gamma_g = GAMMA_G_OVER_H H(M) T/M

def bath(s):
    G = Gamma_g()
    mu = (w_l @ s.distributions_1d['l'] - w_l @ s.distributions_1d['lbar']) / D_eq
    return {'l': (G * (f_eq + mu * g_eq), G * s.distributions_1d['l']),
            'lbar': (G * (f_eq - mu * g_eq), G * s.distributions_1d['lbar'])}

def bath_jacobian(s):
    G = Gamma_g()
    I = np.eye(len(q_l)); R = np.outer(g_eq, w_l) / D_eq
    return {'l': {'l': G * (R - I), 'lbar': -G * R},
            'lbar': {'l': -G * R, 'lbar': G * (R - I)}}

solver.add_custom_term('bath', bath, jacobian=bath_jacobian)
solver.species_list = ['N', 'l', 'lbar']            # Phi, Phibar prescribed
solver.current_time = t0

n_gamma_com = 2.0 * zeta(3) / np.pi**2 * T0**3      # photon density times a^3

def comoving_number(sp):
    q = solver.r_grids[sp]
    return solver.species_dof[sp] / (2 * np.pi**2) * np.trapezoid(q * q * solver.distributions_1d[sp], q) / n_gamma_com

def non_thermal():
    """Fraction of the asymmetry that is not in the thermal shape, sum |delta f - m f_l| /
    sum |delta f| with m = n_asym / n_l over the grid: zero if delta f is proportional
    to f_l (kinetic equilibrium, which HPW assume)."""
    f = solver.distributions_1d['l']; df = f - solver.distributions_1d['lbar']
    m = (w_l @ df) / (w_l @ f)
    return (w_l @ np.abs(df - m * f)) / (w_l @ np.abs(df))

zs, NN, NL, NPAIR = [], [], [], []
def record():
    n = {sp: comoving_number(sp) for sp in ('N', 'l', 'lbar')}
    zs.append(z_of_t(solver.current_time)); NN.append(n['N']); NL.append(n['l'] - n['lbar']); NPAIR.append(n['l'] + n['lbar'])
record()

if solver.world_rank == 0:
    print(f"\nCP run ({'Maxwell-Boltzmann, case D2' if MB else 'quantum statistics, case D4'}): K = {K}, Gamma_rf = {Gamma_rf:.3e}, "
          f"H(M) = {H_of_T(M):.3e}, |M|^2_dec = {M2_DEC:.3e}, Gamma_g/H = {GAMMA_G_OVER_H:g} M/T, epsilon = {EPS}, z {z_init} -> {z_stop}")

# --------------------------- evolution -------------------------------------
rate = 0.0                                          # |d ln N / d ln t| of the step before the last, once N is seeded
for step in range(n_steps):
    if z_of_t(solver.current_time) > z_stop:
        break
    t_now = solver.current_time
    dt = (DT_FRAC_MAX if rate * DT_FRAC_MAX <= DN_MAX else DN_MAX / rate) * t_now
    solver.evolve_step(dt, method='exprb_full', adapt_dt=False)
    record()
    if step >= 2:                                   # the records before are the empty start and the seeded N
        rate = abs(np.log(NN[-2] / NN[-3])) / (2.0 * np.log(zs[-2] / zs[-3]))   # t proportional to z^2
    if solver.world_rank == 0 and solver.step_count % 20 == 0:
        print(f"{solver.step_count:>4} | z = {zs[-1]:7.3f} | dt/t = {dt / t_now:.4f} | N_N = {NN[-1]:.4e} | "
              f"N_asym = {NL[-1]:+.4e} | kappa = {4.0 / 3.0 * NL[-1] / EPS:+.4e} | (l+lbar)/eq - 1 = {NPAIR[-1] / NPAIR[0] - 1:+.1e} | "
              f"non-thermal delta f = {non_thermal():.1e}", flush=True)

# --------------------------- reference: integrated equations D1 --------------
NEQ_COEF = G_N / (4.0 * zeta(3))      # N_eq = NEQ_COEF z^2 K2(z), the Maxwell-Boltzmann number like the run's N at z >> 1
HPW_D1   = 3.0 / 4.0 * zeta(3)        # HPW's (3/8) z^2 K2 is this times smaller; D1 is linear in N_eq

def D1_rhs(z, y):
    NNi, NLi = y
    D = z * K * kve(1, z) / kve(2, z)
    Neq = NEQ_COEF * z * z * kv(2, z)
    W_ID = K * z**3 * kv(1, z) / 4.0
    return [-D * (NNi - Neq), -W_ID * NLi + EPS * D * (NNi - Neq)]

z_grid = np.array(zs)
RTOL = 1e-8
ref = solve_ivp(D1_rhs, (z_init, max(z_stop, z_grid[-1])), [0.0, 0.0], method='Radau',
                dense_output=True, rtol=RTOL, atol=RTOL * EPS)     # absolute tolerance at the scale of the asymmetry
NN_D1, NL_D1 = ref.sol(z_grid)

kappa = 4.0 / 3.0 * NL[-1] / EPS
kappa_D1 = 4.0 / 3.0 * NL_D1[-1] / EPS
if solver.world_rank == 0:
    flip = next((zs[i] for i in range(1, len(NL)) if NL[i - 1] < 0 <= NL[i]), None)
    print(f"\nK = {K}: kappa_f({'D2' if MB else 'D4'}, besthep CP run) = {kappa:+.4e}, kappa_f(D1, integrated) = {kappa_D1:+.4e}, "
          f"ratio = {kappa / kappa_D1:.3f} ({kappa / kappa_D1 / HPW_D1:.3f} to D1 with HPW's N_eq = 3/8 z^2 K2); "
          f"sign flip of the asymmetry at z = {flip}; (l+lbar)/eq - 1 = {NPAIR[-1] / NPAIR[0] - 1:+.1e}; "
          f"non-thermal part of delta f = {non_thermal():.1e}")
    np.savez(f"cp_K{K:g}{'_mb' if MB else ''}.npz", z=z_grid, N_N=np.array(NN), N_asym=np.array(NL), N_pair=np.array(NPAIR),
             N_N_D1=NN_D1, N_asym_D1=NL_D1, f_l=solver.distributions_1d['l'], f_lbar=solver.distributions_1d['lbar'],
             q=q_l, K=K, eps=EPS)