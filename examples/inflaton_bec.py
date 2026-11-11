"""
A condensate decaying into a self-interacting boson that condenses in turn (besthep >= 1.3.2):
the two zero modes are the solver's own, nothing is added here.

    dec    phi -> chi chi, constant |M|^2 set by the width Gamma_phi
    el     chi chi <-> chi chi, constant |M|^2 = lambda^2

The inflaton phi starts as a condensate only, n_phi0 in its zero mode (p < q_min) and nothing on
its grid, and decays at rest: each decay puts two chi at p* = sqrt(m_phi^2/4 - m_chi^2)
on the chi grid, where the elastic scattering thermalizes them. Without expansion the
decays fix the energy per chi at m_phi/2, so once the delivered number exceeds what the
Bose-Einstein distribution with mu = m_chi holds at that energy, the excess has nowhere
to go on the grid: it goes into chi's own zero mode, solver.zero_mode['chi'], a condensate of
the decay products. The occupation at p* enhances the decay (the zero mode decays at
(1 + f(p*))^2 times the bare rate once the deposit line is occupied), and inverse decays
feed a thermal phi population onto the phi grid, which keeps the delivered number short
of 2 n_phi0. The end state follows from the two conserved totals, the energy m_phi n_phi0
and the number N_chi + 2 N_phi = 2 n_phi0: chi at mu = m_chi with its condensate and the
thermal phi in chemical equilibrium with it (mu_phi = 2 mu_chi) at one temperature. The
header prints that prediction; the end compares the run with it.

Printed: the inflaton zero mode against exp(-Gamma_phi t) and the thermal phi on its grid, the
chi grid and zero mode as fractions of 2 n_phi0, the totals of number and energy (the check of
the step), the temperature and chemical potential fitted to the chi grid, the occupation
at the deposit line and the lowest node over its critical value. At the end, the chi zero
mode and the thermal phi against the prediction, the time the grid turned supercritical
against the expectation at the bare decay rate (earlier, by the enhancement). Early on, before
the scattering has spread the daughters, the occupation at p* is that of the deposit line, whose
width is the grid's: that part of the enhancement, and this time with it, move with N_GRID; the
end state does not. checkpoint.pkl (state and history) is written every step and a run resumes
from it; delete it to start fresh.

The step is the full-Jacobian exponential one (see the loop); it needs dt times the elastic
rate below one (the header prints the product), so a larger coupling asks for a smaller step.

"""
import os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST, grid_weights

# n_phi0: the inflaton number density (m_chi^3); the decays deliver 2 n_phi0, about three times the critical number (header).
# lambda: the quartic coupling, |M|^2 = lambda^2; scattering some twenty times faster than the decay at n_phi0 (header).
M_CHI     = 1.0                                                   # the unit of energy
M_PHI     = 4.0 * M_CHI                                           # each chi is born with the kinetic energy m_chi: p* = sqrt(3) m_chi
N_PHI0    = 0.25
LAMBDA    = 3.0
N_GRID    = 80
NEVAL     = 10000
GAMMA_PHI = 1e-4 * M_CHI                                          # the width, a narrow resonance; 1/Gamma_phi is the run's time unit
P_STAR    = np.sqrt(M_PHI**2 / 4.0 - M_CHI**2)                    # the momentum of the daughters
EPS       = M_PHI / 2.0 - M_CHI                                   # the kinetic energy each daughter is born with
Q_MIN     = 0.02 * M_CHI                                          # the zero mode p < q_min, deep in the thermal gas: 2 m T/q_min^2 >> 1
Q_MAX     = 5.0 * P_STAR                                          # holds the thermal tail, T staying below p*
DT        = 0.01 / GAMMA_PHI                                      # resolves the decay and the scattering: dt x elastic rate < 1 (header)
T_END     = 12.0 / GAMMA_PHI   # a few decay times: the inflaton gone, the gas settled
T_REPORT  = 0.1 / GAMMA_PHI

CHECKPOINT = 'checkpoint.pkl'                                      # saved every step; a run resumes from it (delete to start fresh)

solver = BEST(q_min=Q_MIN, q_max=Q_MAX, n_grid=N_GRID)


def mesq_dec(mom):
    """|M|^2 of phi -> chi chi from the width: the solver decays a particle of the zero mode at the rate
    sym |M|^2 p*/(8 pi m_phi^2 g_phi) (_zero_mode_line), sym = 1/2 for two identical daughters."""
    sym, g_phi = solver.process_configs['dec']['symmetry_factor'], solver.species_dof['phi']
    return np.full(mom.shape[2], 8.0 * np.pi * M_PHI**2 * g_phi * GAMMA_PHI / (sym * P_STAR))


def mesq_el(mom):
    return np.full(mom.shape[2], LAMBDA**2)


resume = solver.world_comm.bcast(os.path.exists(CHECKPOINT) and solver.world_rank == 0, root=0)
if resume:
    history = solver.load_checkpoint(CHECKPOINT, matrix_elements_squared={'dec': mesq_dec, 'el': mesq_el})
else:
    solver.initialize_species('phi', lambda q: 0.0 * q, stat='boson', mass=M_PHI, n0=N_PHI0)   # the condensate; nothing on the grid
    solver.initialize_species('chi', lambda q: 0.0 * q, stat='boson', mass=M_CHI)             # empty: seeded from the decay at step one
    solver.add_process('dec', ['phi'], ['chi', 'chi'], mesq_dec, neval=NEVAL, nitn=2)
    solver.add_process('el', ['chi', 'chi'], ['chi', 'chi'], mesq_el, neval=NEVAL, nitn=2)
    history = solver.init_history()
q = solver.r_grids['chi']
E = np.sqrt(q * q + M_CHI**2)
w = grid_weights(q) * q * q / (2.0 * np.pi**2)       # the solver's number weights
q_phi = solver.r_grids['phi']
E_OF = {'chi': E, 'phi': np.sqrt(q_phi * q_phi + M_PHI**2)}
W_OF = {'chi': w, 'phi': grid_weights(q_phi) * q_phi * q_phi / (2.0 * np.pi**2)}
i_star = np.argmin(np.abs(q - P_STAR))               # the node of the deposit line
rank0 = solver.world_rank == 0
# Thermalization against the decay: a chi scatters at the rate n sigma v, sigma = lambda^2/(32 pi s) for
# identical particles and s = 2 m_chi^2 + 2 E*^2 on average for two daughters (v ~ 1); at n = n_phi0:
GAMMA_EL = N_PHI0 * LAMBDA**2 / (64.0 * np.pi * (M_CHI**2 + M_PHI**2 / 4.0))


def root(g, x):
    """The root of the increasing function g, bracketed by doubling from the scale x."""
    while g(x) > 0.0:
        x /= 2.0
    while g(2.0 * x) < 0.0:
        x *= 2.0
    return brentq(g, x, 2.0 * x)


def bose(T, mu, sp):
    """Number and energy on the grid of species sp with the Bose-Einstein distribution; x > 0 for mu <= m, nothing overflows."""
    E_sp, w_sp = E_OF[sp], W_OF[sp]
    x = (E_sp - mu) / T
    f = np.exp(-x) / -np.expm1(-x)
    return w_sp @ f, w_sp @ (E_sp * f)


def chi(T, mu):
    return bose(T, mu, 'chi')


def gas(T, mu):
    """Thermal chi at mu with the thermal phi in chemical equilibrium with it (mu_phi = 2 mu): number counting phi
    twice (what the decays conserve), and energy."""
    n_chi, e_chi = bose(T, mu, 'chi')
    n_phi, e_phi = bose(T, 2.0 * mu, 'phi')
    return n_chi + 2.0 * n_phi, e_chi + e_phi


def critical(eps):
    """T and number of the critical chi gas (mu = m, no condensate yet) with the kinetic energy eps per particle."""
    kinetic = lambda T: (lambda nt, et: (et - M_CHI * nt, nt))(*chi(T, M_CHI))
    T = root(lambda T: kinetic(T)[0] / kinetic(T)[1] - eps, eps)
    return T, kinetic(T)[1]


def equilibrium(n, e, parts=gas):
    """T, mu and the chi condensate of the system with number n and energy e, moments taken on the grid as the
    solver does; parts(T, mu) is its thermal number and energy (gas: chi and phi, chi: the chi grid alone)."""
    excess = lambda T: (lambda nt, et: et - M_CHI * nt)(*parts(T, M_CHI)) - (e - M_CHI * n)   # energy above m per particle
    T = root(excess, (e - M_CHI * n) / n)
    nc = n - parts(T, M_CHI)[0]
    if nc > 0.0:
        return T, M_CHI, nc
    mu_of = lambda T: M_CHI - root(lambda d: n - parts(T, M_CHI - d)[0], T)     # mu < m holding the number n at this T
    T = root(lambda T: parts(T, mu_of(T))[1] - e, T)
    return T, mu_of(T), 0.0


T_CRIT, N_CRIT = critical(EPS)
T_eq, mu_eq, nc_eq = equilibrium(2.0 * N_PHI0, M_PHI * N_PHI0)           # the prediction: chi, its condensate, the thermal phi
nphi_eq = bose(T_eq, 2.0 * mu_eq, 'phi')[0]
share = N_CRIT / (2.0 * N_PHI0)                      # the share of phi that must have decayed for the grid to be supercritical
onset = (f"once a share {share:.4f} of phi has decayed, t Gamma_phi = {-np.log1p(-share):.4f} at the bare rate" if share < 1.0
         else "never: the delivered number stays below critical")
if rank0:
    print(f"\nm_phi = {M_PHI / M_CHI} m_chi: p* = {P_STAR:.4f}, kinetic energy per chi {EPS:.4f}; Gamma_phi = {GAMMA_PHI:g}, "
          f"lambda = {LAMBDA}: elastic rate at n_phi0 ~ {GAMMA_EL / GAMMA_PHI:.0f} Gamma_phi, dt x rate ~ {DT * GAMMA_EL:.2f}")
    print(f"critical number at this energy per particle n_crit = {N_CRIT:.4f} (T = {T_CRIT:.4f}); the decays deliver 2 n_phi0 = "
          f"{2.0 * N_PHI0:.4f} = {1.0 / share:.2f} n_crit: a condensate is {'expected' if nc_eq > 0.0 else 'not expected'}")
    print(f"predicted end state (energy m_phi n_phi0, number N_chi + 2 N_phi = 2 n_phi0, chemical equilibrium): T = {T_eq:.4f}, "
          f"mu_chi = {mu_eq:.4f}, chi condensate {nc_eq / (2.0 * N_PHI0):.4%} of 2 n_phi0, thermal phi {nphi_eq / N_PHI0:.4f} n_phi0; "
          f"the grid turns supercritical {onset}")
    print(f"\n{'t Gam':>6} {'N_phi/n0':>8} {'exp(-Gt)':>8} {'phi grid':>8} {'chi grid':>8} {'chi n0':>9} {'N_tot':>7} {'E_tot':>7} "
          f"{'T_fit':>6} {'mu/m':>7} {'f(p*)':>7} {'f(qmin)/crit':>12}")


def supercritical(n, e, nc):
    """Whether the grid alone (number n - nc, energy e - m nc) holds more than its critical number."""
    return equilibrium(n - nc, e - M_CHI * nc, chi)[2] > 0.0


def report(mom):
    f = solver.distributions_1d['chi']
    N0, nc = solver.zero_mode['phi'], solver.zero_mode['chi']
    b, a = np.polyfit(E, np.log1p(1.0 / f), 1, w=np.sqrt(w * f))      # log(1 + 1/f) = (E - mu)/T; each node weighs its number: no window
    T_fit, mu_fit = 1.0 / b, -a / b
    if rank0:
        print(f"{solver.current_time * GAMMA_PHI:6.3f} {N0 / N_PHI0:8.5f} {np.exp(-GAMMA_PHI * solver.current_time):8.5f} "
              f"{(mom['phi']['n'] - N0) / N_PHI0:8.5f} {(mom['chi']['n'] - nc) / (2.0 * N_PHI0):8.4f} {nc / (2.0 * N_PHI0):9.2e} "
              f"{(mom['chi']['n'] + 2.0 * mom['phi']['n']) / (2.0 * N_PHI0):7.4f} "
              f"{(mom['chi']['e'] + mom['phi']['e']) / (M_PHI * N_PHI0):7.4f} {T_fit:6.3f} {mu_fit / M_CHI:7.4f} {f[i_star]:7.3f} "
              f"{f[0] * np.expm1((E[0] - M_CHI) / T_fit):12.3f}", flush=True)


mom = solver.compute_moments()
next_report = T_REPORT * (np.floor(solver.current_time / T_REPORT) + 1.0)
while solver.current_time < T_END:
    # exprb_full: at the onset of condensation the lowest nodes empty into the chi zero mode within a step; the
    # diagonal exponential steps (exprb, exprb2, exprb_seq) hold the partner legs of each node at their values
    # at the start of the step, so the gain they feed to the rest of the grid outlives the nodes it came from,
    # and number and energy are created. The full Jacobian carries the response of every node to every other
    # (and to the zero modes), and the totals printed below stay conserved.
    solver.evolve_step(DT, method='exprb_full', adapt_dt=False)
    mom = solver.record(history)
    solver.save_checkpoint(CHECKPOINT, history=history)
    if next_report <= solver.current_time < T_END:
        report(mom)
        next_report += T_REPORT
report(mom)
h = history['chi']                                   # the first recorded state (after the initial one, chi empty) with a supercritical grid
t_super = next((t for t, n, e, nc in zip(history['times'][1:], h['n'][1:], h['e'][1:], h['n0'][1:]) if supercritical(n, e, nc)), None)
if rank0:
    print(f"\nchi zero mode / (2 n_phi0) = {solver.zero_mode['chi'] / (2.0 * N_PHI0):.4%} (predicted {nc_eq / (2.0 * N_PHI0):.4%}); "
          f"thermal phi {(mom['phi']['n'] - solver.zero_mode['phi']) / N_PHI0:.4f} n_phi0 (predicted {nphi_eq / N_PHI0:.4f})")
    print("the grid turned supercritical "
          + (f"at t Gamma_phi = {t_super * GAMMA_PHI:.4f}" if t_super is not None else "at no time in the run") + f"; expected {onset}")
