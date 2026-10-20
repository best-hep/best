"""
Check of the CP-asymmetry chain against an exact solution of the mode equations.

Maxwell-Boltzmann statistics, N <-> Phi l with a prescribed thermal bath (HPW case
D2, 0907.0205): N is evolved by the Vegas decay with 'exprb2' and the lepton
asymmetry (kinetic equilibrium of the leptons) by exprb2_update from the integrated
source and the f_l-weighted wash-out (the linearized formulation of the README's
CP section). The
reference solves the same equations without Monte Carlo and without a time
stepper: in HPW's variables (y = p/T, constant in comoving momentum; E = sqrt(y^2 + z^2))
every N mode obeys the scalar linear equation

    df_N/dz = z^2 K / E * (exp(-E) - f_N),

integrated with a stiff ODE solver, and the asymmetry per photon obeys

    dN_asym/dz = -W_ID N_asym + eps * (-dN_N/dz),     W_ID = K z^3 K_1(z) / 4,

with -dN_N/dz the net decay rate of the modes. Checked: N_N(z) at a few z and
kappa at the end of the run, both within TOL. The weak wash-out case is the
sensitive one (kappa_f is a ten-percent residual of cancelling contributions).

Usage: python3 scripts/check_asymmetry.py [K ...]      (default: 10 and 0.1)
"""
import os, sys, io, contextlib
import numpy as np
from scipy.special import kv, zeta
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST, exprb2_update, assemble_A_Gamma

TOL = 0.03
CASES = {10.0: 12.0, 0.1: 20.0}                       # K: z_stop
K_LIST = [float(a) for a in sys.argv[1:]] or list(CASES)

M, G_N, G_BATH, EPS, z_init = 1.0, 2, 2, 1.0e-6, 0.1
M_Pl = 2.435e18 / 1.0e10
H_of_T = lambda T: (np.pi / np.sqrt(90.0)) * np.sqrt(106.75) * T**2 / M_Pl
T0 = M / z_init
t0 = 1.0 / (2.0 * H_of_T(T0))
scale_factor = lambda t: np.sqrt(np.asarray(t, float) / t0)
z_of_t = lambda t: M * float(scale_factor(t)) / T0
n_gamma_com = 2.0 * zeta(3) / np.pi**2 * T0**3


def run_besthep(K, z_stop, dt_frac=0.03, dn_max=0.05):
    Gamma_rf = K * H_of_T(M)
    M2 = 16.0 * np.pi * M * Gamma_rf * G_N
    sol = BEST(q_min=0.005 * T0, q_max=50.0 * T0, n_grid=80)
    sol.scale_factor = scale_factor
    sol.initialize_species('N', lambda q: 0.0, stat='mb', mass=M, dof=G_N)
    sol.initialize_species('Phi', lambda q: np.exp(-q / T0), stat='mb', mass=0.0, dof=G_BATH)
    sol.initialize_species('l', lambda q: np.exp(-q / T0), stat='mb', mass=0.0, dof=G_BATH)
    sol.add_process('dec', ['N'], ['Phi', 'l'], lambda m: np.full(m.shape[2], M2), neval=int(1e4), nitn=2)
    sol.current_time = t0
    q_l, f_l = sol.r_grids['l'], sol.distributions_1d['l']

    def number(sp, f):
        q = sol.r_grids[sp]
        return sol.species_dof[sp] / (2 * np.pi**2) * np.trapezoid(q * q * f, q)

    def coefficients():
        sol.species_list = ['N', 'l']
        sol.compute_rates(['dec'])
        G = sol.gain_rates_by_process['dec']['l'] / 2.0
        L = sol.loss_rates_by_process['dec']['l'] / 2.0
        _, W = assemble_A_Gamma(f_l, G, L, 'mb')
        sol.species_list = ['N']
        return number('l', 2.0 * EPS * (G - L)) / n_gamma_com, np.trapezoid(q_l * q_l * W * f_l, q_l) / np.trapezoid(q_l * q_l * f_l, q_l)

    zs, NN, NL = [z_init], [0.0], [0.0]
    N_asym, rate, n_prev, t_prev = 0.0, 1.0, None, None
    S1, W1 = coefficients()
    while zs[-1] < z_stop:
        t_now, n_now = sol.current_time, NN[-1]
        dt = min(dt_frac, dn_max / rate) * t_now
        sol.evolve_step(dt, method='exprb2', adapt_dt=False)
        S2, W2 = coefficients()
        N_asym = float(exprb2_update(N_asym, S1, W1, S2, W2, sol.current_time - t_now))
        S1, W1 = S2, W2
        zs.append(z_of_t(sol.current_time)); NN.append(number('N', sol.distributions_1d['N']) / n_gamma_com); NL.append(N_asym)
        if n_prev is not None and n_prev > 1e-100:
            rate = max(abs(np.log(n_now / n_prev)) / np.log(t_now / t_prev), 1e-300)
        n_prev, t_prev = n_now, t_now
    return np.array(zs), np.array(NN), np.array(NL), sol.world_rank


def exact_modes(K, z_end):
    """The mode equations solved mode by mode (scalar linear ODEs) and the asymmetry
    from their net decay rate; returns N_N(z) and N_asym(z) on a fine z grid."""
    y = np.logspace(-2.3, 1.7, 300)
    zg = np.geomspace(z_init, z_end, 1500)
    F = np.zeros((len(y), len(zg)))
    for i, yi in enumerate(y):
        s = solve_ivp(lambda z, f: [z * z * K / np.sqrt(yi * yi + z * z) * (np.exp(-np.sqrt(yi * yi + z * z)) - f[0])],
                      (z_init, z_end), [0.0], method='Radau', rtol=1e-10, atol=1e-40, dense_output=True)
        F[i] = s.sol(zg)[0]
    E = np.sqrt(y[:, None]**2 + zg[None, :]**2)
    NN = np.trapezoid(y[:, None]**2 * F, y, axis=0) / (2 * zeta(3))                            # g_N = 2 over 2 zeta(3)
    net = np.trapezoid(y[:, None]**2 * (zg * zg * K / E) * (F - np.exp(-E)), y, axis=0) / (2 * zeta(3))
    net_s = CubicSpline(zg, net)
    sL = solve_ivp(lambda z, u: [-K * z**3 * kv(1, z) / 4.0 * u[0] + EPS * net_s(z)], (z_init, z_end), [0.0],
                   method='Radau', rtol=1e-10, atol=1e-30, dense_output=True)
    return zg, NN, sL.sol(zg)[0]


rows = []
def check(label, ok, detail):
    rows.append(ok); print(f"  {'PASS' if ok else 'FAIL'}  {label:50s} {detail}", flush=True)

for K in K_LIST:
    z_stop = CASES.get(K, 20.0)
    with contextlib.redirect_stdout(io.StringIO()):
        zs, NN, NL, rank = run_besthep(K, z_stop)
    if rank != 0:
        continue
    zg, NN_ex, NL_ex = exact_modes(K, zs[-1])
    print(f"\nK = {K}: {len(zs) - 1} steps to z = {zs[-1]:.2f}")
    for zt in (1.0, 3.0, 10.0):
        if zt > zs[-1]:
            continue
        r = np.interp(zt, zs, NN) / np.interp(zt, zg, NN_ex)
        check(f"N_N at z = {zt:g}, besthep / exact modes", abs(r - 1) < TOL, f"ratio {r:.4f}")
    r = NL[-1] / NL_ex[-1]
    check(f"kappa at z = {zs[-1]:.1f}, besthep / exact modes", abs(r - 1) < TOL,
          f"ratio {r:.4f}   (kappa = {4 / 3 * NL[-1] / EPS:+.4e}, exact {4 / 3 * NL_ex[-1] / EPS:+.4e})")

if rows:
    print(f"\n{sum(rows)}/{len(rows)} checks passed")
    sys.exit(0 if all(rows) else 1)
