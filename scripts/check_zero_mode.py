"""Checks of the zero-mode terms of a 2 -> 2 process (no stepping, one rank).
1. Detailed balance: gain = loss node by node on a Bose-Einstein distribution with the zero mode at
   its own equilibrium occupation N0 = V0 f_BE(p = 0), for mu = m and below; an empty zero mode fills.
2. Energy: int d^3p (E - m) C = 0 (the zero mode carries m per particle), and the number rate.
3. Normalization against BEST's Vegas 2->2 rate with a momentum-dependent |M|^2: a thin low-q
   bump of number n_b stands in for the zero mode; at hard nodes C_el[h + bump] - C_el[h] equals the
   zero-mode term with N0 = n_b up to O(q_bump / p) and O(h).
4. The same for unequal masses, phi chi <-> phi chi: both targets, the zero-mode leg on either side.
Usage: python3 scripts/check_zero_mode.py [neval]"""
import os, sys, time, io, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST
m = 1.0
quiet = io.StringIO()


def solver(f, n_grid, q_min=0.01, neval=1000, nitn=2, M2=None):
    with contextlib.redirect_stdout(quiet):
        s = BEST(q_min=q_min, q_max=50.0, n_grid=n_grid)
        s.initialize_species('phi', f, stat='boson', mass=m)
        s.add_process('el', ['phi', 'phi'], ['phi', 'phi'], M2 or (lambda mom: np.ones(mom.shape[2])), neval=neval, nitn=nitn)
        s.bw_fallback_rel_err = float('inf')
    return s


def zero_mode_rates(s, N0):
    s.zero_mode['phi'] = N0
    terms = s._zero_mode_rates('el', 'phi', 'input', s.current_time)
    return sum(t[5] for t in terms), sum(t[6] for t in terms)


be = lambda T, mu: (lambda q: 1.0 / np.expm1((np.sqrt(np.asarray(q, float)**2 + m * m) - mu) / T))
V0 = 0.01**3 / (6 * np.pi**2)
print("1. detailed balance: max |gain - loss| / gain on BE; the zero mode at its BE occupation N0 = V0 f_BE(0)")
for T, mu in ((3.0, 1.0), (1.0, 1.0), (3.0, 0.9), (3.0, 0.5), (10.0, 0.0)):
    s = solver(be(T, mu), 200)
    N0 = V0 / np.expm1((m - mu) / T) if mu < m else 1.0
    g, l = zero_mode_rates(s, N0)
    q = s.r_grids['phi']
    print(f"   T = {T:5.2f} mu = {mu:4.2f} N0 = {N0:.2e}: {np.max(np.abs(g - l) / g):.1e}")
print("   the same with the zero mode empty (N0 = 0, spontaneous term only): dn_c/dt / V0 and sign")
for T, mu in ((3.0, 0.9), (3.0, 1.0)):
    s = solver(be(T, mu), 200)
    g, l = zero_mode_rates(s, 0.0)
    q = s.r_grids['phi']
    print(f"   T = {T:5.2f} mu = {mu:4.2f}: dN0/dt / V0 = {-np.trapezoid(q * q * (g - l), q) / (2 * np.pi**2) / V0:+.3e}")

print("2. f = 2/(1 + exp((p - 3)/2)): int (E - m) C / int (E - m) gain (energy), int C / int gain (number)")
f = lambda q: 2.0 / (1.0 + np.exp((np.asarray(q, float) - 3.0) / 2.0))
for n_grid in (100, 200, 400):
    s = solver(f, n_grid)
    g, l = zero_mode_rates(s, 1.0)
    q = s.r_grids['phi']; K = q * q / (np.sqrt(q * q + m * m) + m)
    print(f"   n_grid {n_grid}: {np.trapezoid(q*q*K*(g-l), q)/np.trapezoid(q*q*K*g, q):+.1e}   {np.trapezoid(q*q*(g-l), q)/np.trapezoid(q*q*g, q):+.3e}")

print("3. normalization: Vegas C_el[h + bump] - C_el[h] vs the zero-mode term with N0 = n_bump, momentum-dependent |M|^2")
def mesq(mom):
    E = np.sqrt(np.sum(mom**2, axis=1) + m * m)
    s_ = (E[0] + E[1])**2 - np.sum((mom[0] + mom[1])**2, axis=0)
    cos13 = np.sum(mom[0] * mom[2], axis=0) / np.maximum(np.sqrt(np.sum(mom[0]**2, axis=0) * np.sum(mom[2]**2, axis=0)), 1e-300)
    return 1.0 + 0.3 * (s_ - 4 * m * m) / m**2 + 0.2 * cos13
h = lambda q: 1e-4 * np.exp(-(np.sqrt(np.asarray(q, float)**2 + m * m) - m) / 3.0)
bump = lambda q: 1e3 * np.exp(-0.5 * ((np.asarray(q, float) - 0.1) / 0.02)**2)
neval = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100000
sb, sh = solver(lambda q: h(q) + bump(q), 160, 0.005, neval, 6, mesq), solver(h, 160, 0.005, neval, 6, mesq)
qq = np.linspace(0.005, 0.5, 20001)
n_b = np.trapezoid(qq * qq * (sb.interpolators['phi'](qq) - sh.interpolators['phi'](qq)), qq) / (2.0 * np.pi**2)
t0 = time.time()
with contextlib.redirect_stdout(quiet):
    C = sb.compute_rates(['el'])['phi'] - sh.compute_rates(['el'])['phi']
g, l = zero_mode_rates(sh, n_b)
q = sh.r_grids['phi']
print(f"   Vegas {time.time() - t0:.0f} s; n_b = {n_b:.4e}")
for qi in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
    j = int(np.argmin(np.abs(q - qi)))
    print(f"   q = {q[j]:6.2f}: {C[j]:12.4e} {g[j] - l[j]:12.4e}  ratio {C[j] / (g[j] - l[j]):6.3f}")

print("4. the same for unequal masses, phi chi <-> phi chi (m_chi = 0.3): bump in chi, both targets, every zero-mode leg")
m_chi = 0.3
def mesq2(mom):
    E0 = np.sqrt(np.sum(mom[0]**2, axis=0) + m * m); E1 = np.sqrt(np.sum(mom[1]**2, axis=0) + m_chi**2)
    s_ = (E0 + E1)**2 - np.sum((mom[0] + mom[1])**2, axis=0)
    cos13 = np.sum(mom[0] * mom[2], axis=0) / np.maximum(np.sqrt(np.sum(mom[0]**2, axis=0) * np.sum(mom[2]**2, axis=0)), 1e-300)
    return 1.0 + 0.3 * (s_ - (m + m_chi)**2) + 0.2 * cos13
h_chi = lambda q: 1e-4 * np.exp(-(np.sqrt(np.asarray(q, float)**2 + m_chi**2) - m_chi) / 2.0)
bump2 = lambda q: 1e3 * np.exp(-0.5 * ((np.asarray(q, float) - 0.02) / 0.004)**2)
def solver2(f_chi):
    with contextlib.redirect_stdout(quiet):
        s = BEST(q_min=0.002, q_max=50.0, n_grid=160)
        s.initialize_species('phi', lambda q: 1e-3 * np.exp(-(np.sqrt(np.asarray(q, float)**2 + m * m) - m) / 2.0), stat='boson', mass=m)
        s.initialize_species('chi', f_chi, stat='boson', mass=m_chi)
        s.add_process('sc', ['phi', 'chi'], ['phi', 'chi'], mesq2, neval=neval, nitn=6)
        s.bw_fallback_rel_err = float('inf')
    return s
sb, sh = solver2(lambda q: h_chi(q) + bump2(q)), solver2(h_chi)
qq = np.linspace(0.002, 0.5, 40001)
n_b = np.trapezoid(qq * qq * (sb.interpolators['chi'](qq) - sh.interpolators['chi'](qq)), qq) / (2.0 * np.pi**2)
t0 = time.time()
with contextlib.redirect_stdout(quiet):
    Cb, Ch = sb.compute_rates(['sc']), sh.compute_rates(['sc'])
print(f"   Vegas {time.time() - t0:.0f} s; n_b = {n_b:.4e}")
sh.zero_mode['chi'] = n_b
for target in ('phi', 'chi'):
    terms = sh._zero_mode_rates('sc', target, 'input', 0.0)
    q = sh.r_grids[target]
    for qi in (0.5, 1.0, 2.0, 5.0, 10.0):
        j = int(np.argmin(np.abs(q - qi)))
        C, T_ = Cb[target][j] - Ch[target][j], sum(t[5][j] - t[6][j] for t in terms)
        print(f"   {target} q = {q[j]:6.2f}: {C:12.4e} {T_:12.4e}  ratio {C / T_:6.3f}")
