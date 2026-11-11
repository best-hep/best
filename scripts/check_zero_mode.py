"""Checks of the zero-mode terms (no stepping, one rank).
1. Detailed balance, phi phi <-> phi phi with one leg at rest: gain = loss node by node on a Bose-Einstein
   distribution with the zero mode at its own equilibrium occupation N0 = V0 f_BE(p = 0), for mu = m and below;
   an empty zero mode fills.
2. Energy: int d^3p (E - m) C = 0 (the zero mode carries m per particle), and the number rate.
3. Normalization against BEST's Vegas 2->2 rate with a momentum-dependent |M|^2: a thin low-q
   bump of number n_b stands in for the zero mode; at hard nodes C_el[h + bump] - C_el[h] equals the
   zero-mode term with N0 = n_b up to O(q_bump / p) and O(h).
4. The same for unequal masses, phi chi <-> phi chi: both targets, the leg at rest on either side.
5. Several legs at rest, closed forms: phi phi -> chi chi with both phi at rest, dN0/dt = -N0^2 sigma v / a^3
   with sigma v = |M|^2 p_f / (64 pi m^3), and phi phi <-> phi phi phi with the three at rest,
   dN0/dt = -3 N0^3 |M|^2 p_f / (1152 pi m^4); the lines carry the number and the energy.
6. A moving parent with one daughter at rest, phi <-> chi chi, against the Vegas rate with the bump of 3:
   the integrated change of the phi rate against the term.
7. Detailed balance of every class of terms of a list of processes, the zero modes at V0 f_BE(0).
8. Four or more legs in motion, the Monte Carlo with the legs at rest frozen: against the quadrature on parts with
   three legs in motion, where both exist; and for chi chi <-> chi chi phi with phi at rest, which changes no grid
   and is counted on its incoming legs, the Vegas rate of everything below 1.6 q_bump (the bump as the observed
   leg) against dN0/dt of the terms plus the same nodes.
Usage: python3 scripts/check_zero_mode.py [neval]"""
import os, sys, time, io, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST, grid_weights
m = 1.0
quiet = io.StringIO()
const = lambda c: (lambda mom: np.full(mom.shape[2], c))


def solver(f, n_grid, q_min=0.01, neval=1000, nitn=2, M2=None):
    with contextlib.redirect_stdout(quiet):
        s = BEST(q_min=q_min, q_max=50.0, n_grid=n_grid)
        s.initialize_species('phi', f, stat='boson', mass=m)
        s.add_process('el', ['phi', 'phi'], ['phi', 'phi'], M2 or const(1.0), neval=neval, nitn=nitn)
        s.bw_fallback_rel_err = float('inf')
    return s


def zero_mode_rates(s, N0):
    """Gain and loss of phi phi <-> phi phi with one leg at rest: the other incoming leg, or either outgoing one."""
    s.zero_mode['phi'] = N0
    g_in, l_in = s._zero_mode_rates('el', 0, [1], s.current_time)[:2]
    g_out, l_out = s._zero_mode_rates('el', 0, [2], s.current_time)[:2]
    return g_in + 2.0 * g_out, l_in + 2.0 * l_out


def number(s, sp, C, E=None):
    q = s.r_grids[sp]
    return (grid_weights(q) * q * q * s.species_dof.get(sp, 1) / (2.0 * np.pi**2)) @ (C if E is None else C * E)


be = lambda T, mu, mass=m: (lambda q: 1.0 / np.expm1((np.sqrt(np.asarray(q, float)**2 + mass * mass) - mu) / T))
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

print("4. the same for unequal masses, phi chi <-> phi chi (m_chi = 0.3): bump in chi, both targets, every leg at rest")
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
for tgt, target in enumerate(('phi', 'chi')):
    terms = [sh._zero_mode_rates('sc', tgt, [k], 0.0) for k in range(4) if k != tgt]
    q = sh.r_grids[target]
    for qi in (0.5, 1.0, 2.0, 5.0, 10.0):
        j = int(np.argmin(np.abs(q - qi)))
        C, T_ = Cb[target][j] - Ch[target][j], sum(t[0][j] - t[1][j] for t in terms)
        print(f"   {target} q = {q[j]:6.2f}: {C:12.4e} {T_:12.4e}  ratio {C / T_:6.3f}")

print("5. several legs at rest: dN0/dt against the closed form; number and energy the grid receives per particle the zero mode gives")
dilute = lambda q: 1e-14 * np.exp(-np.asarray(q, float))
def rates(s, name):          # dN0/dt of the zero modes, and the number and comoving energy the zero-mode terms give each grid
    with contextlib.redirect_stdout(quiet):
        s.compute_rates([name])
    f = {sp: s.distributions_1d[sp] for sp in s.species_list}
    a = s.scale_factor(s.current_time)
    num, en = {}, {}
    for _, sp, comp, gain, loss, dnet, J, side in s._zero_mode_terms:
        num[sp] = num.get(sp, 0.0) + number(s, sp, gain - loss)
        en[sp] = en.get(sp, 0.0) + number(s, sp, gain - loss, np.sqrt(s.r_grids[sp]**2 + (a * s.species_mass[sp])**2))
    return s._zero_mode_step(f, f, 1.0, apply=False), num, en
for a in (1.0, 2.5):
    with contextlib.redirect_stdout(quiet):
        s = BEST(q_min=0.01, q_max=20.0, n_grid=80)
        s.scale_factor = lambda t, a=a: a
        s.initialize_species('phi', lambda q: 0.0 * q, stat='boson', mass=m, n0=0.01)
        s.initialize_species('chi', dilute, stat='boson', mass=m_chi)
        s.add_process('ann', ['phi', 'phi'], ['chi', 'chi'], const(1.0), neval=1000, nitn=2)
    N0 = s.zero_mode['phi']
    dN, num, en = rates(s, 'ann')
    exact = -N0**2 * np.sqrt(m * m - m_chi**2) / (64 * np.pi * m**3) / a**3
    print(f"   phi phi -> chi chi, a = {a}: dN0/dt = {dN['phi']:+.6e} (ratio {dN['phi'] / exact:.6f}); chi number {num['chi'] / -dN['phi']:.6f}, energy / (a m_phi) {en['chi'] / (-a * m * dN['phi']):.6f}")
with contextlib.redirect_stdout(quiet):
    s = BEST(q_min=0.01, q_max=20.0, n_grid=80)
    s.initialize_species('phi', dilute, stat='boson', mass=m, n0=0.01)
    s.add_process('can', ['phi', 'phi'], ['phi', 'phi', 'phi'], const(1.0), neval=1000, nitn=2)
N0 = s.zero_mode['phi']
dN, num, en = rates(s, 'can')
exact = -3.0 * N0**3 * np.sqrt(1.25) * m / (1152 * np.pi * m**4)
print(f"   phi phi phi -> phi phi, the three at rest: dN0/dt = {dN['phi']:+.6e} (ratio {dN['phi'] / exact:.6f}); grid number / (2/3) {num['phi'] / (-2.0 / 3.0 * dN['phi']):.6f}, energy / m {en['phi'] / (-m * dN['phi']):.6f}")

print("6. phi <-> chi chi (m_phi = 4, m_chi = 1), a moving phi with one chi at rest: Vegas int (C_phi[h + bump] - C_phi[h]) vs the term with N0 = n_bump")
h_dec = lambda q: 1e-3 * np.exp(-(np.sqrt(np.asarray(q, float)**2 + 1.0) - 1.0) / 2.0)
for q_b in (0.1, 0.05):
    bump3 = lambda q, q_b=q_b: 1e3 * np.exp(-0.5 * ((np.asarray(q, float) - q_b) / (0.2 * q_b))**2)
    def solver3(f_chi):
        with contextlib.redirect_stdout(quiet):
            s = BEST(q_min=0.005, q_max=50.0, n_grid=160)
            s.initialize_species('phi', be(2.0, 1.0, 4.0), stat='boson', mass=4.0)
            s.initialize_species('chi', f_chi, stat='boson', mass=1.0)
            s.add_process('dec', ['phi'], ['chi', 'chi'], const(3.0), neval=neval, nitn=6)
            s.bw_fallback_rel_err = float('inf')
            s.zero_mode['phi'] = s.zero_mode['chi'] = 0.0
        return s
    sb, sh = solver3(lambda q: h_dec(q) + bump3(q)), solver3(h_dec)
    t0 = time.time()
    with contextlib.redirect_stdout(quiet):
        Cb, Ch = sb.compute_rates(['dec'])['phi'], sh.compute_rates(['dec'])['phi']
        Cb, Ch = (C - sum(t[3] - t[4] for t in s._zero_mode_terms if t[1] == 'phi') for C, s in ((Cb, sb), (Ch, sh)))   # without the spontaneous parts
    n_b = number(sb, 'chi', sb.distributions_1d['chi'] - sh.distributions_1d['chi'])
    sh.zero_mode['chi'] = n_b
    g, l = sh._zero_mode_line('dec', 0, [1], 0.0)[:2]
    print(f"   q_bump = {q_b}: Vegas {time.time() - t0:.0f} s; n_b = {n_b:.4e}; {number(sh, 'phi', Cb - Ch):+.4e} {number(sh, 'phi', 2.0 * (g - l)):+.4e}  ratio {number(sh, 'phi', Cb - Ch) / number(sh, 'phi', 2.0 * (g - l)):6.3f}")

print("7. detailed balance of every class of terms (the legs at rest: species, on the input side, on the output side): max |gain - loss| / max gain")
for tag, species, process, mu in (("phi phi <-> chi chi", {'phi': 1.0, 'chi': 0.3}, (['phi', 'phi'], ['chi', 'chi']), {'phi': 0.2, 'chi': 0.2}),
                                  ("phi <-> chi chi", {'phi': 4.0, 'chi': 1.0}, (['phi'], ['chi', 'chi']), {'phi': 1.0, 'chi': 0.5}),
                                  ("phi <-> chi chi chi", {'phi': 4.0, 'chi': 1.0}, (['phi'], ['chi', 'chi', 'chi']), {'phi': 0.9, 'chi': 0.3}),
                                  ("phi phi <-> chi chi chi", {'phi': 1.0, 'chi': 0.4}, (['phi', 'phi'], ['chi', 'chi', 'chi']), {'phi': 0.3, 'chi': 0.2}),
                                  ("phi phi <-> phi phi phi", {'phi': 1.0}, (['phi', 'phi'], ['phi', 'phi', 'phi']), {'phi': 0.0})):
    with contextlib.redirect_stdout(quiet):
        s = BEST(q_min=0.02, q_max=40.0, n_grid=120)
        for sp, mass in species.items():
            s.initialize_species(sp, be(1.5, mu[sp], mass), stat='boson', mass=mass)
        s.add_process('p', *process, const(2.0), neval=1000, nitn=2)
        s.bw_fallback_rel_err = float('inf')
        for sp, mass in species.items():
            s.zero_mode[sp] = 0.02**3 / (6 * np.pi**2) / np.expm1((mass - mu[sp]) / 1.5)
        s.compute_rates(['p'])
    for comp in dict.fromkeys(t[2] for t in s._zero_mode_terms):
        terms = [t for t in s._zero_mode_terms if t[2] == comp and np.max(t[3]) > 0.0]
        if terms:
            print(f"   {tag}: {', '.join(f'{c} {k_in} {k_out}' for c, k_in, k_out in comp)}: {max(np.max(np.abs(t[3] - t[4])) / np.max(t[3]) for t in terms):.1e}")

print("8. the Monte Carlo terms. Against the quadrature (three legs in motion): gain, median and range over the nodes")
def mesq3(masses):
    def f(mom):
        E = [np.sqrt(np.sum(mom[i]**2, axis=0) + masses[i]**2) for i in (0, 1)]
        s_ = (E[0] + E[1])**2 - np.sum((mom[0] + mom[1])**2, axis=0)
        cos = np.sum(mom[0] * mom[2], axis=0) / np.maximum(np.sqrt(np.sum(mom[0]**2, axis=0) * np.sum(mom[2]**2, axis=0)), 1e-300)
        return 1.0 + 0.2 * s_ / (masses[0] + masses[1])**2 + 0.2 * cos
    return f
for tag, species, process, tgt, rest in (("phi chi <-> phi chi, chi at rest", {'phi': (1.0, 0.3), 'chi': (0.3, 0.1)}, (['phi', 'chi'], ['phi', 'chi']), 0, [1]),
                                         ("phi phi <-> chi chi chi, both phi at rest", {'phi': (1.0, 0.3), 'chi': (0.4, 0.2)}, (['phi', 'phi'], ['chi', 'chi', 'chi']), 2, [0, 1]),
                                         ("phi <-> chi chi chi, one chi at rest", {'phi': (4.0, 0.6), 'chi': (1.0, 0.2)}, (['phi'], ['chi', 'chi', 'chi']), 0, [1])):
    with contextlib.redirect_stdout(quiet):
        s = BEST(q_min=0.02, q_max=20.0, n_grid=40)
        for sp, (mass, mu) in species.items():
            s.initialize_species(sp, be(2.0, mu, mass), stat='boson', mass=mass)
        s.add_process('p', *process, mesq3([species[l][0] for l in process[0] + process[1]]), neval=neval, nitn=4)
        s.bw_fallback_rel_err = float('inf')
        for sp in species:
            s.zero_mode[sp] = 0.05
        mc, gl = s._zero_mode_sampled('p', tgt, rest, 0.0), s._zero_mode_rates('p', tgt, rest, 0.0)
    r = (mc[0] / np.where(gl[0] > 0, gl[0], 1.0))[gl[0] > 1e-3 * gl[0].max()]
    print(f"   {tag}: {np.median(r):.4f}, {r.min():.3f} - {r.max():.3f}")
print("   chi chi <-> chi chi phi (m_chi = 1, m_phi = 0.5), phi at rest, against Vegas with a bump in phi")
for q_b in (0.1, 0.05):
    bump4 = lambda q, q_b=q_b: 30.0 * np.exp(-0.5 * ((np.asarray(q, float) - q_b) / (0.2 * q_b))**2)
    h_phi = lambda q: 1e-3 * np.exp(-np.sqrt(np.asarray(q, float)**2 + 0.25) / 2.0)
    def solver4(f_phi):
        with contextlib.redirect_stdout(quiet):
            s = BEST(q_min=0.005, q_max=30.0, n_grid=120)
            s.initialize_species('chi', be(2.0, 0.5), stat='boson', mass=1.0)
            s.initialize_species('phi', f_phi, stat='boson', mass=0.5)
            s.add_process('brem', ['chi', 'chi'], ['chi', 'chi', 'phi'], const(50.0), neval=neval, nitn=4)
            s.bw_fallback_rel_err = float('inf')
            s.zero_mode['phi'] = s.zero_mode['chi'] = 0.0
        return s
    sb, sh = solver4(lambda q: h_phi(q) + bump4(q)), solver4(h_phi)
    t0 = time.time()
    with contextlib.redirect_stdout(quiet):
        Cb = sb.compute_rates(['brem'])['phi']
    low = sb.r_grids['phi'] < 1.6 * q_b
    n_b = number(sb, 'phi', sb.distributions_1d['phi'] - sh.distributions_1d['phi'])
    sh.zero_mode['phi'] = n_b
    with contextlib.redirect_stdout(quiet):
        Ch = sh.compute_rates(['brem'])['phi']
        f = {sp: sh.distributions_1d[sp] for sp in sh.species_list}
        dN = sh._zero_mode_step(f, f, 1.0, apply=False)['phi']
    V, T_ = number(sb, 'phi', np.where(low, Cb, 0.0)), dN + number(sh, 'phi', np.where(low, Ch, 0.0))
    print(f"   q_bump = {q_b}: Vegas {time.time() - t0:.0f} s; n_b = {n_b:.4e}; {V:+.4e} {T_:+.4e}  ratio {V / T_:6.3f}")
