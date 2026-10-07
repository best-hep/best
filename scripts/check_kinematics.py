#!/usr/bin/env python3
"""Regression check of the collision integrand's phase-space normalization.

With Maxwell-Boltzmann statistics, massless legs and a constant |M|^2 the n-body
phase space is closed-form (Phi_n(s) = s^(n-2) / (2 (4 pi)^(2n-3) (n-1)! (n-2)!)),
so the loss (or gain) rate at a given momentum is a one- or two-dimensional
quadrature. The integrand of besthep is integrated here by plain Monte Carlo
over its own domain (no Vegas, no MPI: one process) and compared with it:

  2->2 with massive legs, 2->2 below threshold, 1->2 and 1->3 (parent and daughter slot),
  3->2 (input and output slot), 2->3 and 2->4 (input slot), and the per-sample
  detailed balance BW/FW = 1 of a 2->3 process with mixed Bose/Fermi species at
  equilibrium.

Every ratio should be 1 within its Monte Carlo error. Run: python scripts/check_kinematics.py
"""
import os, sys
import numpy as np
from scipy.integrate import dblquad, quad
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST

T = 1.0; Q_MAX = 30.0; P_CUT = 45.0          # bath temperature, grid top, sampled-momentum top (1.5 q_max)
rng = np.random.default_rng(5)

def build(ins, outs, species, sym=1.0, M2=1.0):
    """species: {name: (mass, stat, 'thermal' | 'one')}; 'one' sets f = 1 (pure phase space)."""
    sol = BEST(q_min=0.05, q_max=Q_MAX, n_grid=60); sol.verbose = False
    for name, (m, st, kind) in species.items():
        if kind == 'one':
            sol.initialize_species(name, lambda q: 1.0, stat=st, mass=m, dof=1)
        else:
            eta = {'boson': 1.0, 'fermion': -1.0}.get(st, 0.0)
            sol.initialize_species(name, lambda q, m=m, eta=eta: 1.0 / (np.exp(np.sqrt(q * q + m * m) / T) - eta),
                                   stat=st, mass=m, dof=1)
    sol.add_process('p', ins, outs, lambda mom: np.full(mom.shape[2], M2), symmetry_factor=sym)
    return sol

def integrand(sol, target, p1, side):
    sol._force_target_side = side                # the slot of the target species (as the solver does per slot)
    f = sol.collision_integrand_batch('p', target, p1, mode='joint', t=0.0)
    sol._force_target_side = None
    return f

def domain(dims):
    """Uniform box of the integrand's variables: (q, theta) for the first free leg,
    (q, theta, phi) for every further one, then the pair's two angular fractions."""
    if dims == 1:
        return np.array([0.0]), np.array([1.0])
    lo, hi = [0.05, 0.0], [P_CUT, np.pi]
    for _ in range(1, 1 + (dims - 4) // 3):
        lo += [0.05, 0.0, 0.0]; hi += [P_CUT, np.pi, 2 * np.pi]
    lo += [0.0, 0.0]; hi += [1.0, 1.0]
    return np.array(lo), np.array(hi)

def mc(sol, target, p1, dims, side, reps=8, n=200000, importance=False):
    """Monte Carlo of the loss (FW) column. importance=True draws the free legs'
    momenta from e^{-q/2} (needed for 2->4, where the uniform box is too sparse)."""
    f = integrand(sol, target, p1, side); lo, hi = domain(dims)
    acc = acc2 = 0.0; N = 0
    for _ in range(reps):
        x = lo + rng.random((n, dims)) * (hi - lo); w = np.full(n, np.prod(hi - lo))
        if importance:
            cols = [0] + [2 + 3 * k for k in range((dims - 4) // 3)]
            for c in cols:
                q = 0.05 + rng.exponential(2.0, n); x[:, c] = q
                w *= (2.0 * np.exp((q - 0.05) / 2.0)) / (hi[c] - lo[c])
        o = f(x)[:, 1] * w
        acc += o.sum(); acc2 += (o * o).sum(); N += n
    return abs(acc / N), np.sqrt(max(acc2 / N - (acc / N) ** 2, 0.0) / N)

# ---- closed forms ----------------------------------------------------------
def Phi2(s, m1, m2):
    if s <= (m1 + m2) ** 2: return 0.0
    pst = np.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * np.sqrt(s)); return pst / (4 * np.pi * np.sqrt(s))
def Phi_massless(n):
    from math import factorial
    return lambda s: s ** (n - 2) / (2 * (4 * np.pi) ** (2 * n - 3) * factorial(n - 1) * factorial(n - 2))
def dPi_thermal(m):                      # int d^3p/((2pi)^3 2E) e^{-E/T}
    return quad(lambda p: 4 * np.pi * p * p / ((2 * np.pi) ** 3 * 2 * np.sqrt(p * p + m * m)) * np.exp(-np.sqrt(p * p + m * m) / T), 0, P_CUT)[0]
def partner_integral(p1, m1, m2, Phi):   # int dPi_2 e^{-E2/T} Phi(s),  s = (p1 + p2)^2
    E1 = np.sqrt(p1 * p1 + m1 * m1)
    g = lambda c, p2: p2 * p2 * 2 * np.pi / ((2 * np.pi) ** 3 * 2 * np.sqrt(p2 * p2 + m2 * m2)) * np.exp(-np.sqrt(p2 * p2 + m2 * m2) / T) \
        * Phi(m1 * m1 + m2 * m2 + 2 * (E1 * np.sqrt(p2 * p2 + m2 * m2) - p1 * p2 * c))
    return dblquad(g, 0.0, P_CUT, lambda p2: -1.0, lambda p2: 1.0)[0]
def loss_2in(p1, m1, m2, Phi):           # loss of the target: f1/(2E1) int dPi_2 f2 Phi(s), |M|^2 = sym = 1
    E1 = np.sqrt(p1 * p1 + m1 * m1); return np.exp(-E1 / T) / (2 * E1) * partner_integral(p1, m1, m2, Phi)

rows = []
def check(label, got, err, ref):
    r = got / ref; ok = abs(r - 1.0) <= max(3 * err / ref, 0.02)
    rows.append(ok); print(f"  {'PASS' if ok else 'FAIL'}  {label:46s} ratio {r:.4f}  (MC error {err / ref:.3f})")

print("Phase-space normalization of the collision integrand (ratio to the closed form)")
# A. massive 2->2
sol = build(['A', 'C'], ['B', 'B'], {'A': (1.0, 'mb', 'thermal'), 'C': (0.3, 'mb', 'thermal'), 'B': (0.5, 'mb', 'one')})
for p1 in (0.3, 2.0):
    m, e = mc(sol, 'A', p1, 4, 'input'); check(f"2->2 massive legs, loss, p = {p1}", m, e, loss_2in(p1, 1.0, 0.3, lambda s: Phi2(s, 0.5, 0.5)))
# B. threshold
sol = build(['A', 'A'], ['B', 'B'], {'A': (1.0, 'mb', 'thermal'), 'B': (1.6, 'mb', 'one')})
for p1 in (0.5, 3.0):
    m, e = mc(sol, 'A', p1, 4, 'input', reps=10); check(f"2->2 near threshold, loss, p = {p1}", m, e, loss_2in(p1, 1.0, 1.0, lambda s: Phi2(s, 1.6, 1.6)))
# C0. 1->2 (the three-leg path): parent loss, daughter gain (massless daughters, m_parent = 1)
sol = build(['A'], ['B', 'B'], {'A': (1.0, 'mb', 'thermal'), 'B': (0.0, 'mb', 'one')})
for p1 in (0.5, 3.0):
    m, e = mc(sol, 'A', p1, 1, 'input'); E1 = np.sqrt(p1 * p1 + 1.0)
    check(f"1->2 parent loss, p = {p1}", m, e, np.exp(-E1 / T) / (2 * E1) * Phi2(1.0, 0.0, 0.0))
for p1 in (0.3, 1.0):
    m, e = mc(sol, 'B', p1, 1, 'output')
    # (p0 - p1)^2 = 0 fixes cos theta: int dc delta(1 - 2 E0 p1 + 2 p0 p1 c) = 1/(2 p0 p1) where |c| <= 1
    def g(p0):
        E0 = np.sqrt(p0 * p0 + 1.0); c = (E0 * p1 - 0.5) / (p0 * p1)
        return 0.0 if abs(c) > 1.0 else p0 * p0 * 2 * np.pi / ((2 * np.pi) ** 3 * 2 * E0) * np.exp(-E0 / T) * 2 * np.pi / (2 * p0 * p1)
    check(f"1->2 daughter gain, p = {p1}", m, e, quad(g, 1e-6, P_CUT, limit=400)[0] / (2 * p1))
# C. 1->3: parent loss, daughter gain
sol = build(['A'], ['B', 'B', 'B'], {'A': (1.0, 'mb', 'thermal'), 'B': (0.0, 'mb', 'one')})
for p1 in (0.5, 2.0):
    m, e = mc(sol, 'A', p1, 4, 'input'); E1 = np.sqrt(p1 * p1 + 1.0)
    check(f"1->3 parent loss, p = {p1}", m, e, np.exp(-E1 / T) / (2 * E1) * Phi_massless(3)(1.0))
for p1 in (0.3, 0.6):
    m, e = mc(sol, 'B', p1, 4, 'output', reps=10)
    g = lambda c, p0: p0 * p0 * 2 * np.pi / ((2 * np.pi) ** 3 * 2 * np.sqrt(p0 * p0 + 1)) * np.exp(-np.sqrt(p0 * p0 + 1) / T) \
        * Phi2(1.0 - 2 * (np.sqrt(p0 * p0 + 1) * p1 - p0 * p1 * c), 0.0, 0.0)
    check(f"1->3 daughter gain, p = {p1}", m, e, dblquad(g, 0.0, P_CUT, lambda p0: -1.0, lambda p0: 1.0)[0] / (2 * p1))
# D. 3->2: input loss, output gain
sol = build(['A', 'A', 'A'], ['B', 'B'], {'A': (0.0, 'mb', 'thermal'), 'B': (0.0, 'mb', 'one')})
for p1 in (0.5, 2.0):
    m, e = mc(sol, 'A', p1, 7, 'input'); check(f"3->2 input loss, p = {p1}", m, e, np.exp(-p1 / T) / (2 * p1) * dPi_thermal(0.0) ** 2 / (8 * np.pi))
    m, e = mc(sol, 'B', p1, 7, 'output'); check(f"3->2 output gain, p = {p1}", m, e, np.exp(-p1 / T) / (2 * p1) * partner_integral(p1, 0.0, 0.0, Phi_massless(3)))
# E. 2->3 and 2->4 loss (massless)
for nout, dims in ((3, 7), (4, 10)):
    sol = build(['A', 'A'], ['B'] * nout, {'A': (0.0, 'mb', 'thermal'), 'B': (0.0, 'mb', 'one')})
    for p1 in (1.0, 3.0):
        m, e = mc(sol, 'A', p1, dims, 'input', reps=20, importance=(nout == 4))
        check(f"2->{nout} input loss, p = {p1}", m, e, loss_2in(p1, 0.0, 0.0, Phi_massless(nout)))
# F. detailed balance per sample, mixed statistics
print("Detailed balance at equilibrium, 2->3 with Bose and Fermi species (BW/FW per sample)")
sol = build(['A', 'C'], ['A', 'D', 'C'], {'A': (0.0, 'boson', 'thermal'), 'C': (0.3, 'fermion', 'thermal'), 'D': (0.0, 'boson', 'thermal')})
for tgt, side in (('A', 'input'), ('C', 'output'), ('D', 'output')):
    f = integrand(sol, tgt, 1.3, side); lo, hi = domain(7)
    x = lo + rng.random((300000, 7)) * (hi - lo); o = f(x); ok = (o[:, 1] != 0) & (o[:, 2] != 0); r = o[ok, 2] / o[ok, 1]
    good = abs(np.median(r) - 1.0) < 1e-3
    rows.append(good); print(f"  {'PASS' if good else 'FAIL'}  target {tgt} ({side}): median {np.median(r):.6f}, spread {np.std(np.log(r)):.1e}")
print(f"\n{sum(rows)}/{len(rows)} checks passed")
sys.exit(0 if all(rows) else 1)