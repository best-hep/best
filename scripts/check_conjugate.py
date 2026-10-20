"""
Check of the CP-conjugate pairs and the explicit reverse |M|^2 (besthep >= 1.3.1).

The system is the decay N <-> Phi l and its CP conjugate N <-> Phibar lbar with
|M(N -> Phi l)|^2 = (1 + eps) |M0|^2 and |M(N -> Phibar lbar)|^2 = (1 - eps) |M0|^2,
registered as a CP-conjugate pair ('decA', 'decB') in two forms:
  - the CPT form: each one's reverse |M|^2 is the other's forward one; the on-shell
    remainder of the RIS-subtracted Delta L = 2 scattering, which in the narrow-width
    limit cancels the CP-odd part of the inverse decays in the lepton equations
    (Hahn-Woernle, Plumacher & Wong 2009, section 3), is written as two unpaired
    processes with no forward direction and their own reverse |M|^2 ('risA':
    Phi l -> N with +2 eps |M0|^2, 'risB': Phibar lbar -> N with -2 eps |M0|^2);
  - the effective form: each member carries its own forward |M|^2 in both
    directions (mesq_backward = the forward function), still on shared samples.
In total every direction then carries the T-symmetric |M|^2 of the unpaired
processes 'tA' (N <-> Phi l, (1 + eps) |M0|^2) and 'tB' (N <-> Phibar lbar,
(1 - eps) |M0|^2), which serve as the reference. The species are declared with
their antiparticles (N and x self-conjugate).

Checked:
  1. the structure checks of add_process(..., cp_conjugate=...): a partner that is
     missing, the process itself, a reverse |M|^2 for one member only, different
     neval, one already paired, legs differing slot by slot, identical particles in
     different slots, a slot whose species is not the declared antiparticle, species
     not initialized, are rejected and nothing is registered; re-registering a member
     drops the old link; contradicting antiparticle declarations are rejected;
  2. rates, gain/loss, per-process ledgers and Jacobian of both paired forms equal
     those of the T-symmetric reference; a pair equals the same two processes
     written unpaired with explicit reverse |M|^2 at eps = 0.5, for 1 -> 2 and 2 -> 2
     legs, identical particles and the observed species on either side, with the CPT
     reverse and with four different functions (a forward/reverse swap in the partner
     wiring would be a factor 3);
  3. within one pass, the ratios of the paired rates are the ratios of the
     matrix elements to rounding (same samples): gain(Phi)/gain(Phibar) =
     (1+eps)/(1-eps) and the two decay channels of N likewise;
  4. at eps = 0 the paired species have equal rates to rounding although every
     integrator draws its own samples (the unpaired reference does not), also when
     the backward fallback integration is forced on every node and after an
     exprb_seq step (a pair is one substep);
  5. a run of the paired set with Phi, Phibar prescribed: the asymmetry
     f_l - f_lbar of the direct run against the linear asymmetry equation
     d delta/dt = S - W delta stepped with exprb2_update from the same run's
     coefficients; the difference is of the time discretization, so it falls
     faster than the step (second order: by four when the step is halved);
  6. a checkpoint restores the pair, the reverse |M|^2 and the antiparticle
     declarations, refuses to drop a reverse |M|^2 it cannot resolve and refuses to
     add one to a process saved without it;
  7. only one member of a pair active: the partner's species get nothing;
     compute_collision_rate with both members books each (species, process) once;
     a mass function given to one species of a pair only stops the rate pass;
  8. a reaction that is its own reverse (Phi l -> Phi l) is counted once whether its
     reverse |M|^2 is the default, given, or taken from a CP-conjugate partner: the
     loss rate of massless Maxwell-Boltzmann l with constant |M|^2 is
     |M|^2 /(64 pi^3 q g_l) int p e^{-p/T} dp over the sampled momenta in all three
     forms.

Criteria. Two integrations of the same quantity on different samples agree to
their Monte Carlo spread: the reference is run twice and the spread of the two
runs, times K_SIGMA, bounds the difference between a paired form and the
reference (the Jacobian is compared as J . f, since its rows are sums over the
samples near a node). The second run traverses the grid in the opposite
direction (the solver's alternating order), so that the two runs do not share
the adaptation history of their integrator maps, which otherwise correlates
their errors and understates the spread. A spread measured from two runs has one
degree of freedom, so a single line of section 2 can exceed K_SIGMA times it by
chance (up to about one in five when one node dominates both the difference and
the spread); such a FAIL is repeated before it is read as a defect. On common
samples the spread is zero
and the quantities must agree to the rounding of a sum over the samples
(ROUNDING). The solver combines the iterations of every integration with one
weight per iteration so that the columns stay on common samples; Vegas'
inverse-covariance weighting of the columns would not keep them there.

Usage: python3 scripts/check_conjugate.py
"""
import os, sys, io, contextlib, tempfile, functools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST, exprb2_update, assemble_A_Gamma

EPS = 1.0e-3                                   # CP asymmetry of the test model
T = 1.0
NEVAL, NITN = int(2e4), 2
K_SIGMA = 3                                    # statistical margin on the measured Monte Carlo spread
ROUNDING = np.finfo(float).eps * NEVAL * NITN  # relative rounding of a sum over the samples
W_DT = 0.05                                    # check 5: step in units of the wash-out time


def M0(m):
    """|M0|^2 with a mild dependence on the angle between the first and the last leg."""
    return 3.0 + 0.5 * np.sum(m[0] * m[-1], axis=0) / np.sqrt((1.0 + np.sum(m[0]**2, axis=0)) * (1.0 + np.sum(m[-1]**2, axis=0)))

def MA(m): return (1.0 + EPS) * M0(m)
def MB(m): return (1.0 - EPS) * M0(m)
def remA(m): return 2.0 * EPS * M0(m)
def remB(m): return -2.0 * EPS * M0(m)
def zero(m): return np.zeros(m.shape[2])
def scaled(m, c): return c * M0(m)


def species(sol):
    f_N = lambda q: 0.5 * np.exp(-np.sqrt(q * q + 4.0) / T)
    f_Phi = lambda q: 1.0 / np.expm1(np.sqrt(q * q + 0.09) / T)
    f_l = lambda q: 0.9 / (np.exp(np.sqrt(q * q + 0.04) / T) + 1.0)
    f_x = lambda q: 0.3 * np.exp(-np.sqrt(q * q + 0.25) / T)
    sol.initialize_species('N', f_N, stat='fermion', mass=2.0, dof=2, antiparticle='N')
    for s, anti in (('Phi', 'Phibar'), ('Phibar', 'Phi')):
        sol.initialize_species(s, f_Phi, stat='boson', mass=0.3, dof=2, antiparticle=anti)
    for s, anti in (('l', 'lbar'), ('lbar', 'l')):
        sol.initialize_species(s, f_l, stat='fermion', mass=0.2, dof=2, antiparticle=anti)
    sol.initialize_species('x', f_x, stat='boson', mass=0.5, dof=1, antiparticle='x')


def build(paired, eps=EPS, evolving=None, n_grid=24, effective=False):
    """paired: the CPT pair plus the remainder processes, or (effective=True) the pair
    with each member's own forward |M|^2 in both directions; else the T-symmetric
    reference tA, tB."""
    sol = BEST(q_min=0.02, q_max=20.0, n_grid=n_grid)
    sol.verbose = False
    species(sol)
    ma, mb = (MA, MB) if eps != 0.0 else (M0, M0)
    if paired and effective:
        sol.add_process('decA', ['N'], ['Phi', 'l'], ma, neval=NEVAL, nitn=NITN, mesq_backward=ma)
        sol.add_process('decB', ['N'], ['Phibar', 'lbar'], mb, neval=NEVAL, nitn=NITN, mesq_backward=mb, cp_conjugate='decA')
    elif paired:
        sol.add_process('decA', ['N'], ['Phi', 'l'], ma, neval=NEVAL, nitn=NITN)
        sol.add_process('decB', ['N'], ['Phibar', 'lbar'], mb, neval=NEVAL, nitn=NITN, cp_conjugate='decA')
        ra, rb = (remA, remB) if eps != 0.0 else (zero, zero)
        sol.add_process('risA', ['N'], ['Phi', 'l'], zero, neval=NEVAL, nitn=NITN, mesq_backward=ra)
        sol.add_process('risB', ['N'], ['Phibar', 'lbar'], zero, neval=NEVAL, nitn=NITN, mesq_backward=rb)
    else:
        sol.add_process('tA', ['N'], ['Phi', 'l'], ma, neval=NEVAL, nitn=NITN)
        sol.add_process('tB', ['N'], ['Phibar', 'lbar'], mb, neval=NEVAL, nitn=NITN)
    if evolving is not None:
        sol.species_list = list(evolving)
    return sol


rows = []
def check(label, ok, detail=""):
    rows.append(bool(ok)); print(f"  {'PASS' if ok else 'FAIL'}  {label:62s} {detail}", flush=True)

def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)

def opposite_order(sol):
    """The next rate pass of sol runs over the grid in the opposite direction (the
    solver alternates the direction with the step count): a second run whose
    integrator maps do not adapt along the same history as the first one's."""
    sol.step_count = 1
    return sol

def within(diff, spread):
    """diff against K_SIGMA times the measured spread of two runs, or rounding on common samples."""
    return diff <= K_SIGMA * spread + ROUNDING, f"diff {diff:.1e}, spread of two runs {spread:.1e}"

def same_rates(s1, s2, r1, r2, procs1, procs2, J=False):
    """Largest relative difference of rates (None: gain and loss only, for an
    elastic process near equilibrium whose net is noise), gain, loss and the
    ledgers summed over procs1 and over procs2, over the species of s1; with J,
    also of the Jacobian applied to the distributions (J . f), returned separately."""
    worst, worst_J = 0.0, 0.0
    f_all = np.concatenate([s1.distributions_1d[sp_] for sp_ in s1.species_list])
    for sp_ in s1.species_list:
        if r1 is not None:
            worst = max(worst, rel(r1[sp_], r2[sp_]))
        worst = max(worst, rel(s1.gain_rates[sp_], s2.gain_rates[sp_]), rel(s1.loss_rates[sp_], s2.loss_rates[sp_]))
        for group1, group2 in zip(procs1, procs2):
            for ledger in ('gain_rates_by_process', 'loss_rates_by_process'):
                worst = max(worst, rel(sum(getattr(s1, ledger)[p][sp_] for p in group1), sum(getattr(s2, ledger)[p][sp_] for p in group2)))
        if J:
            worst_J = max(worst_J, rel(s1.jacobian[sp_] @ f_all, s2.jacobian[sp_] @ f_all))
    return (worst, worst_J) if J else worst

quiet = contextlib.redirect_stdout(io.StringIO())


# ---- 1. structure checks ---------------------------------------------------------
print("\n1. structure checks of CP-conjugate pairs")
with quiet:
    sol = build(paired=False)
def rejects(label, name='bad', inp=('N',), out=('Phibar', 'lbar'), **kw):
    kw.setdefault('neval', NEVAL)
    try:
        with quiet:
            sol.add_process(name, list(inp), list(out), MB, **kw)
    except ValueError as e:
        check(label, name not in sol.process_configs, f"({str(e)[:60]}...)")
        return
    check(label, False, "no ValueError")
rejects("partner not registered: rejected, nothing registered", cp_conjugate='nothing')
rejects("a process as its own CP conjugate: rejected", cp_conjugate='bad')
rejects("reverse |M|^2 for one member only: rejected", cp_conjugate='tA', mesq_backward=M0)
rejects("different neval: rejected", cp_conjugate='tA', neval=NEVAL // 2)
rejects("legs differing slot by slot: rejected", cp_conjugate='tA', out=('lbar', 'Phibar'))
rejects("different numbers of legs: rejected", cp_conjugate='tA', inp=('Phi', 'x'), out=('N', 'x'))
with quiet:
    for name, anti in (('Phi2', 'Phi2bar'), ('Phi2bar', 'Phi2')):
        sol.initialize_species(name, lambda q: 1.0 / np.expm1(np.sqrt(q * q + 0.09) / T), stat='boson', mass=0.3, dof=2, antiparticle=anti)
    sol.add_process('tA2', ['Phi', 'Phi'], ['N', 'x'], MA, neval=NEVAL)
rejects("identical particles in different slots: rejected", cp_conjugate='tA2', inp=('Phibar', 'Phi2'), out=('N', 'x'), symmetry_factor=0.5)
rejects("slot species not the declared antiparticle: rejected", cp_conjugate='tA', out=('Phi2bar', 'lbar'))
try:
    with quiet:
        sol.initialize_species('Phi3', lambda q: 1.0 / np.expm1(np.sqrt(q * q + 0.09) / T), stat='boson', mass=0.3, dof=2, antiparticle='Phibar')
    check("contradicting antiparticle declaration: rejected", False, "no ValueError")
except ValueError as e:
    check("contradicting antiparticle declaration: rejected", sol.species_antiparticle['Phibar'] == 'Phi', f"({str(e)[:60]}...)")
with quiet:
    sol.add_process('tB2', ['N'], ['Phibar', 'lbar'], MB, neval=NEVAL, cp_conjugate='tA')
rejects("partner already paired: rejected", cp_conjugate='tA')
with quiet:
    sol.add_process('tA', ['N'], ['Phi', 'l'], MA, neval=NEVAL)          # re-registered without the link
check("re-registering a member drops the old link", sol.process_configs['tA']['cp_conjugate'] is None
      and sol.process_configs['tB2']['cp_conjugate'] is None)
with quiet:
    sol2 = BEST(q_min=0.02, q_max=20.0, n_grid=8)
    sol2.add_process('x', ['A'], ['B', 'C'], M0)
try:
    with quiet:
        sol2.add_process('y', ['A'], ['B', 'C'], M0, cp_conjugate='x')
    check("uninitialized species: rejected", False, "no ValueError")
except ValueError as e:
    check("uninitialized species: rejected", 'y' not in sol2.process_configs, f"({str(e)[:60]}...)")


# ---- 2. paired sets == T-symmetric reference --------------------------------------
print("\n2. paired sets against the T-symmetric reference (all species evolving)")
ref_rates = {}
for want_J in (False, True):
    with quiet:
        sp, se, sr, sr2 = build(paired=True), build(paired=True, effective=True), build(paired=False), opposite_order(build(paired=False))
        for s_ in (sp, se, sr, sr2):
            s_._want_jacobian = want_J
        rp = sp._compute_rates_vegas(['decA', 'decB', 'risA', 'risB'], t=0.0)
        re_ = se._compute_rates_vegas(['decA', 'decB'], t=0.0)
        rr = sr._compute_rates_vegas(['tA', 'tB'], t=0.0)
        rr2 = sr2._compute_rates_vegas(['tA', 'tB'], t=0.0)
    ref_rates[want_J] = (sp, rp)
    spread = same_rates(sr, sr2, rr[0], rr2[0], (('tA',), ('tB',)), (('tA',), ('tB',)), J=want_J)
    for label, s_, r_, groups in (("CPT pair + remainder", sp, rp, (('decA', 'risA'), ('decB', 'risB'))), ("effective pair", se, re_, (('decA',), ('decB',)))):
        diff = same_rates(s_, sr, r_[0], rr[0], groups, (('tA',), ('tB',)), J=want_J)
        if want_J:
            check(f"{label}: rates, gain, loss, ledgers = reference (J pass)", *within(diff[0], spread[0]))
            check(f"{label}: J . f = reference", *within(diff[1], spread[1]))
        else:
            check(f"{label}: rates, gain, loss, ledgers = reference", *within(diff, spread))

print("   pair = the same two processes written unpaired with explicit reverse |M|^2 (eps = 0.5; gain, loss, ledgers, J . f)")
big, small = functools.partial(scaled, c=1.5), functools.partial(scaled, c=0.5)
rev_a, rev_b = functools.partial(scaled, c=1.2), functools.partial(scaled, c=0.8)
legs = [(['N'], ['Phi', 'l'], ['N'], ['Phibar', 'lbar']),
        (['Phi', 'l'], ['N', 'x'], ['Phibar', 'lbar'], ['N', 'x']),
        (['Phi', 'l'], ['Phibar', 'x'], ['Phibar', 'lbar'], ['Phi', 'x']),
        (['Phi', 'Phi'], ['x', 'x'], ['Phibar', 'Phibar'], ['x', 'x']),
        (['N', 'Phi'], ['N', 'Phi'], ['N', 'Phibar'], ['N', 'Phibar'])]
for inA, outA, inB, outB in legs:
    for want_J in (False, True):
        for own in (False, True):         # the CPT reverse, or four different functions
            if own and inA != ['N'] and inA != ['Phi', 'l']:
                continue
            ra, rb = (rev_a, rev_b) if own else (small, big)
            with quiet:
                solvers = [BEST(q_min=0.05, q_max=15.0, n_grid=10) for _ in range(3)]   # pair, unpaired, unpaired again
                for s_ in solvers:
                    s_.verbose = False
                    species(s_)
                    s_._want_jacobian = want_J
                s1, s2, s3 = solvers
                opposite_order(s3)
                s1.add_process('A', inA, outA, big, neval=NEVAL, nitn=NITN, mesq_backward=ra if own else None)
                s1.add_process('B', inB, outB, small, neval=NEVAL, nitn=NITN, mesq_backward=rb if own else None, cp_conjugate='A')
                for s_ in (s2, s3):
                    s_.add_process('A', inA, outA, big, neval=NEVAL, nitn=NITN, mesq_backward=ra)
                    s_.add_process('B', inB, outB, small, neval=NEVAL, nitn=NITN, mesq_backward=rb)
                for s_ in solvers:
                    s_._compute_rates_vegas(['A', 'B'], t=0.0)
            label = f"{'+'.join(inA)} -> {'+'.join(outA)} | {'+'.join(inB)} -> {'+'.join(outB)}{', own reverse' if own else ''}"
            groups = ((('A',), ('B',)), (('A',), ('B',)))
            diff = same_rates(s1, s2, None, None, *groups, J=want_J)
            spread = same_rates(s2, s3, None, None, *groups, J=want_J)
            if want_J:
                check(f"{label}: J . f", *within(diff[1], spread[1]))
            else:
                check(f"{label}: gain, loss, ledgers", *within(diff, spread))

# ---- 3. exact ratios within one pass ----------------------------------------------
print("\n3. ratios within one pass (same samples): to rounding")
sp = ref_rates[False][0]
ratio = (1.0 + EPS) / (1.0 - EPS)
def ratio_check(label, a, b, r):
    m = (a != 0.0) & (b != 0.0)
    worst = float(np.max(np.abs(a[m] / b[m] / r - 1.0))) if m.any() else np.inf
    check(label, worst < ROUNDING and m.any(), f"max |ratio/expected - 1| = {worst:.1e} on {m.sum()} nodes")
ratio_check("gain(Phi)/gain(Phibar) = (1+eps)/(1-eps)", sp.gain_rates_by_process['decA']['Phi'], sp.gain_rates_by_process['decB']['Phibar'], ratio)
ratio_check("gain(l)/gain(lbar) = (1+eps)/(1-eps)", sp.gain_rates_by_process['decA']['l'], sp.gain_rates_by_process['decB']['lbar'], ratio)
ratio_check("loss(Phi)/loss(Phibar) = (1-eps)/(1+eps)  (CPT reverse)", sp.loss_rates_by_process['decA']['Phi'], sp.loss_rates_by_process['decB']['Phibar'], 1.0 / ratio)
ratio_check("loss(N | decA)/loss(N | decB) = (1+eps)/(1-eps)", sp.loss_rates_by_process['decA']['N'], sp.loss_rates_by_process['decB']['N'], ratio)
ratio_check("gain(N | decA)/gain(N | decB) = (1-eps)/(1+eps)", sp.gain_rates_by_process['decA']['N'], sp.gain_rates_by_process['decB']['N'], 1.0 / ratio)

# ---- 4. noise-free difference at eps = 0 ------------------------------------------
print("\n4. eps = 0: paired species equal to rounding, unpaired ones not")
def mirror_diff(sol, rates):
    return max(max(rel(rates[a], rates[b]), rel(sol.gain_rates[a], sol.gain_rates[b]), rel(sol.loss_rates[a], sol.loss_rates[b]))
               for a, b in (('l', 'lbar'), ('Phi', 'Phibar')))
with quiet:
    sp0, sr0 = build(paired=True, eps=0.0), build(paired=False, eps=0.0)
    rp0 = sp0._compute_rates_vegas(['decA', 'decB'], t=0.0)
    rr0 = sr0._compute_rates_vegas(['tA', 'tB'], t=0.0)
d = mirror_diff(sp0, rp0[0])
check("paired: rates of l and lbar, Phi and Phibar equal to rounding", d < ROUNDING, f"rel diff {d:.1e}")
diff = max(rel(rr0[0]['l'], rr0[0]['lbar']), rel(rr0[0]['Phi'], rr0[0]['Phibar']))
check("unpaired: the two integrations differ (Monte Carlo noise)", diff > ROUNDING, f"rel diff {diff:.1e}")
with quiet:
    sp0.bw_fallback_rel_err = 0.0                       # the direct backward integration on every node
    rp0 = sp0._compute_rates_vegas(['decA', 'decB'], t=0.0)
d = mirror_diff(sp0, rp0[0])
check("paired, backward fallback on every node: equal to rounding", d < ROUNDING,
      f"rel diff {d:.1e}, fallback taken on {sp0.error_stats.get('bw_fallback', 0)} nodes")
with quiet:
    sp0 = build(paired=True, eps=0.0)
    sp0.evolve_step(0.05, active_processes=['decA', 'decB', 'risA', 'risB'], adapt_dt=False, method='exprb_seq')
d = max(rel(sp0.distributions_1d['l'], sp0.distributions_1d['lbar']), rel(sp0.distributions_1d['Phi'], sp0.distributions_1d['Phibar']))
check("paired, one exprb_seq step: f_l and f_lbar equal to rounding", d < ROUNDING, f"rel diff {d:.1e}")

# ---- 5. direct asymmetry against the linear asymmetry equation --------------------
print("\n5. run with Phi, Phibar prescribed: f_l - f_lbar against the linear asymmetry equation")
active = ['decA', 'decB', 'risA', 'risB']

def coefficients(sol):
    """S = 2 eps (X_dec (1 - fbar) - X_ID fbar) and W = X_dec + X_ID (Pauli blocking of
    the produced lepton) from the run's own ledgers: every direction of l and lbar
    carries (1 +- eps) times the same kernels, so the linear equation for
    delta = f_l - f_lbar is d delta/dt = S - W delta exactly."""
    f_l, f_lb = sol.distributions_1d['l'], sol.distributions_1d['lbar']
    net_l = sol.gain_rates['l'] - sol.loss_rates['l']
    net_lb = sol.gain_rates['lbar'] - sol.loss_rates['lbar']
    W_l = assemble_A_Gamma(f_l, sol.gain_rates['l'], sol.loss_rates['l'], 'fermion')[1] / (1.0 + EPS)
    W_lb = assemble_A_Gamma(f_lb, sol.gain_rates['lbar'], sol.loss_rates['lbar'], 'fermion')[1] / (1.0 - EPS)
    return EPS * (net_l / (1.0 + EPS) + net_lb / (1.0 - EPS)), 0.5 * (W_l + W_lb)

def asymmetry_run(halvings):
    """The direct run against the linear equation over the same time span; the step
    is W_DT / 2^halvings in units of the wash-out time."""
    with quiet:
        sol = build(paired=True, evolving=['N', 'l', 'lbar'])
        sol.compute_rates(active)
    S1, W1 = coefficients(sol)
    delta = np.zeros_like(sol.distributions_1d['l'])
    dt = W_DT / np.max(W1) / 2**halvings
    for _ in range(20 * 2**halvings):
        with quiet:
            sol.evolve_step(dt, active_processes=active, adapt_dt=False, method='exprb2')
            sol.compute_rates(active)
        S2, W2 = coefficients(sol)
        delta = exprb2_update(delta, S1, W1, S2, W2, dt)
        S1, W1 = S2, W2
    d_direct = sol.distributions_1d['l'] - sol.distributions_1d['lbar']
    q = sol.r_grids['l']
    err = np.sqrt(np.trapezoid(q * q * (d_direct - delta)**2, q) / np.trapezoid(q * q * delta**2, q))
    size = np.sqrt(np.trapezoid(q * q * delta**2, q) / np.trapezoid(q * q * sol.distributions_1d['l']**2, q))
    return err, size

err1, size = asymmetry_run(0)
err2, _ = asymmetry_run(1)
check(f"f_l - f_lbar vs the linear equation: W dt = {W_DT:g} and {W_DT / 2:g}", err2 < err1 / 2,
      f"rel L2 diff {err1:.1e} and {err2:.1e} (ratio {err1 / err2:.2f}; second order gives 4), asymmetry/f = {size:.1e}")

# ---- 6. checkpoint ----------------------------------------------------------------
print("\n6. checkpoint round trip of the paired set")
with quiet:
    sp = build(paired=True)
    sp.add_process('risC', ['N'], ['Phi', 'l'], zero, neval=NEVAL, nitn=NITN, mesq_backward=functools.partial(scaled, c=0.1))
    active = ['decA', 'decB', 'risA', 'risB', 'risC']
    sp.compute_rates(active)
    fn = os.path.join(tempfile.mkdtemp(), 'pair.pkl')
    sp.save_checkpoint(fn)
    rp2 = sp.compute_rates(active)            # a second pass: the same integrator maps as the checkpoint's
    sq, sq2 = BEST(q_min=0.02, q_max=20.0, n_grid=24), BEST(q_min=0.02, q_max=20.0, n_grid=24)
    sq.verbose = sq2.verbose = False
try:
    with quiet:
        sq.load_checkpoint(fn, matrix_elements_squared={'decA': MA, 'decB': MB, 'risA': (zero, remA), 'risB': (zero, remB), 'risC': zero})
    check("a reverse |M|^2 that cannot be resolved is refused", False, "no ValueError")
except ValueError as e:
    check("a reverse |M|^2 that cannot be resolved is refused", True, f"({str(e)[:60]}...)")
with quiet:
    restore = {'decA': MA, 'decB': MB, 'risA': (zero, remA), 'risB': (zero, remB), 'risC': (zero, sp.process_configs['risC']['mesq_backward'])}
    sq.load_checkpoint(fn, matrix_elements_squared=restore)
    sq2.load_checkpoint(fn, matrix_elements_squared=restore)
    rq, rq2 = sq.compute_rates(active), opposite_order(sq2).compute_rates(active)
c = sq.process_configs
check("pair links, reverse |M|^2 and antiparticles restored",
      c['decA']['cp_conjugate'] == 'decB' and c['decB']['cp_conjugate'] == 'decA' and c['risA']['mesq_backward'] is remA
      and c['risB']['mesq_backward'] is remB and c['risA']['matrix_element_squared'] is zero
      and sq.species_antiparticle == sp.species_antiparticle)
# the restored run and the second pass start from the same adapted maps; two restores give the spread
check("rates after the restore = a further pass of the original", *within(max(rel(rp2[s_], rq[s_]) for s_ in sp.species_list),
                                                                       max(rel(rq[s_], rq2[s_]) for s_ in sp.species_list)))
try:
    with quiet:
        BEST(q_min=0.02, q_max=20.0, n_grid=24).load_checkpoint(fn, matrix_elements_squared={'decA': (MA, M0), 'decB': MB, 'risA': (zero, remA), 'risB': (zero, remB), 'risC': (zero, M0)})
    check("a reverse |M|^2 for a process saved without one is refused", False, "no ValueError")
except ValueError as e:
    check("a reverse |M|^2 for a process saved without one is refused", True, f"({str(e)[:60]}...)")

# ---- 7. one member active ---------------------------------------------------------
print("\n7. only decA active")
with quiet:
    sp, sp2, sf, sf2 = (build(paired=True) for _ in range(4))     # first passes all: two with decA alone, two with the full set
    r1 = sp.compute_rates(['decA'])
    opposite_order(sp2).compute_rates(['decA'])
    sf.compute_rates(['decA', 'decB', 'risA', 'risB'])
    opposite_order(sf2).compute_rates(['decA', 'decB', 'risA', 'risB'])
check("partner's species get nothing", np.all(r1['Phibar'] == 0.0) and np.all(r1['lbar'] == 0.0))
def ledger_diff(a, b):
    return max(max(rel(a.gain_rates_by_process['decA'][s_], b.gain_rates_by_process['decA'][s_]),
                   rel(a.loss_rates_by_process['decA'][s_], b.loss_rates_by_process['decA'][s_])) for s_ in ('N', 'Phi', 'l'))
check("N, Phi, l get decA's ledger of the full run", *within(ledger_diff(sp, sf), max(ledger_diff(sp, sp2), ledger_diff(sf, sf2))))
with quiet:
    r = sp.r_grids['N'][12]
    one = sp.compute_collision_rate(r, 'N', ['decA'])
    both = sp.compute_collision_rate(r, 'N', ['decA', 'decB'])
    again = [sp.compute_collision_rate(r, 'N', ['decA']) for _ in range(4)]   # one node: the spread needs a few draws
diff = max(rel(one[k][:3], both[k][:3]) for k in one)
spread = max(rel(a[k][:3], b[k][:3]) for a in again for b in again for k in one)
check("compute_collision_rate books each (species, process) once",
      set(one) == set(both) == {('N', 'decA'), ('N', 'decB')} and within(diff, spread)[0], within(diff, spread)[1])
try:
    with quiet:
        sp.set_mass_func('l', lambda t: 0.2 + 0.1 * t)
        sp.compute_rates(['decA', 'decB'], t=1.0)
    check("a mass function for one species of a pair only stops the rate pass", False, "no ValueError")
except ValueError as e:
    check("a mass function for one species of a pair only stops the rate pass", True, f"({str(e)[:60]}...)")

# ---- 8. a reaction that is its own reverse ----------------------------------------
print("\n8. Phi l -> Phi l counted once: loss rate of l against |M|^2 /(64 pi^3 q g_l) int p e^{-p/T} dp")
M2_EL = 3.0
def el_const(m): return np.full(m.shape[2], M2_EL)
def el_const2(m): return np.full(m.shape[2], M2_EL)
forms = {'default': {}, 'own reverse': {'mesq_backward': el_const2}, 'CP pair': None}
for label, kw in forms.items():
    with quiet:
        se = BEST(q_min=0.05, q_max=25.0, n_grid=48)      # the sampled f_Phi is interpolated between the nodes
        se.verbose = False
        for sp_, anti in (('l', 'lbar'), ('lbar', 'l'), ('Phi', 'Phibar'), ('Phibar', 'Phi')):
            se.initialize_species(sp_, lambda q: np.exp(-q / T), stat='mb', mass=0.0, dof=2, antiparticle=anti)
        if kw is None:
            se.add_process('el', ['Phi', 'l'], ['Phi', 'l'], el_const, neval=NEVAL, nitn=NITN)
            se.add_process('elbar', ['Phibar', 'lbar'], ['Phibar', 'lbar'], el_const2, neval=NEVAL, nitn=NITN, cp_conjugate='el')
            se.compute_rates(['el', 'elbar'])
        else:
            se.add_process('el', ['Phi', 'l'], ['Phi', 'l'], el_const, neval=NEVAL, nitn=NITN, **kw)
            se.compute_rates(['el'])
    q = se.r_grids['l']
    # the momentum of the Phi is sampled on [q_min, q_max * domain_extension]
    a_, b_ = se.q_min / T, se.q_max * se.domain_extension / T
    gamma = M2_EL * T * T * ((a_ + 1.0) * np.exp(-a_) - (b_ + 1.0) * np.exp(-b_)) / (64.0 * np.pi**3 * q * se.species_dof['l'])
    r = se.loss_rates['l'] / se.distributions_1d['l'] / gamma
    mean, sigma = r.mean(), r.std(ddof=1) / np.sqrt(r.size)        # the nodes are independent estimates
    check(f"{label}: loss rate / analytic", abs(mean - 1.0) <= K_SIGMA * sigma + ROUNDING,
          f"mean ratio {mean:.4f} +- {sigma:.4f} over {r.size} nodes")

print(f"\n{sum(rows)}/{len(rows)} checks passed")
sys.exit(0 if all(rows) else 1)
