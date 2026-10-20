#!/usr/bin/env python3
"""Custom collision terms against the Vegas decay N -> Phi l.

The decay and inverse decay are written as the one-dimensional kernels that the
two-body kinematics leave after the angular integrations (the forms of
Hahn-Woernle, Pluemacher and Wong 2009, eqs. 3.3 and 3.8, with Bose and Fermi
factors), registered with add_custom_term, and compared with the Vegas process
'dec' of the solver on the same state:

  1. rates of N, Phi, l (gain and loss) node by node;
  2. the kernel's Jacobian (finite differences of the kernel) against the
     solver's Vegas Jacobian, and the plumbing of a user-supplied Jacobian;
  3. a few exprb_full steps with either term, and exprb_seq with a mixed list;
  4. seeding of an empty species from a custom term and a checkpoint round trip.

Run: python3 scripts/check_custom_term.py            (one rank, a few minutes)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST

T = 1.0
mN, mP, ml = 1.0, 0.3, 0.2
M2 = 2.0                                         # Sum over all internal states of |M|^2
NQ = 400                                         # Gauss-Legendre points of the kernels

def fd_(E, Tx): return 1.0 / (np.exp(E / Tx) + 1.0)
def be_(E, Tx): return 1.0 / np.expm1(E / Tx)

def build(seed_N=False):
    sol = BEST(q_min=0.05, q_max=15.0, n_grid=24); sol.verbose = False
    sol.bw_fallback_rel_err = np.inf
    sol.initialize_species('N', (lambda q: 0.0) if seed_N else (lambda q: 0.5 * fd_(np.sqrt(q * q + mN * mN), 1.2)),
                           stat='fermion', mass=mN, dof=2)
    sol.initialize_species('Phi', lambda q: be_(np.sqrt(q * q + mP * mP), 1.0), stat='boson', mass=mP, dof=2)
    sol.initialize_species('l', lambda q: fd_(np.sqrt(q * q + ml * ml), 0.9), stat='fermion', mass=ml, dof=2)
    sol.current_time = 0.0
    return sol

# ---- the kernels -------------------------------------------------------------
XG, WG = np.polynomial.legendre.leggauss(NQ)

def two_body_window(E, p, m, m_s, m_c):
    """Lab energies of daughter s when a parent (E, p, m) decays to s + c."""
    Es = (m * m + m_s * m_s - m_c * m_c) / (2 * m)
    ps = np.sqrt(np.maximum(Es * Es - m_s * m_s, 0.0))
    g, b = E / m, p / E
    return g * (Es - b * ps), g * (Es + b * ps)

def decay_kernel(sol):
    """{species: (gain, loss)} of N <-> Phi l from the current distributions."""
    a = sol.scale_factor(sol.current_time)
    fN, fP, fl = (sol.interpolators[s] for s in ('N', 'Phi', 'l'))
    g = sol.species_dof
    out = {}
    # N: average over the daughter energy, uniform in the window (3.3)
    q = sol.r_grids['N']; p = q / a; E = np.sqrt(p * p + mN * mN)
    lo, hi = two_body_window(E, p, mN, mP, ml)
    EP = 0.5 * (hi - lo)[:, None] * XG[None, :] + 0.5 * (hi + lo)[:, None]
    w = 0.5 * (hi - lo)[:, None] * WG[None, :]
    k = np.sqrt(EP * EP - mP * mP); kp = np.sqrt(np.maximum((E[:, None] - EP) ** 2 - ml * ml, 0.0))
    FP, Fl, FN = fP(k * a), fl(kp * a), fN(q)[:, None]
    coef = M2 / (2 * E * g['N']) / (8 * np.pi * p)
    out['N'] = (coef * (w * FP * Fl * (1 - FN)).sum(1), coef * (w * FN * (1 + FP) * (1 - Fl)).sum(1))
    # daughters: integral over the parent momentum, the angle fixed by energy conservation (3.8);
    # the parent momenta that reach the daughter momentum k form an interval, found from the
    # window of the daughter energy, E-(pN) <= E_k <= E+(pN), and integrated exactly inside it.
    for sp, m_s, m_c, sp_c, eta_s, eta_c in (('Phi', mP, ml, 'l', +1, -1), ('l', ml, mP, 'Phi', -1, +1)):
        q = sol.r_grids[sp]; ks = q / a; Es = np.sqrt(ks * ks + m_s * m_s)
        p_top = sol.q_max * sol.domain_extension / a
        scan = np.linspace(0.0, p_top, 20001)[1:]
        lo_s, hi_s = two_body_window(np.sqrt(scan * scan + mN * mN), scan, mN, m_s, m_c)
        gain = np.zeros_like(q); loss = np.zeros_like(q)
        for i in range(len(q)):
            inside = (lo_s <= Es[i]) & (Es[i] <= hi_s)
            if not inside.any():
                continue
            idx = np.flatnonzero(inside)
            pa = scan[idx[0] - 1] if idx[0] > 0 else 0.0; pb = scan[idx[0]]
            for _ in range(40):                              # lower edge by bisection
                pm = 0.5 * (pa + pb); l_, h_ = two_body_window(np.sqrt(pm * pm + mN * mN), pm, mN, m_s, m_c)
                if l_ <= Es[i] <= h_: pb = pm
                else: pa = pm
            p_lo = pb
            if idx[-1] == len(scan) - 1:
                p_hi = p_top
            else:
                pa = scan[idx[-1]]; pb = scan[idx[-1] + 1]
                for _ in range(40):                          # upper edge by bisection
                    pm = 0.5 * (pa + pb); l_, h_ = two_body_window(np.sqrt(pm * pm + mN * mN), pm, mN, m_s, m_c)
                    if l_ <= Es[i] <= h_: pa = pm
                    else: pb = pm
                p_hi = pa
            pN = 0.5 * (p_hi - p_lo) * XG + 0.5 * (p_hi + p_lo); wN = 0.5 * (p_hi - p_lo) * WG
            EN = np.sqrt(pN * pN + mN * mN)
            kc = np.sqrt(np.maximum((EN - Es[i]) ** 2 - m_c * m_c, 0.0))
            FN = fN(pN * a); Fc = sol.interpolators[sp_c](kc * a); Fs = sol.interpolators[sp](q[i])
            meas = wN * pN / (8 * np.pi * ks[i] * EN)
            coef = M2 / (2 * Es[i] * g[sp])
            gain[i] = coef * (meas * FN * (1 + eta_s * Fs) * (1 + eta_c * Fc)).sum()
            loss[i] = coef * (meas * Fs * Fc * (1 - FN)).sum()
        out[sp] = (gain, loss)
    return out

def decay_kernel_jacobian(sol):
    """Exact Jacobian of decay_kernel by central differences of the kernel itself."""
    from besthep import ExtrapolatingInterp
    sps = sol.species_list
    f0 = {s: sol.distributions_1d[s].copy() for s in sps}
    def net(fd):
        for s in sps:
            sol.distributions_1d[s] = fd[s]
            sol.interpolators[s] = ExtrapolatingInterp(sol.r_grids[s], fd[s], mass=sol.species_mass[s],
                                                       stat=sol.species_config[s], a=sol.scale_factor(sol.current_time))
        out = decay_kernel(sol)
        return {s: out[s][0] - out[s][1] for s in sps}
    J = {s: {s2: np.zeros((len(sol.r_grids[s]), len(sol.r_grids[s2]))) for s2 in sps} for s in sps}
    for s2 in sps:
        for k in range(len(sol.r_grids[s2])):
            d = 1e-4 * f0[s2][k]
            fp = {s: f0[s].copy() for s in sps}; fm = {s: f0[s].copy() for s in sps}
            fp[s2][k] += d; fm[s2][k] -= d
            cp, cm = net(fp), net(fm)
            for s in sps:
                J[s][s2][:, k] = (cp[s] - cm[s]) / (2 * d)
    net(f0)
    return J

# ---- helpers -----------------------------------------------------------------
import io, contextlib
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)

def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

rows = []
def check(label, ok, detail):
    rows.append(ok); print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} {detail}")

# ---- 1. rates ------------------------------------------------------------------
print("1. rates of the kernel vs the Vegas decay (gain and loss per species)")
sol = build()
sol.add_process('dec', ['N'], ['Phi', 'l'], lambda m: np.full(m.shape[2], M2), neval=40000)
sol._want_jacobian = True
r_v, _, _ = quiet(sol._compute_rates_vegas, ['dec'], t=0.0)
gain_v = {s: sol.gain_rates_by_process['dec'][s].copy() for s in sol.species_list}
loss_v = {s: sol.loss_rates_by_process['dec'][s].copy() for s in sol.species_list}
J_v = {s: sol.jacobian[s].copy() for s in sol.species_list}
ker = decay_kernel(sol)
for s in sol.species_list:
    g, l = ker[s]
    eg, el = rel(g, gain_v[s]), rel(l, loss_v[s])
    check(f"{s}: gain and loss, rel L2 vs Vegas", max(eg, el) < 2e-2, f"gain {eg:.2e}  loss {el:.2e}")

# ---- 2. Jacobian -----------------------------------------------------------------
print("\n2. Jacobian: kernel (finite differences) vs Vegas J, and the plumbing of a user Jacobian")
J_k = decay_kernel_jacobian(sol)
off = sol._jacobian_offsets()
for s in sol.species_list:
    for s2 in sol.species_list:
        blk_v = J_v[s][:, off[s2]:off[s2] + len(sol.r_grids[s2])]
        blk_k = J_k[s][s2]
        if np.linalg.norm(blk_k) == 0:
            continue
        # compare on a smooth direction: the derivative along delta f = 1e-3 f
        v = 1e-3 * sol.distributions_1d[s2]
        e = rel(blk_v @ v, blk_k @ v)
        check(f"dC_{s}/df_{s2} . (1e-3 f), Vegas vs kernel", e < 3e-2, f"rel {e:.2e}")
sol2 = build()
sol2.add_custom_term('dec_k', decay_kernel, decay_kernel_jacobian)
sol2._want_jacobian = True
r_k, _, _ = quiet(sol2._compute_rates_vegas, ['dec_k'], t=0.0)
Jk_flat = {s: np.hstack([J_k[s][s2] for s2 in sol2.species_list]) for s in sol2.species_list}
ok = all(np.allclose(sol2.jacobian[s], Jk_flat[s], rtol=0, atol=1e-14 + 1e-10 * np.abs(Jk_flat[s]).max()) for s in sol2.species_list)
check("user Jacobian lands in solver.jacobian (all blocks)", ok, "")
ok = all(np.allclose(r_k[s], ker[s][0] - ker[s][1]) for s in sol2.species_list)
check("custom term net rate == gain - loss of the kernel", ok, "")

# ---- 3. evolution ---------------------------------------------------------------
print("\n3. evolution: exprb_full with the Vegas decay vs with the kernel (same dt), exprb_seq mixed")
dt, nst = 0.5, 4
sa = build(); sa.add_process('dec', ['N'], ['Phi', 'l'], lambda m: np.full(m.shape[2], M2), neval=40000)
sb = build(); sb.add_custom_term('dec_k', decay_kernel, decay_kernel_jacobian)
for _ in range(nst):
    quiet(sa.evolve_step, dt=dt, method='exprb_full', adapt_dt=False)
    quiet(sb.evolve_step, dt=dt, method='exprb_full', adapt_dt=False)
for s in sa.species_list:
    e = rel(sb.distributions_1d[s], sa.distributions_1d[s])
    dn = abs(np.trapezoid(sb.distributions_1d[s] * sb.r_grids[s] ** 2, sb.r_grids[s])
             / np.trapezoid(sa.distributions_1d[s] * sa.r_grids[s] ** 2, sa.r_grids[s]) - 1)
    check(f"{s}: f after {nst} steps, kernel vs Vegas", e < 2e-2, f"rel L2 {e:.2e}  number {dn:.2e}")
sc = build()
sc.add_process('el', ['Phi', 'l'], ['Phi', 'l'], lambda m: np.full(m.shape[2], 1.0), neval=20000)
sc.add_custom_term('dec_k', decay_kernel)
quiet(sc.evolve_step, dt=dt, method='exprb_seq', adapt_dt=False)
check("exprb_seq with a Vegas process and a custom term runs", np.all(np.isfinite(np.concatenate(list(sc.distributions_1d.values())))), "")
sd = build(); sd.add_custom_term('dec_k', decay_kernel, decay_kernel_jacobian)
quiet(sd.evolve_step, dt=dt, method='exprb_full', adapt_dt=False)
ok = all(np.allclose(sd.jacobian[s], Jk_flat[s], rtol=0, atol=1e-14 + 1e-10 * np.abs(Jk_flat[s]).max()) for s in sd.species_list)
check("after exprb_full, solver.jacobian is J of the step's initial state", ok, "")

# ---- 4. seeding and checkpoint ----------------------------------------------------
print("\n4. seeding from a custom term and checkpoint restore")
se = build(seed_N=True)
se.add_custom_term('dec_k', decay_kernel)
quiet(se.evolve_step, dt=dt, method='exprb', adapt_dt=False)
nN = np.trapezoid(se.distributions_1d['N'] * se.r_grids['N'] ** 2, se.r_grids['N'])
check("empty N seeded and produced by the custom term alone", nN > 0 and np.isfinite(nN), f"n_N = {nN:.3e}")
fn = 'check_custom_term.pkl'
quiet(se.save_checkpoint, fn)
quiet(se.evolve_step, dt=dt, method='exprb', adapt_dt=False)        # one more step of the original
sf = build(seed_N=True)                                             # a script that resumes may still initialize N empty
quiet(sf.load_checkpoint, fn, custom_terms={'dec_k': decay_kernel})
quiet(sf.evolve_step, dt=dt, method='exprb', adapt_dt=False)
diff = max(np.max(np.abs(sf.distributions_1d[s] - se.distributions_1d[s]) / np.max(se.distributions_1d[s])) for s in se.species_list)
check("checkpoint restores the custom term; a step after the restore equals the original's next step",
      'dec_k' in sf.custom_terms and diff <= NQ * np.finfo(float).eps,
      f"max rel diff {diff:.1e} (deterministic kernel: the rounding of its {NQ}-point sums; a species restored non-empty must not be re-seeded)")
os.remove(fn)

print(f"\n{sum(rows)}/{len(rows)} checks passed")
sys.exit(0 if all(rows) else 1)
