#!/usr/bin/env python3
"""Diagnostics for the Majorana decay run (examples/Mdecay.py) from its checkpoint.

Panels: (1) comoving N number vs z = M/T with the Fermi-Dirac equilibrium value,
(2) deviation of the bath species (Phi, l) from the Bose-Einstein / Fermi-Dirac
distribution with the (T, mu) solved from each species' own comoving number and
energy (the bath has no number-changing process of its own, so a mu is physical
and must not be counted as a deviation); (3) energy budget: total comoving
energy vs the integrated redshift of the N mass term (collisions conserve energy
if the two agree); (4) f_N(q) snapshots with the Fermi-Dirac distribution at the
same T.

T(t) is fitted per snapshot from the stored Phi spectrum (log(1/f + 1) = E/T), so
no run-script constants are duplicated here. Species names, masses, statistics
and dof come from the checkpoint. Usage: python plot_decay.py [checkpoint.pkl]
"""
import sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.pkl"
N_SP, BATH_B, BATH_F = "N", "Phi", "l"          # decaying species, boson bath, fermion bath
N_SNAP = 8

with open(CHECKPOINT, "rb") as fh:
    state = pickle.load(fh)
h = state["history"]
mass = state["species_mass"]; stat = state["species_config"]; dof = state.get("species_dof", {})
t = np.asarray(h["times"], float); a = np.asarray(h["a"], float)
q = {sp: np.asarray(state["r_grids"][sp], float) for sp in (N_SP, BATH_B, BATH_F)}
F = {sp: [np.asarray(f, float) for f in h[sp]["f"]] for sp in (N_SP, BATH_B, BATH_F)}
M = float(mass[N_SP]); gN = float(dof.get(N_SP, 2))
def mass_at(sp, i):                                        # per-record mass when the history has it
    m_hist = h[sp].get("m")
    return float(m_hist[i]) if m_hist else float(mass[sp])
eta = {sp: (1.0 if stat[sp] == "boson" else -1.0) for sp in (N_SP, BATH_B, BATH_F)}

def eq_f(sp, qq, ai, T, i):                                # equilibrium occupation, comoving q
    E = np.sqrt((qq / ai) ** 2 + mass_at(sp, i) ** 2)
    return 1.0 / (np.exp(np.clip(E / T, 1e-12, 700.0)) - eta[sp])

def fit_T(i):                                              # temperature from the boson bath snapshot
    f = F[BATH_B][i]; E = np.sqrt((q[BATH_B] / a[i]) ** 2 + mass_at(BATH_B, i) ** 2)
    ok = (f > 1e-12) & (f < 1e3)
    slope, _ = np.polyfit(E[ok], np.log(1.0 / f[ok] + 1.0), 1)
    return 1.0 / slope

T = np.array([fit_T(i) for i in range(len(t))]); z = M / T
print(f"z from Phi fits: {z[0]:.3g} .. {z[-1]:.3g}")

def comoving_n(sp, f, ai):
    return dof.get(sp, 1) * np.trapezoid(q[sp] ** 2 * f, q[sp]) / (2 * np.pi ** 2)

def comoving_e(sp, f, ai, i):                              # comoving energy a^4 rho
    E = np.sqrt(q[sp] ** 2 + (ai * mass_at(sp, i)) ** 2)
    return dof.get(sp, 1) * np.trapezoid(q[sp] ** 2 * E * f, q[sp]) / (2 * np.pi ** 2)

nN = np.array([comoving_n(N_SP, F[N_SP][i], a[i]) for i in range(len(t))])
nN_eq = np.array([comoving_n(N_SP, eq_f(N_SP, q[N_SP], a[i], T[i], i), a[i]) for i in range(len(t))])

# (T, mu) of each bath species from its own comoving (n, E), with its mass.
# In units of T: n = g T^3/(2 pi^2) int x^2 dx /(e^{sqrt(x^2 + y^2) - mu/T} - eta),
# E likewise with sqrt(x^2 + y^2) in the integrand, y = m/T. Bosons: mu <= 0.
from scipy.integrate import quad
from scipy.optimize import least_squares
def _mom(k, mu_T, y, e):
    w = lambda x: np.sqrt(x * x + y * y)
    return quad(lambda x: x ** 2 * w(x) ** (k - 2) / (np.exp(w(x) - mu_T) - e), 0.0, 60.0 + max(mu_T, 0.0), limit=200)[0]
def fit_T_mu(sp, i):
    g = dof.get(sp, 1); e = eta[sp]; m = mass_at(sp, i)
    n_, E_ = comoving_n(sp, F[sp][i], a[i]), comoving_e(sp, F[sp][i], a[i], i)
    def res(p):
        Tc, mu_T = p                                       # comoving T; y = m/T = a m / Tc
        y = a[i] * m / Tc
        return [g * Tc ** 3 * _mom(2, mu_T, y, e) / (2 * np.pi ** 2) / n_ - 1.0,
                g * Tc ** 4 * _mom(3, mu_T, y, e) / (2 * np.pi ** 2) / E_ - 1.0]
    upper = 0.0 if e > 0 else np.inf
    sol = least_squares(res, [a[i] * T[i], -1e-6], bounds=([0.0, -np.inf], [np.inf, upper]), xtol=1e-14, ftol=1e-14)
    Tc, mu_T = sol.x
    return Tc / a[i], mu_T                                 # physical T, mu/T
dev, mu_T = {}, {}
for sp in (BATH_B, BATH_F):
    d, m = [], []
    for i in range(len(t)):
        Ts, mts = fit_T_mu(sp, i)
        E = np.sqrt((q[sp] / a[i]) ** 2 + mass_at(sp, i) ** 2)
        feq = 1.0 / (np.exp(np.clip(E / Ts - mts, -700.0, 700.0)) - eta[sp])
        f = F[sp][i]; sel = feq > 1e-6                     # ignore the empty tail
        d.append(np.max(np.abs(f[sel] / feq[sel] - 1.0))); m.append(mts)
    dev[sp], mu_T[sp] = np.array(d), np.array(m)

E_tot = np.array([sum(comoving_e(sp, F[sp][i], a[i], i) for sp in (N_SP, BATH_B, BATH_F)) for i in range(len(t))])
# redshift of the mass terms: dE_com/dt = sum_s g int q^2 f (a m) d(a m)/dt / E_com; a conformal
# mass (m proportional to 1/a) contributes nothing, a constant one H a^2 m^2 / E_com
H = np.gradient(np.log(a), t)
def mass_term(i):
    tot = 0.0
    for sp in (N_SP, BATH_B, BATH_F):
        am = a[i] * mass_at(sp, i)
        am_next = a[min(i + 1, len(t) - 1)] * mass_at(sp, min(i + 1, len(t) - 1))
        am_prev = a[max(i - 1, 0)] * mass_at(sp, max(i - 1, 0))
        dam_dt = (am_next - am_prev) / (t[min(i + 1, len(t) - 1)] - t[max(i - 1, 0)]) if len(t) > 1 else 0.0
        E_com = np.sqrt(q[sp] ** 2 + am ** 2)
        tot += dof.get(sp, 1) * np.trapezoid(q[sp] ** 2 * F[sp][i] * am * dam_dt / E_com, q[sp]) / (2 * np.pi ** 2)
    return tot
src = np.array([mass_term(i) for i in range(len(t))])
E_pred = E_tot[0] + np.concatenate([[0.0], np.cumsum(0.5 * (src[1:] + src[:-1]) * np.diff(t))])

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0, 0].loglog(z, nN, "C0-", lw=2, label=r"$n_N$ (comoving)")
ax[0, 0].loglog(z, nN_eq, "k--", lw=1.5, label=r"$n_N^{\rm eq}$ (FD)")
ax[0, 0].set_xlabel(r"$z = M/T$"); ax[0, 0].set_ylabel("comoving number"); ax[0, 0].legend()
ax[0, 0].set_ylim(max(nN.max(), nN_eq.max()) * 1e-8, max(nN.max(), nN_eq.max()) * 3)

ax[0, 1].semilogx(z, dev[BATH_B], "C1-", label=f"{BATH_B} vs BE(T, mu)")
ax[0, 1].semilogx(z, dev[BATH_F], "C2-", label=f"{BATH_F} vs FD(T, mu)")
ax[0, 1].set_xlabel(r"$z$"); ax[0, 1].set_ylabel(r"max $|f/f_{\rm eq} - 1|$"); ax[0, 1].legend()

ax[1, 0].semilogx(z, E_tot / E_tot[0], "C0-", lw=2, label=r"$E_{\rm tot}$ (comoving)")
ax[1, 0].semilogx(z, E_pred / E_tot[0], "k--", lw=1.5, label="redshift of the mass terms only")
ax[1, 0].set_xlabel(r"$z$"); ax[1, 0].set_ylabel(r"$E/E_0$"); ax[1, 0].legend()

idx = np.unique(np.linspace(0, len(t) - 1, N_SNAP).astype(int))
cols = plt.cm.viridis(np.linspace(0, 1, len(idx)))
for c, i in zip(cols, idx):
    ax[1, 1].loglog(q[N_SP], np.maximum(F[N_SP][i], 1e-30), color=c, label=f"z = {z[i]:.2g}")
    ax[1, 1].loglog(q[N_SP], np.maximum(eq_f(N_SP, q[N_SP], a[i], T[i], i), 1e-30), color=c, ls=":")
ax[1, 1].set_xlabel("comoving q"); ax[1, 1].set_ylabel(r"$f_N$ (dotted: FD)")
ax[1, 1].set_ylim(1e-12, 2); ax[1, 1].legend(fontsize=8, ncol=2)

fig.suptitle(f"N -> {BATH_B} {BATH_F} with a self-consistent bath: M = {M:g}, g_N = {gN:g}")
fig.tight_layout(); fig.savefig("decay_diagnostics.png", dpi=130)
print(f"n_N/n_eq at the end: {nN[-1] / nN_eq[-1]:.3e}; E_tot/E_pred at the end: {E_tot[-1] / E_pred[-1]:.6f}")
print(f"bath mu/T at the end: {BATH_B} {mu_T[BATH_B][-1]:+.2e}, {BATH_F} {mu_T[BATH_F][-1]:+.2e}; "
      f"from N alone both would be about {-nN[-1] / comoving_n(BATH_B, F[BATH_B][0], a[0]):+.1e}")
print("wrote decay_diagnostics.png")
