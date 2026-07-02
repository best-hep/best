#!/usr/bin/env python3
"""Yield plot Y = n/s vs x = m1/T from a BESThep checkpoint.

Hardcoding policy: only the IRREDUCIBLE inputs are declared here --
  M1_PHYS_GEV : physical mass of phi1 (defines code->GeV conversion; this
                information exists nowhere in the checkpoint), and
  DOF_FILE    : the dof table (data needed for the SM entropy density).
Everything else -- code-unit masses, statistics, r = m2/m1, a(t), n_com(t),
and the temperature axis itself -- is derived from the checkpoint. T(t) is
extracted per snapshot by fitting the stored prescribed-bath distribution
(log(1/f+eta) = E_phys/T), so no x_init / anchor constants are duplicated
from the run script (the past x_init 2-vs-5 accident is structurally
impossible here).

Because T now comes from an INDEPENDENT source (the bath fit, not entropy
inversion), the printed s*a^3 spread is a genuine consistency check of
cosmology + fit + unit conversion, not a tautology.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- irreducible inputs (everything else comes from the checkpoint) ----
CHECKPOINT   = "checkpoint.pkl"
DOF_FILE     = "dof_Drees_etal.dat"     # T[GeV]  heff  sqrt(geff)  sqrt(g*)
M1_PHYS_GEV  = 100.0                     # physical phi1 mass <-> code m1
BATH_SPECIES = "phi2"
OUT          = "yield_Y_of_x.png"

# ---------------------------------------------------------------- load
try:
    with open(CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Unpickling needs module '{e.name}' (checkpoint stores vegas "
        f"integrators); run this where the solver environment is available.")

history = state["history"]
masses  = state["species_mass"]
configs = state["species_config"]
dofs    = state.get("species_dof", {}) or {}

t  = np.asarray(history["times"], float)
a  = np.asarray(history["a"], float)
sp = "phi1"
n_com = np.asarray(history[sp]["n"], float)          # code units, comoving
m1_code = float(masses[sp])
m_bath  = float(masses[BATH_SPECIES])
g1      = float(dofs.get(sp, 1.0))
conv    = M1_PHYS_GEV / m1_code                       # code energy -> GeV
r_ratio = m_bath / m1_code
eta_b   = 1.0 if configs.get(BATH_SPECIES, "boson") == "boson" else -1.0
eta_1   = 1.0 if configs.get(sp, "boson") == "boson" else -1.0

# ------------------------- T(t) from per-snapshot bath fits (code units)
q_b = np.asarray(state["r_grids"][BATH_SPECIES], float)
def fit_T(idx):
    f = np.asarray(history[BATH_SPECIES]["f"][idx], float)
    E = np.sqrt((q_b / a[idx]) ** 2 + m_bath ** 2)
    ok = np.isfinite(f) & (f > 1e-290) & (f < 1e3)
    if ok.sum() < 4:
        raise RuntimeError(f"snapshot {idx}: too few usable bath points")
    slope, _ = np.polyfit(E[ok], np.log(1.0 / f[ok] + eta_b), 1)
    if slope <= 0:
        raise RuntimeError(f"snapshot {idx}: non-positive 1/T slope")
    return 1.0 / slope

T_code = np.array([fit_T(i) for i in range(len(t))])
x = m1_code / T_code
print(f"x = m1/T from bath fits: {x[0]:.3g} .. {x[-1]:.3g}  "
      f"(r = m_bath/m1 = {r_ratio:.3g})")

# --------------------------------------- SM entropy from the dof table
tab = np.loadtxt(DOF_FILE, skiprows=1)
T_tab, heff_tab = tab[:, 0], tab[:, 1]
lo, hi = T_tab.min(), T_tab.max()
T_GeV = T_code * conv
if T_GeV.min() < lo or T_GeV.max() > hi:
    raise SystemExit(f"T range {T_GeV.min():.3g}..{T_GeV.max():.3g} GeV "
                     f"outside dof table [{lo:.3g}, {hi:.3g}] -- check "
                     f"M1_PHYS_GEV.")
order = np.argsort(T_tab)
heff = np.interp(T_GeV, T_tab[order], heff_tab[order])
s_phys = (2.0 * np.pi ** 2 / 45.0) * heff * T_GeV ** 3          # GeV^3

# genuine consistency check: comoving entropy must be constant
s_com = s_phys * a ** 3
spread = (s_com.max() - s_com.min()) / s_com.mean()
print(f"s*a^3 spread: {spread:.2e}  "
      f"(independent check of cosmology + bath fit + conversion)")

# ------------------------------------------------------------ Y and Yeq
n_phys_GeV = (n_com / a ** 3) * conv ** 3
Y = n_phys_GeV / s_phys

p = np.logspace(-3, 2, 400)                                    # GeV grid
m1_GeV = M1_PHYS_GEV
def n_eq_phys(TG):
    E = np.sqrt(p ** 2 + m1_GeV ** 2)
    return g1 / (2 * np.pi ** 2) * np.trapezoid(
        p ** 2 / (np.exp(np.clip(E / TG, 1e-12, 500)) - eta_1), p)
Y_eq = np.array([n_eq_phys(TG) for TG in T_GeV]) / s_phys

# ------------------------------------------------------------------ plot
# Left: yield vs equilibrium. Right: the WHY -- expansion vs realized net
# chemical rate on the SAME x axis, so the Gamma_net = H crossing aligns
# with the departure of Y from Y_eq. Twin axis: a(t) and heff(T) (the dof
# variation; note T itself vs x = m1/T would be a tautology).
H   = np.gradient(np.log(a), t)
Gam = np.abs(np.gradient(np.log(np.maximum(n_com, 1e-300)), t))
below = np.nonzero(Gam < H)[0]
x_fo = x[below[0]] if below.size else None

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5))

axL.loglog(x, Y, "C0-", lw=2, label=r"$Y$ (phi1)")
axL.loglog(x, Y_eq, "k--", lw=1.5, label=r"$Y_{\rm eq}$")
axL.set_xlabel(r"$x = m_1/T$")
axL.set_ylabel(r"$Y = n/s$")
axL.legend()

axR.loglog(x, H,   "k-",  lw=2, label=r"$H$")
axR.loglog(x, Gam, "C0-", lw=2,
           label=r"$\Gamma_{\rm net}=|\mathrm{d}\ln n_{\rm com}/\mathrm{d}t|$")
axR.set_xlabel(r"$x = m_1/T$")
axR.set_ylabel(r"rate  $[t^{-1}]$")
axRb = axR.twinx()
axRb.loglog(x, a,    "C3--", lw=1.5, label=r"$a$")
axRb.loglog(x, heff, "C2-.", lw=1.5, label=r"$h_{\rm eff}(T)$")
axRb.set_ylabel(r"$a$,  $h_{\rm eff}$")
h1, l1 = axR.get_legend_handles_labels()
h2, l2 = axRb.get_legend_handles_labels()
axR.legend(h1 + h2, l1 + l2, fontsize=9, loc="center left")

if x_fo is not None:
    for axx in (axL, axR):
        axx.axvline(x_fo, color="gray", ls=":", lw=1.5)
    axR.text(x_fo, axR.get_ylim()[0] * 3, r"  $\Gamma_{\rm net}=H$",
             fontsize=9, color="gray")

fig.suptitle(f"Sub-threshold freeze-out (r = {r_ratio:.2g}), "
             f"m1 = {M1_PHYS_GEV:g} GeV")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}   Y_final = {Y[-1]:.4e}"
      + (f"   x_freeze-out = {x_fo:.3g}" if x_fo is not None else ""))
