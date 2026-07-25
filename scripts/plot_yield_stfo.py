#!/usr/bin/env python3
"""Yield plot Y = n/s vs x = m1/T from a BESThep PROTOCOL checkpoint.

CONST-dof pipeline: this script pairs with protocol_subthreshold.py and uses
the same constant heff = geff = 104.04 for the entropy -- no dof table. Do
NOT point it at table-dof checkpoints (and do not point the Drees-table plot
script at protocol checkpoints; mixing the two inflates Y by ~20%).

Hardcoding policy: only the IRREDUCIBLE inputs are declared here --
  M1_PHYS_GEV : physical mass of phi1 (code->GeV conversion; not stored in
                the checkpoint), and
  HEFF        : the protocol's constant dof (must equal the run script's).
Everything else -- code-unit masses, statistics, r = m2/m1, a(t), n_com(t),
and the temperature axis -- is derived from the checkpoint. T(t) is fitted
per snapshot from the stored prescribed-bath distribution
(log(1/f+eta) = E_phys/T), so no x_init / anchor constants are duplicated
from the run script.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- irreducible inputs (everything else comes from the checkpoint) ----
CHECKPOINT   = "checkpoint.pkl"
HEFF         = 10.2**2                   # = 104.04; MUST match the run script
M1_PHYS_GEV  = 100.0                     # physical phi1 mass <-> code m1
BATH_SPECIES = "phi2"
OUT          = "yield_Y_of_x.png"
Y_TARGETS    = {"reference fBE": 4.894e-11}                        # optional horizontal reference lines,
                                         # {label: Y value}; empty by default
Y_TEMP_YLIM  = (1.2, 1.6)                # y ("temperature") panel window
                                         # (align with reference code for
                                         # side-by-side comparison); None = auto

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

# ----------------------------------------- entropy (protocol: const dof)
T_GeV  = T_code * conv
s_phys = (2.0 * np.pi ** 2 / 45.0) * HEFF * T_GeV ** 3          # GeV^3

# ------------------------------------------------------------ Y and Yeq
n_phys_GeV = (n_com / a ** 3) * conv ** 3
Y = n_phys_GeV / s_phys

p = np.logspace(-3, 3, 400)                                    # GeV grid
m1_GeV = M1_PHYS_GEV
def n_eq_phys(TG):
    E = np.sqrt(p ** 2 + m1_GeV ** 2)
    return g1 / (2 * np.pi ** 2) * np.trapezoid(
        p ** 2 / (np.exp(np.clip(E / TG, 1e-12, 500)) - eta_1), p)
Y_eq = np.array([n_eq_phys(TG) for TG in T_GeV]) / s_phys


# ------------------------------------------------- y = m * T_chi / s^(2/3)
# T_chi = <p^2/3E> (DRAKE, below Eq. 11 of 2103.01944). Computed from the
# stored phi1 snapshots on the comoving grid: q = a*p, E_com = a*E_phys.
# y_eq uses the SAME moment formula on the equilibrium distribution so that
# grid-truncation systematics cancel between the two curves (in equilibrium
# T_chi = T exactly -- ideal-gas identity P = nT).
if "f" not in history.get(sp, {}):
    raise SystemExit(f"history['{sp}'] has no 'f' snapshots; y panel needs them")
q1 = np.asarray(state["r_grids"][sp], float)

def _T_chi_code(f_arr, ai):
    E_com = np.sqrt(q1 ** 2 + (ai * m1_code) ** 2)
    num = np.trapezoid(f_arr * q1 ** 4 / E_com, q1)
    den = 3.0 * ai * max(np.trapezoid(f_arr * q1 ** 2, q1), 1e-300)
    return num / den                                   # physical, code units

def _y_of(f_arr, ai, T_code_i):
    TG = T_code_i * conv
    s_i = (2.0 * np.pi ** 2 / 45.0) * HEFF * TG ** 3
    return m1_GeV * (_T_chi_code(f_arr, ai) * conv) / s_i ** (2.0 / 3.0)

def _f_eq1(ai, T_code_i):
    E_com = np.sqrt(q1 ** 2 + (ai * m1_code) ** 2)
    return 1.0 / (np.exp(np.clip(E_com / (ai * T_code_i), 1e-12, 700)) - eta_1)

y_dm = np.array([_y_of(np.asarray(history[sp]["f"][i], float), a[i], T_code[i])
                 for i in range(len(t))])
y_eq = np.array([_y_of(_f_eq1(a[i], T_code[i]), a[i], T_code[i])
                 for i in range(len(t))])

# ------------------------------------------------------------------ plot
H   = np.gradient(np.log(a), t)
Gam = np.abs(np.gradient(np.log(np.maximum(n_com, 1e-300)), t))

fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(17.5, 5))

axL.loglog(x, Y, "C0-", lw=2, label=r"$Y$ (phi1)")
axL.loglog(x, Y_eq, "k--", lw=1.5, label=r"$Y_{\rm eq}$")
for lab, yt in Y_TARGETS.items():
    axL.axhline(yt, ls=":", lw=1.2, color="C3" if "fBE" in lab else "gray")
    axL.text(x[0] * 1.05, yt * 1.1, lab, fontsize=8,
             color="C3" if "fBE" in lab else "gray")
axL.set_xlabel(r"$x = m_1/T$")
axL.set_ylabel(r"$Y = n/s$")
axL.legend()

axM.semilogx(x, y_dm, "C0-", lw=2, label=r"$y$ (phi1)")
axM.semilogx(x, y_eq, "k--", lw=1.5, label=r"$y_{\rm eq}$")
axM.set_xlabel(r"$x = m_1/T$")
axM.set_ylabel(r"$y = m_1 T_\chi\, s^{-2/3}$")
axM.legend()
if Y_TEMP_YLIM is not None:
    axM.set_ylim(*Y_TEMP_YLIM)

axR.loglog(x, H,   "k-",  lw=2, label=r"$H$")
axR.loglog(x, Gam, "C0-", lw=2,
           label=r"$\Gamma_{\rm net}=|\mathrm{d}\ln n_{\rm com}/\mathrm{d}t|$")
axR.set_xlabel(r"$x = m_1/T$")
axR.set_ylabel(r"rate  $[t^{-1}]$")
axRb = axR.twinx()
axRb.loglog(x, a, "C3--", lw=1.5, label=r"$a$")
axRb.set_ylabel(r"$a$")
h1, l1 = axR.get_legend_handles_labels()
h2, l2 = axRb.get_legend_handles_labels()
axR.legend(h1 + h2, l1 + l2, fontsize=9, loc="center left")

fig.suptitle(f"Sub-threshold freeze-out (r = {r_ratio:.2g}), "
             f"m1 = {M1_PHYS_GEV:g} GeV, const dof")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"Y(x_end) = {Y[-1]:.4e}   y(x_end) = {y_dm[-1]:.6f}")
print(f"wrote {OUT}")