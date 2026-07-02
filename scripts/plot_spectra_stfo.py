#!/usr/bin/env python3
"""Evolution plots from a BESThep checkpoint (one figure per species).

One panel per species: f(q) snapshots over time (comoving grid) + BE
overlay at the final bath temperature. (The H vs Gamma_net rate panel
lives in plot_yield.py, on the shared x = m1/T axis.)

Design rule: EVERYTHING (grids, masses, statistics, temperature, cosmology)
is derived from the checkpoint itself -- there are no constants here that
must be kept in sync with the run script. The temperature is extracted by
fitting the prescribed bath distribution (BATH_SPECIES) at the last
snapshot: for a Bose-Einstein f, log(1/f + 1) = E_phys / T is linear in
E_phys with slope 1/T.

"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

CHECKPOINT   = "checkpoint.pkl"
BATH_SPECIES = "phi2"      # species whose stored f defines T (loud KeyError if wrong)
MAX_CURVES   = 40          # max snapshot curves on the left panel
F_FLOOR      = 1e-30       # lower y-limit for f plots

# ---------------------------------------------------------------- load
try:
    with open(CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Unpickling needs module '{e.name}' (checkpoint stores vegas "
        f"integrators); run this where the solver environment is available.")

history  = state["history"]
r_grids  = state["r_grids"]
masses   = state["species_mass"]
configs  = state["species_config"]          # 'boson' / 'fermion'

t = np.asarray(history["times"], float)
a = np.asarray(history["a"], float)
species_list = [k for k in history if k not in ("times", "a")]
print(f"checkpoint: {len(t)} records, t = {t[0]:.3e} .. {t[-1]:.3e}, "
      f"species = {species_list}")

# ------------------------------------------------- bath temperature fit
def extract_T(sp, idx=-1):
    """Fit log(1/f + eta) = E_phys/T on the stored snapshot of species sp."""
    q    = np.asarray(r_grids[sp], float)
    f    = np.asarray(history[sp]["f"][idx], float)
    m    = float(masses[sp])
    eta  = 1.0 if configs.get(sp, "boson") == "boson" else -1.0
    E    = np.sqrt((q / a[idx]) ** 2 + m ** 2)
    ok   = np.isfinite(f) & (f > 1e-290) & (f < 1e3)
    if ok.sum() < 4:
        raise RuntimeError(f"too few usable points to fit T from '{sp}'")
    slope, _ = np.polyfit(E[ok], np.log(1.0 / f[ok] + eta), 1)
    if slope <= 0:
        raise RuntimeError(f"non-positive 1/T slope fitting '{sp}' -- is it a bath?")
    return 1.0 / slope

T_fin = extract_T(BATH_SPECIES)
print(f"T(final) extracted from '{BATH_SPECIES}': {T_fin:.4g} (code units)")

# ---------------------------------------------------------------- plots
for sp in species_list:
    q    = np.asarray(r_grids[sp], float)
    fs   = history[sp]["f"]
    n    = np.asarray(history[sp]["n"], float)
    m    = float(masses[sp])
    eta  = 1.0 if configs.get(sp, "boson") == "boson" else -1.0

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # ---- left: f(q) snapshots (early = red, late = purple) ----
    idxs = np.unique(np.linspace(0, len(fs) - 1, MAX_CURVES).astype(int))
    for j, i in enumerate(idxs):
        col = cm.rainbow(1.0 - j / max(len(idxs) - 1, 1))
        lab = None
        if j % max(len(idxs) // 5, 1) == 0:
            lab = f"t = {t[i]:.1e}"
        ax1.plot(q, np.asarray(fs[i], float),
                 color=col, lw=1.0, label=lab)
    # BE overlay at final bath temperature, this species' mass, final a
    E_fin = np.sqrt((q / a[-1]) ** 2 + m ** 2)
    f_be  = 1.0 / (np.exp(np.clip(E_fin / T_fin, 1e-12, 500)) - eta)
    ax1.plot(q, f_be, "k--", lw=2,
             label=f"BE (T={T_fin:.2g})")
    ax1.set_yscale("log")
    ax1.set_ylim(F_FLOOR, None)
    ax1.set_xlabel(r"$q$  (comoving)")
    ax1.set_ylabel(r"$f(q)$")
    ax1.legend(fontsize=8)

    fig.suptitle(f"{sp}   (m = {m:g})")
    fig.tight_layout()
    out = f"evolution_{sp}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
