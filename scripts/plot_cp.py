#!/usr/bin/env python3
"""Plots for the CP run (examples/Mdecay_cp.py) from its cp_K*.npz files.

Panels: (1) comoving N number vs z = M/T, the run and the integrated reference D1;
(2) the asymmetry |n_l - n_lbar| / n_gamma, run and D1 (the dip is the sign
flip); (3) delta f / f_l at the end of the run against q / T_0, over its mean
n_asym / n_l: 1 where the asymmetry sits in the thermal shape (kinetic
equilibrium). One colour per file.

Usage: python plot_cp.py [cp_K10_mb.npz cp_K0.1_mb.npz ...]   (default: every cp_K*.npz here)
"""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = sys.argv[1:] or sorted(glob.glob("cp_K*.npz"))
if not files:
    sys.exit("no cp_K*.npz file")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for i, name in enumerate(files):
    d = np.load(name)
    z, K = d["z"], float(d["K"])
    c = f"C{i}"
    label = f"K = {K:g}" + (" (MB)" if name.endswith("_mb.npz") else " (quantum)")
    ax[0].plot(z[1:], d["N_N"][1:], c, label=label)          # the first record is the empty start
    ax[0].plot(z[1:], d["N_N_D1"][1:], c, ls="--", lw=1)
    ax[1].plot(z[1:], np.abs(d["N_asym"][1:]), c, label=label)
    ax[1].plot(z[1:], np.abs(d["N_asym_D1"][1:]), c, ls="--", lw=1)
    q, fl, flb = d["q"], d["f_l"], d["f_lbar"]
    x = q * z[0]                                   # q / T_0: the comoving grid in units of the initial temperature (M = 1)
    w = np.zeros_like(q); w[:-1] += np.diff(q) / 2; w[1:] += np.diff(q) / 2; w *= q * q   # trapezoid weights of the number
    m = (w @ (fl - flb)) / (w @ fl)                # n_asym / n_l
    ax[2].plot(x, (fl - flb) / fl / m, c, label=label)

ax[0].set(xscale="log", yscale="log", xlabel="z = M/T", ylabel=r"$N_N$ (per photon)", title="N: run (solid), D1 (dashed)")
ax[1].set(xscale="log", yscale="log", xlabel="z = M/T", ylabel=r"$|n_l - n_{\bar l}| / n_\gamma$", title="asymmetry: run (solid), D1 (dashed)")
ax[2].set(xscale="log", xlabel=r"$q / T_0$", ylabel=r"$\delta f / f_l$ at the end, over $n_{asym} / n_l$", title="shape of the asymmetry (1: thermal)")
for a in ax:
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig("cp_run.png", dpi=130)
print("wrote cp_run.png")
