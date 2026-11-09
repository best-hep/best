#!/usr/bin/env python3
"""
A condensate decaying into a boson that condenses (examples/inflaton_bec.py), from its checkpoint.
Left: particles (solid, on the grid) and condensate (dash-dot, the zero mode) of every species, the
parent against its initial number, each daughter against the number the decays deliver (the decay's
multiplicity times the parent's initial number). Then one panel per species with its occupation
f(p) over the run; the condensate is the short dash-dot segment below q_min at the occupation of the
zero mode, N0/V0 (V0 the states of p < q_min). Dashed black: the Bose-Einstein distribution at the
temperature fitted to the daughter's final grid, with mu = m for the daughter (the critical line
its particles settle on) and mu = the daughters' masses for the parent (chemical equilibrium with
them); p* marks where the decay puts the daughters. Line styles and the colours of the snapshots
are those of scripts/plot.py.
Usage: python3 scripts/plot_bec.py [checkpoint.pkl]  ->  bec_diagnostics.png
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import grid_weights

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint.pkl'
with open(CHECKPOINT, 'rb') as fh:
    state = pickle.load(fh)
h = state['history']
t = np.array(h['times'])
dec = next(c for c in state['process_configs'].values() if len(c['input']) == 1 and len(c['output']) == 2)
parent, daughters = dec['input'][0], list(dict.fromkeys(dec['output']))
species = [parent] + daughters
n_parent0 = h[parent]['n'][0]                                   # particles and condensate at the start
norm = {sp: (dec['output'].count(sp) if sp != parent else 1) * n_parent0 for sp in species}
mass = {sp: h[sp]['m'][-1] for sp in species}
n0 = lambda sp: np.array(h[sp]['n0'])
grid = lambda sp: np.array(h[sp]['n']) - n0(sp)
color = dict(zip(species, ('k', 'C0', 'C3')))

fig, axes = plt.subplots(1, 1 + len(species), figsize=(5.5 * (1 + len(species)), 5))
ax = axes[0]
for sp in species:
    unit = '' if sp == parent else f'{dec["output"].count(sp)} '
    ax.plot(t, grid(sp) / norm[sp], '-', color=color[sp], lw=2, label=f'{sp} particles / {unit}N$_{{{parent}}}$(0)')
    ax.plot(t, n0(sp) / norm[sp], '-.', color=color[sp], lw=2, label=f'{sp} condensate / {unit}N$_{{{parent}}}$(0)')
ax.set_xlabel('t'); ax.set_ylabel('number'); ax.set_ylim(bottom=0.0)
ax.grid(alpha=0.3); ax.legend()

sp_d = daughters[0]                                             # the temperature from the daughter's final grid
q_d = state['r_grids'][sp_d]; E_d = np.sqrt(q_d**2 + mass[sp_d]**2); f_d = np.array(h[sp_d]['f'][-1])
b, a = np.polyfit(E_d, np.log1p(1.0 / f_d), 1, w=np.sqrt(grid_weights(q_d) * q_d**2 * f_d))   # log(1 + 1/f) = (E - mu)/T, each node by its number
T = 1.0 / b
mu = {sp: mass[sp] for sp in daughters}; mu[parent] = sum(mass[s] for s in dec['output'])
m_o = mass[daughters[-1]]
p_star = np.sqrt(((mass[parent]**2 + mass[sp_d]**2 - m_o**2) / (2.0 * mass[parent]))**2 - mass[sp_d]**2)
picks = np.unique(np.linspace(0, len(t) - 1, 8).astype(int))[1:]
colors = plt.cm.rainbow_r(np.linspace(0.0, 1.0, len(picks)))
q_min = state['q_min']
for ax, sp in zip(axes[1:], species):
    q = state['r_grids'][sp]; E = np.sqrt(q * q + mass[sp]**2)
    V0 = state['species_dof'].get(sp, 1) * q_min**3 / (6.0 * np.pi**2)
    for i, c in zip(picks, colors):
        ax.plot(q, h[sp]['f'][i], '-', color=c, lw=1.2, label=f't = {t[i]:.3g}' if sp == parent else None)
        ax.plot([q_min / 4.0, q_min], [n0(sp)[i] / V0] * 2, '-.', color=c, lw=2)
    ax.plot(q, 1.0 / np.expm1(np.maximum(E - mu[sp], 1e-300) / T), 'k--', lw=2,
            label=f'Bose-Einstein, T = {T:.3g}, $\\mu$ = ' + ('m' if sp != parent else ' + '.join(f'm$_{{{s}}}$' for s in dec['output'])))
    ax.axvline(p_star, color='gray', ls=':', lw=1)
    f_end = np.array(h[sp]['f'][-1]); top = max(f_end.max(), n0(sp).max() / V0)
    ax.text(p_star, top, ' p*', va='top', color='gray')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('|p|'); ax.set_ylabel(f'f$_{{{sp}}}$(p)')
    ax.set_ylim(1e-6, 3.0 * top); ax.set_title(f'{sp}: particles (solid), condensate below q$_{{min}}$ (dash-dot)', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
out = 'bec_diagnostics.png'
plt.tight_layout(); plt.savefig(out, dpi=150)
print(f'condensate of {sp_d}: {n0(sp_d)[-1] / norm[sp_d]:.4%} of the delivered number; fitted T = {T:.4f}; saved {out}')
