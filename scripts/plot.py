#!/usr/bin/env python3
"""
Plot evolution with equilibrium overlay from BEST checkpoint.
Interactive mode: prompts for all options.
Usage: python plot_evolution.py [checkpoint_file]
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from scipy.integrate import quad
import pickle
import sys
import os
import glob
import warnings

plt.rcParams.update({'axes.labelsize': 16})

# ======================================================================
# Equilibrium distributions
# ======================================================================
def energy(p, m):
    return np.sqrt(p**2 + m**2)


def be_dist(p, T, mu, m):
    E = energy(p, m)
    x = (E - mu) / T
    x = np.clip(x, -500, 500)
    return 1.0 / (np.exp(x) - 1.0 + 1e-30)


def fd_dist(p, T, mu, m):
    E = energy(p, m)
    x = (E - mu) / T
    x = np.clip(x, -500, 500)
    return 1.0 / (np.exp(x) + 1.0)


# ======================================================================
# Predict equilibrium T (and mu) from conservation laws
# ======================================================================
def predict_equilibrium(E0, N0, mass, stat, q_min, q_max, number_changing):
    if stat == 'boson':
        dist = lambda p, T, mu: be_dist(p, T, mu, mass)
    else:
        dist = lambda p, T, mu: fd_dist(p, T, mu, mass)

    def E_integral(T, mu):
        def integrand(p):
            return p**2 * energy(p, mass) * dist(p, T, mu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return 4 * np.pi * quad(integrand, q_min, q_max, limit=200)[0]

    def N_integral(T, mu):
        def integrand(p):
            return p**2 * dist(p, T, mu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return 4 * np.pi * quad(integrand, q_min, q_max, limit=200)[0]

    if number_changing:
        try:
            T_eq = brentq(lambda T: E_integral(T, 0.0) - E0, 0.01, 50.0)
            return T_eq, 0.0
        except Exception as e:
            print(f"  Equilibrium prediction failed: {e}")
            return None, None
    else:
        try:
            def eqs(params):
                T, mu = params
                if T <= 0 or (stat == 'boson' and mu >= (mass if mass > 0 else 0)):
                    return [1e10, 1e10]
                return [N_integral(T, mu) - N0, E_integral(T, mu) - E0]
            T_eq, mu_eq = fsolve(eqs, [2.0, -0.5])
            return T_eq, mu_eq
        except Exception as e:
            print(f"  Equilibrium prediction failed: {e}")
            return None, None


# ======================================================================
# Load checkpoint
# ======================================================================
def load_checkpoint(filename):
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found")
        return None, None
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    history = state.get('history', None)
    if history:
        n_snap = len(history['times'])
        t0 = history['times'][0] if n_snap > 0 else 0
        t1 = history['times'][-1] if n_snap > 0 else 0
        print(f"  Snapshots: {n_snap}")
        print(f"  Time: {t0:.3e} ~ {t1:.3e}")
        print(f"  Step: {state.get('step_count', '?')}")
    return state, history


# ======================================================================
# r_grid fallback
# ======================================================================
def get_r_grid(history, species, snapshot_idx, state):
    """Get r_grid with fallback for old checkpoints."""
    if 'r_grid' in history[species]:
        rg = history[species]['r_grid']
        return np.array(rg[snapshot_idx] if isinstance(rg, list) else rg)
    if 'r_grids' in state and species in state['r_grids']:
        return np.array(state['r_grids'][species])
    if 'q_grid' in history[species]:
        return np.array(history[species]['q_grid'])
    n_pts = len(history[species]['f'][snapshot_idx])
    return np.linspace(0.1, 50.0, n_pts)


# ======================================================================
# Select snapshots uniformly in time
# ======================================================================
def select_snapshots(times, n_lines, max_snapshot):
    if max_snapshot is not None:
        max_snapshot = min(max_snapshot, len(times) - 1)
    else:
        max_snapshot = len(times) - 1

    times_cut = times[:max_snapshot + 1]
    t_targets = np.linspace(times_cut[0], times_cut[-1], n_lines)
    indices = []
    for t in t_targets:
        idx = int(np.argmin(np.abs(times_cut - t)))
        if idx not in indices:
            indices.append(idx)
        else:
            diffs = np.abs(times_cut - t)
            for candidate in np.argsort(diffs):
                if int(candidate) not in indices:
                    indices.append(int(candidate))
                    break
    indices.sort()
    return indices, max_snapshot


# ======================================================================
# Main plot
# ======================================================================
def plot_evolution(state, history, output_file='evolution.png',
                   n_lines=5, max_snapshot=None, ymode='fp'):
    if history is None or len(history['times']) == 0:
        print("No history data")
        return

    species_config = state['species_config']
    species_mass = state.get('species_mass', {})
    process_configs = state.get('process_configs', {})
    q_min = state['q_min']
    q_max = state['q_max']
    times = np.array(history['times'])
    species_list = [k for k in history.keys() if k != 'times']

    if max_snapshot is not None:
        max_snap = min(max_snapshot, len(times) - 1)
    else:
        max_snap = len(times) - 1
    all_indices = list(range(max_snap + 1))
    label_indices, _ = select_snapshots(times, n_lines, max_snapshot)
    label_set = set(label_indices)
    times_cut = times[:max_snap + 1]

    # Detect number-changing processes
    number_changing = {}
    for sp in species_list:
        number_changing[sp] = False
        for pconf in process_configs.values():
            all_sp = pconf['input'] + pconf['output']
            if sp in all_sp and pconf['n_in'] != pconf['n_out']:
                number_changing[sp] = True
                break

    for sp in species_list:
        mass = species_mass.get(sp, 0.0)
        stat = species_config.get(sp, 'boson')
        N0 = history[sp]['n'][0]
        E0 = history[sp]['e'][0]

        # Predict equilibrium
        print(f"\n{sp} ({stat}, m={mass}):")
        T_eq, mu_eq = predict_equilibrium(
            E0, N0, mass, stat, q_min, q_max, number_changing[sp])
        if T_eq is not None:
            print(f"  Predicted: T_eq = {T_eq:.4f}, mu_eq = {mu_eq:.4f}")

        # --- Figure ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: distribution evolution
        colors = plt.cm.rainbow_r(np.linspace(0.0, 1.0, len(all_indices)))

        for j, i in enumerate(all_indices):
            f = np.array(history[sp]['f'][i])
            r_grid = get_r_grid(history, sp, i, state)
            label = f't = {times[i]:.1e}' if i in label_set else None

            if ymode == 'p2f':
                ax1.semilogy(r_grid, r_grid**2 * f, color=colors[j],
                             alpha=0.8, lw=1.5, label=label)
            else:
                ax1.semilogy(r_grid, f, color=colors[j],
                             alpha=0.8, lw=1.5, label=label)

        # Equilibrium overlay
        if T_eq is not None:
            r_smooth = np.logspace(np.log10(q_min), np.log10(q_max), 300)
            if stat == 'boson':
                f_eq = be_dist(r_smooth, T_eq, mu_eq, mass)
            else:
                f_eq = fd_dist(r_smooth, T_eq, mu_eq, mass)
            mu_str = f', μ={mu_eq:.2f}' if not number_changing[sp] else ''

            if ymode == 'p2f':
                ax1.semilogy(r_smooth, r_smooth**2 * f_eq, 'k--', lw=2,
                             label=f'BE (T={T_eq:.2f}{mu_str})')
            else:
                ax1.semilogy(r_smooth, f_eq, 'k--', lw=2,
                             label=f'BE (T={T_eq:.2f}{mu_str})')

        ax1.set_xlabel('|p|')
        ax1.set_ylabel(r'$p^2 f(p)$' if ymode == 'p2f' else r'$f(p)$')
#        ax1.set_title(f'{sp} distribution ({stat}, m={mass})')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=1e-6)

        # Right: conservation check
        n_hist = np.array(history[sp]['n'][:max_snap + 1])
        e_hist = np.array(history[sp]['e'][:max_snap + 1])

        ax2.plot(times_cut, n_hist / N0, '-', lw=2, label='N/N₀')
        ax2.plot(times_cut, e_hist / E0, '--', lw=2, label='E/E₀')
        ax2.axhline(1.0, color='gray', ls=':', alpha=0.5)
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel('Ratio to initial')
#        ax2.set_title(f'{sp} conservation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        outname = output_file.replace('.png', f'_{sp}.png')
        plt.savefig(outname, dpi=150)
        plt.close()
        print(f"  Saved {outname}")
        print(f"  Total snapshots plotted: {len(all_indices)}")
        print(f"  Labeled: {label_indices}")
        print(f"  Label times: {[f'{times[i]:.3e}' for i in label_indices]}")


# ======================================================================
# Interactive prompt
# ======================================================================
def prompt(msg, default):
    val = input(f"{msg} [{default}]: ").strip()
    return val if val else str(default)


def main():
    if len(sys.argv) > 1:
        ckpt_file = sys.argv[1]
    else:
        pkls = sorted(glob.glob('*.pkl'))
        if pkls:
            print("Found checkpoints:")
            for i, p in enumerate(pkls):
                print(f"  {i}: {p}")
            choice = prompt("Select number or filename", pkls[-1])
            ckpt_file = pkls[int(choice)] if choice.isdigit() else choice
        else:
            ckpt_file = prompt("Checkpoint file", "checkpoint.pkl")

    print(f"\nLoading: {ckpt_file}")
    state, history = load_checkpoint(ckpt_file)
    if history is None:
        return

    n_snap = len(history['times'])
    output_file = prompt("Output filename", "evolution.png")
    n_lines = int(prompt("Number of labeled lines", 5))
    max_input = prompt("Max snapshot index (enter for all)", n_snap - 1)
    max_snapshot = int(max_input)
    ymode = prompt("Y-axis: fp or p2f", "fp")

    plot_evolution(state, history, output_file, n_lines, max_snapshot, ymode)


if __name__ == "__main__":
    main()
