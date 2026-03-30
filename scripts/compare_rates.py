"""
Compare Vegas vs Analytical collision rates.
Computes C[f](p) once from the same initial distribution using both methods.

Run: mpirun -np <N> python3 compare_rates.py
"""
import numpy as np
import os
from besthep import BEST

# ======================================================================
# Matrix element
# ======================================================================
def matrix_element_constant(momenta, coupling):
    if momenta is not None and momenta.ndim == 3:
        return np.full(momenta.shape[2], coupling**2)
    return coupling**2

# ======================================================================
# Initial condition
# ======================================================================
def init_nonthermal(r, r0=3.0, width=2.0, amplitude=1.0):
    return amplitude / (1 + np.exp((r - r0) / width))

def bose_einstein(r, T=2.0, mu=0.0, mass=1.0):
    E = np.sqrt(r**2 + mass**2)
    x = (E - mu) / T
    if x > 500:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)
# ======================================================================
# Configuration
# ======================================================================
q_min = 0.1
q_max = 50.0
n_grid = 200
coupling = 1.0
mass = 1.0
neval = int(1e6)
delta_width = 0.01

# ======================================================================
# Setup
# ======================================================================
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid)

solver.initialize_species(
    'phi', lambda r: init_nonthermal(r),
    stat='boson', mass=mass)

#solver.initialize_species(
#    'phi', lambda r: bose_einstein(r, T=2.0, mu=0.0, mass=mass),
#    stat='boson', mass=mass)

solver.add_process(
    'phi_2to2', ['phi', 'phi'], ['phi', 'phi'],
    matrix_element_constant,
    coupling=coupling, neval=neval, nitn=2,
    delta_width=delta_width)

# ======================================================================
# Compute rates: Vegas
# ======================================================================
if solver.world_rank == 0:
    print("\n=== Computing Vegas rates ===")

rates_vegas = solver._compute_rates_vegas(
    list(solver.process_configs.keys()), t=0.0)

# ======================================================================
# Compute rates: Analytical
# ======================================================================
if solver.world_rank == 0:
    print("\n=== Computing Analytical rates ===")

rates_anal = solver._compute_rates_all_species('phi_2to2', n_F=n_grid)

# ======================================================================
# Save and plot (rank 0 only)
# ======================================================================
if solver.world_rank == 0:
    ci = solver._analytical_integrators[
        list(solver._analytical_integrators.keys())[0]]
    f_interp = solver.interpolators['phi']



    for sp in solver.species_list:
        r_grid = solver.r_grids[sp]
        rv = rates_vegas[sp]
        ra = rates_anal[sp]

        np.savez(f'compare_{sp}.npz',
                 r_grid=r_grid, vegas=rv, analytical=ra)

        # Print comparison
        print(f"\n{'p':>10} {'C_vegas':>12} {'C_anal':>12} {'ratio':>10}")
        print("-" * 48)
        indices = np.linspace(0, len(r_grid)-1, 15, dtype=int)
        for i in indices:
            ratio = rv[i] / ra[i] if abs(ra[i]) > 1e-20 else float('nan')
            print(f"{r_grid[i]:10.3f} {rv[i]:12.4e} {ra[i]:12.4e} {ratio:10.3f}")

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)

        ax1.loglog(r_grid, np.abs(ra), 'r-', lw=2, label='Analytical')
        ax1.loglog(r_grid, np.abs(rv), 'ko', ms=3, alpha=0.7, label='Vegas MC')
        ax1.set_ylabel('|C[f](p)|')
        ax1.set_title(f'{sp}: Vegas vs Analytical (neval={neval})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        ratio = np.where(np.abs(ra) > 1e-30, rv / ra, np.nan)
        ax2.semilogx(r_grid, ratio, 'ko', ms=3)

#        mask = np.abs(ra) > np.max(np.abs(ra)) * 1e-8
#        if np.any(mask):
#            ax2.semilogx(r_grid[mask], rv[mask] / ra[mask], 'ko', ms=3)
        ax2.axhline(1.0, color='r', ls='--')
        ax2.set_xlabel('|p|')
        ax2.set_ylabel('Vegas / Analytical')
        ax2.set_ylim(0.5, 1.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        outname = f'compare_{sp}.png'
        plt.savefig(outname, dpi=150)
        plt.close()
        print(f"\nSaved {outname}")
