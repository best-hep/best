"""
BESThep: Boltzmann Equation Solver for Thermalization (hep)

A general-purpose solver for momentum-resolved Boltzmann equations
using Vegas Monte Carlo integration. Supports arbitrary n->m processes.

Usage:
    from besthep import BEST
    solver = BEST(q_min=0.1, q_max=20.0)
    solver.initialize_species('phi', my_init_func, stat='boson', mass=1.0)
    solver.add_process('my_process', ['phi','phi'], ['phi','phi','phi'],
                       my_matrix_element, coupling=1.0, neval=1e6)
    solver.evolve_step(dt=1.0)
"""
import numpy as np
from mpi4py import MPI
from scipy.integrate import quad
from scipy.interpolate import interp1d
import time
import os
import pickle
import warnings

# ======================================================================
# Interpolation with power-law / exponential extrapolation
# ======================================================================
class ExtrapolatingInterp:
    def __init__(self, r_grid, f, mass=0.0, stat='boson'):
        self.mass = mass
        # eta: +1 boson, -1 fermion, 0 Maxwell-Boltzmann
        if stat == 'boson':
            self.eta = 1.0
        elif stat == 'fermion':
            self.eta = -1.0
        else:
            self.eta = 0.0

        # Energy grid: E = sqrt(p^2 + m^2)
        E_grid = np.sqrt(r_grid**2 + mass**2)

        # Exact form: log(1/f + eta) = a + b*E
        #   boson:  f = 1/(exp(y) - 1)
        #   fermion: f = 1/(exp(y) + 1)
        #   MB:     f = exp(-y)
        f_safe = np.maximum(f, 1e-30)
        y = np.log(1.0 / f_safe + self.eta)

        # Low-p fit
        n_low = max(10, len(r_grid) // 10)
        self.b_low, self.a_low = np.polyfit(E_grid[:n_low], y[:n_low], 1)

        # High-p fit
        n_high = max(10, len(r_grid) // 10)
        self.b_high, self.a_high = np.polyfit(E_grid[-n_high:], y[-n_high:], 1)

        kind = 'cubic' if len(r_grid) >= 4 else 'linear'
        self._interp = interp1d(r_grid, f, kind=kind, bounds_error=False,
                                fill_value=(f[0], f[-1]))
        self.q_min = r_grid[0]
        self.q_max = r_grid[-1]

    def _energy(self, p):
        return np.sqrt(p**2 + self.mass**2)

    def _f_from_y(self, y):
        y = np.clip(y, -500, 500)
        if self.eta == 0.0:
            return np.exp(-y)
        if self.eta == -1.0:  # fermion
            return 1.0 / (np.exp(y) + 1.0)
        # boson
        return np.where(y > 0, 1.0 / (np.exp(y) - 1.0 + 1e-30), 0.0)


    def __call__(self, p):
        if isinstance(p, (int, float)):
            if p < self.q_min:
                E = self._energy(p)
                y = self.a_low + self.b_low * E
                return max(0.0, float(self._f_from_y(np.array([y]))[0]))
            elif p > self.q_max:
                E = self._energy(p)
                y = self.a_high + self.b_high * E
                return max(0.0, float(self._f_from_y(np.array([y]))[0]))
            else:
                return float(self._interp(p))

        p = np.atleast_1d(np.asarray(p, dtype=float))
        result = np.empty_like(p)

        mask_low = p < self.q_min
        mask_high = p > self.q_max
        mask_mid = ~mask_low & ~mask_high

        if np.any(mask_mid):
            result[mask_mid] = self._interp(p[mask_mid])
        if np.any(mask_low):
            E_low = self._energy(p[mask_low])
            y = self.a_low + self.b_low * E_low
            result[mask_low] = self._f_from_y(y)
        if np.any(mask_high):
            E_high = self._energy(p[mask_high])
            y = self.a_high + self.b_high * E_high
            result[mask_high] = self._f_from_y(y)

        result = np.maximum(result, 0.0)
        return float(result[0]) if result.size == 1 else result


# ======================================================================
# Time integration helpers
# ======================================================================
def euler_update(f, rates, dt):
    f_safe = np.maximum(f, 1e-30)
    log_f = np.log(f_safe)
    log_f += dt * rates / f_safe
    return np.exp(log_f)


def heun_update(f_orig, k1, k2, dt, f_pred=None):
    f0 = np.maximum(f_orig, 1e-30)
    fp = np.maximum(f_pred, 1e-30) if f_pred is not None else f0
    log_f = np.log(f0)
    log_f += dt * 0.5 * (k1 / f0 + k2 / fp)
    return np.exp(log_f)


# ======================================================================
# Main solver
# ======================================================================
class BEST:
    """Boltzmann Equation Solver for Thermalization
    using Vegas Monte Carlo integration."""

    # Default numerical cutoffs (can be overridden per-instance)
    cutoff_zero = 1e-50
    cutoff_energy_min = 1e-50

    def __init__(self, q_min, q_max, n_grid=500, n_r_parallel=None):
        # ---- MPI setup ----
        self.world_comm = MPI.COMM_WORLD
        self.world_rank = self.world_comm.Get_rank()
        self.world_size = self.world_comm.Get_size()

        # Default: one r-point per MPI rank (no sub-grouping)
        if n_r_parallel is None:
            n_r_parallel = self.world_size
        self.n_r_parallel = min(n_r_parallel, self.world_size)

        ranks_per_group = self.world_size // self.n_r_parallel
        if ranks_per_group < 1:
            ranks_per_group = 1
            self.n_r_parallel = self.world_size
        self.ranks_per_group = ranks_per_group

        self.color = self.world_rank // ranks_per_group
        self.sub_comm = self.world_comm.Split(self.color, self.world_rank)
        self.sub_rank = self.sub_comm.Get_rank()
        self.sub_size = self.sub_comm.Get_size()

        # Patch MPI.COMM_WORLD so vegas uses sub_comm
        import mpi4py.MPI
        mpi4py.MPI.COMM_WORLD = self.sub_comm
        import vegas
        self._vegas = vegas

        # ---- Physics ----
        self.species_config = {}   # {name: 'boson'/'fermion'}
        self.species_mass = {}     # {name: mass} (0.0 = massless)
        self.species_mass_func = {}  # {name: func(t) -> mass}
        self.q_min = q_min
        self.q_max = q_max
        self.n_grid = n_grid

        self.distributions_1d = {}
        self.interpolators = {}
        self.r_grids = {}

        self.process_configs = {}
        self.vegas_integrators = {}
        self._integrator_suffix = ''
        self._analytical_integrators = {}

        self.current_time = 0.0
        self.step_count = 0
        self.scale_factor = lambda t: 1.0  # default: no expansion
        if self.world_rank == 0:
            print(f"BESThep initialized")
            print(f"q_min = {q_min}, q_max = {q_max}, grid size = {n_grid}")
            print(f"MPI: {self.world_size} total ranks, "
                  f"{self.n_r_parallel} r-groups, "
                  f"{self.ranks_per_group} ranks/group")

    def set_mass_func(self, species, func):
        """Set time-dependent mass: func(t) -> mass"""
        self.species_mass_func[species] = func
        if self.world_rank == 0:
            print(f"Species {species}: time-dependent mass set")

    def set_radiation_dominated(self, a0=1.0, t0=1.0):
        self.scale_factor = lambda t, _a0=a0, _t0=t0: _a0 * (t / _t0)**0.5
        if self.world_rank == 0:
            print(f"Scale factor: radiation dominated, a0={a0}, t0={t0}")
    # ------------------------------------------------------------------
    # Species energy function (supports massive particles)
    # ------------------------------------------------------------------
    def energy(self, p_vec, species=None):
        """E = sqrt(|p|^2 + m^2). p_vec is 3-vector or magnitude."""
        m = self.species_mass.get(species, 0.0) if species is not None else 0.0
        if isinstance(p_vec, (int, float)):
            return np.sqrt(p_vec**2 + m**2)
        p_sq = np.sum(np.asarray(p_vec)**2, axis=0) if np.asarray(p_vec).ndim > 1 \
            else np.asarray(p_vec)**2
        return np.sqrt(p_sq + m**2)

    # ------------------------------------------------------------------
    # Process registration
    # ------------------------------------------------------------------
    def add_process(self, name, input_species, output_species, matrix_element,
                    coupling=1.0, neval=1000, nitn=2, alpha=0.5,
                    delta_width=0.01):
        self.process_configs[name] = {
            'input': input_species,
            'output': output_species,
            'matrix_element': matrix_element,
            'coupling': coupling,
            'n_in': len(input_species),
            'n_out': len(output_species),
            'n_total': len(input_species) + len(output_species),
            'neval': neval,
            'nitn': nitn,
            'alpha': alpha,
            'delta_width': delta_width,
        }
        for species in input_species + output_species:
            if species not in self.species_config:
                self.species_config[species] = 'boson'
                self.species_mass[species] = 0.0
        if self.world_rank == 0:
            print(f"Added process {name}: {input_species} -> {output_species} "
                  f"(neval={neval}, nitn={nitn}, delta_width={delta_width})")

    def set_species(self, name, stat='boson', mass=0.0):
        """Set species statistics and mass."""
        self.species_config[name] = stat
        self.species_mass[name] = mass
        if self.world_rank == 0:
            print(f"Species {name}: stat={stat}, mass={mass}")

    # ------------------------------------------------------------------
    # Species initialization
    # ------------------------------------------------------------------
    def initialize_species(self, species, init_func, grid='log',
                           stat='boson', mass=0.0):
        """Initialize species distribution on a fixed grid.
        grid: 'log' or 'linear'."""
        self.species_config[species] = stat
        self.species_mass[species] = mass
        self.species_list = list(self.species_config.keys())

        if grid == 'log':
            self.r_grids[species] = np.logspace(
                np.log10(self.q_min), np.log10(self.q_max), self.n_grid)
        else:
            self.r_grids[species] = np.linspace(
                self.q_min, self.q_max, self.n_grid)

        f_values = np.array([init_func(r) for r in self.r_grids[species]])
        self.distributions_1d[species] = f_values
        self.interpolators[species] = ExtrapolatingInterp(
            self.r_grids[species], f_values,
            mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

        if self.world_rank == 0:
            stat = self.species_config[species]
            mass = self.species_mass.get(species, 0.0)
            total = np.trapezoid(f_values * self.r_grids[species]**2,
                                 self.r_grids[species]) * 4 * np.pi
            print(f"Initialized {species} ({stat}, m={mass}, grid={grid}): "
                  f"total number = {total:.3e}")

    def get_f(self, r, species):
        if species not in self.interpolators:
            return 0.0
        return self.interpolators[species](r)

    # ------------------------------------------------------------------
    # Vegas integrator setup
    # ------------------------------------------------------------------
    def setup_collision_integrator(self, process_name, section=None, mode='net'):
        suffix = getattr(self, '_integrator_suffix', '')
        proc_key = process_name + suffix

        if proc_key not in self.vegas_integrators:
            self.vegas_integrators[proc_key] = {}

        key = f"{section}_{mode}" if section is not None else mode
        if key not in self.vegas_integrators[proc_key]:
            config = self.process_configs[process_name]
            n_integrate = config['n_in'] + config['n_out'] - 2
            domain = []
            for _ in range(n_integrate):
                domain.extend([
                    [self.q_min, self.q_max], [0, np.pi], [0, 2 * np.pi]
                ])
            if self.world_rank == 0:
                print(f"  Creating Vegas integrator for {proc_key} ({mode}): "
                      f"{len(domain)} dimensions", flush=True)
            self.vegas_integrators[proc_key][key] = \
                self._vegas.Integrator(domain, mpi=True)
        return self.vegas_integrators[proc_key][key]

    # ------------------------------------------------------------------
    # Batch collision integrand (core)
    # ------------------------------------------------------------------
    def collision_integrand_batch(self, process_name, target_species,
                                  q1_mag, delta_width, mode='net', t=0.0):
        config = self.process_configs[process_name]
        input_species = config['input']
        output_species = config['output']
        all_species = input_species + output_species
        n_in = config['n_in']
        n_out = config['n_out']
        n_total = n_in + n_out


        # Place target on the side with more particles
        force = getattr(self, '_force_target_side', None)
        if force == 'input' and target_species in input_species:
            target_idx = input_species.index(target_species)
            sign = 1
        elif force == 'output' and target_species in output_species:
            target_idx = n_in + output_species.index(target_species)
            sign = -1
        elif n_out > n_in and target_species in output_species:
            target_idx = n_in + output_species.index(target_species)
            sign = -1
        elif target_species in input_species:
            target_idx = input_species.index(target_species)
            sign = 1
        else:
            target_idx = n_in + output_species.index(target_species)
            sign = -1

        if n_in == n_out:
            # same side as target
            if target_idx < n_in:
                candidates = [i for i in range(n_in) if i != target_idx]
            else:
                candidates = [i for i in range(n_in, n_total) if i != target_idx]
        else:
            # fewer-particle side
            if n_in <= n_out:
                candidates = [i for i in range(n_in) if i != target_idx]
            else:
                candidates = [i for i in range(n_in, n_total) if i != target_idx]

        if not candidates:
            candidates = [i for i in range(n_total) if i != target_idx]
        conserved_idx = candidates[0]

        integrate_indices = [i for i in range(n_total)
                             if i != target_idx and i != conserved_idx]

        stats = [self.species_config[s] for s in all_species]
        interps = {s: self.interpolators[s] for s in set(all_species)}
        matrix_element = config['matrix_element']
        coupling = config['coupling']
        pi_power = (2 * np.pi) ** (3 * (n_total - 1) - 4)
        cutoff_e = self.cutoff_energy_min
        vegas_mod = self._vegas
        a = self.scale_factor(t)
        masses = [self.species_mass.get(s, 0.0) for s in all_species]


        @vegas_mod.lbatchintegrand
        def integrand(x):
            N = x.shape[0]

            mom_x = np.zeros((n_total, N))
            mom_y = np.zeros((n_total, N))
            mom_z = np.zeros((n_total, N))
            energies = np.zeros((n_total, N))

            # Target along x-axis
            mom_x[target_idx] = q1_mag / a
            energies[target_idx] = np.sqrt(q1_mag**2 / a**2 + masses[target_idx]**2)

            # Integrated particles
            jacobian = np.ones(N)
            x_idx = 0
            for part_idx in integrate_indices:
                q = x[:, x_idx]
                theta = x[:, x_idx + 1]
                phi = x[:, x_idx + 2]
                x_idx += 3

                sin_theta = np.sin(theta)
                jacobian *= q**2 * sin_theta / a**3
                p = q / a
                mom_x[part_idx] = p * sin_theta * np.cos(phi)
                mom_y[part_idx] = p * sin_theta * np.sin(phi)
                mom_z[part_idx] = p * np.cos(theta)
                energies[part_idx] = np.sqrt(p**2 + masses[part_idx]**2)

            # Momentum conservation -> conserved particle
            p_in_x = np.zeros(N); p_in_y = np.zeros(N); p_in_z = np.zeros(N)
            p_out_x = np.zeros(N); p_out_y = np.zeros(N); p_out_z = np.zeros(N)

            for k in range(n_in):
                if k != conserved_idx:
                    p_in_x += mom_x[k]; p_in_y += mom_y[k]; p_in_z += mom_z[k]
            for k in range(n_in, n_total):
                if k != conserved_idx:
                    p_out_x += mom_x[k]; p_out_y += mom_y[k]; p_out_z += mom_z[k]

            if conserved_idx < n_in:
                pc_x = p_out_x - p_in_x
                pc_y = p_out_y - p_in_y
                pc_z = p_out_z - p_in_z
            else:
                pc_x = p_in_x - p_out_x
                pc_y = p_in_y - p_out_y
                pc_z = p_in_z - p_out_z

            mom_x[conserved_idx] = pc_x
            mom_y[conserved_idx] = pc_y
            mom_z[conserved_idx] = pc_z
            pc_mag_sq = pc_x**2 + pc_y**2 + pc_z**2
            energies[conserved_idx] = np.sqrt(
                pc_mag_sq + masses[conserved_idx]**2)

            # Energy conservation delta
            E_in = np.sum(energies[:n_in], axis=0)
            E_out = np.sum(energies[n_in:], axis=0)
            E_diff = np.abs(E_in - E_out)

            eff_width = delta_width * (E_in + E_out) /2.
            norm_d = 1.0 / (eff_width * np.sqrt(2 * np.pi))
            delta_f = norm_d * np.exp(-E_diff**2 / (2 * eff_width**2))

            # Phase space: 1 / prod(2E)
            prod_2E = np.ones(N)
            for k in range(n_total):
                prod_2E *= 2.0 * energies[k]
            phase = jacobian / (prod_2E * pi_power)

            # Distribution functions
            f_arr = []
            for k in range(n_total):
                sp = all_species[k]
                if k == target_idx:
                    r_k = np.full(N, q1_mag)
                elif k == conserved_idx:
                    r_k = np.sqrt(pc_mag_sq)* a
                else:
                    int_pos = integrate_indices.index(k)
                    r_k = x[:, int_pos * 3]
                f_arr.append(np.clip(interps[sp](r_k), 0, None))

            # Statistical factors
            if mode == 'backward' or mode == 'net':
                BW = np.ones(N)
                for k in range(n_in, n_total):
                    BW *= f_arr[k]
                for k in range(n_in):
                    if stats[k] == 'boson':
                        BW *= (1 + f_arr[k])
                    elif stats[k] == 'fermion':
                        BW *= (1 - f_arr[k])

            if mode == 'forward' or mode == 'net':
                FW = np.ones(N)
                for k in range(n_in):
                    FW *= f_arr[k]
                for k in range(n_in, n_total):
                    if stats[k] == 'boson':
                        FW *= (1 + f_arr[k])
                    elif stats[k] == 'fermion':
                        FW *= (1 - f_arr[k])

            if mode == 'backward':
                stat_factor = sign * BW
            elif mode == 'forward':
                stat_factor = sign * FW
            else:
                stat_factor = sign * (BW - FW)

            momenta_batch = np.stack([mom_x, mom_y, mom_z], axis=1)
            M_sq = matrix_element(momenta_batch, coupling)

            result = stat_factor * delta_f * phase * M_sq

            # Zero out unphysical points
            bad = np.zeros(N, dtype=bool)
            for k in range(n_total):
                bad |= (energies[k] < cutoff_e)
            result[bad] = 0.0

            return result

        return integrand

    # ------------------------------------------------------------------
    # Collision rate computation
    # ------------------------------------------------------------------
    def compute_collision_rate(self, r_target, species, active_processes,
                            r_index=0, t=0.0):
        if not hasattr(self, 'error_stats'):
            self.error_stats = {'dropped': 0, 'neglected': 0}
        if not hasattr(self, 'adaptive_widths'):
            self.adaptive_widths = {}
        
        total_rate = 0.0

        for process_name in active_processes:
            config = self.process_configs[process_name]
            all_species = config['input'] + config['output']
            if species not in all_species:
                continue

            section = self.color
            integrator_f = self.setup_collision_integrator(
                process_name, section, 'forward')
            integrator_b = self.setup_collision_integrator(
                process_name, section, 'backward')

            neval = config.get('neval', 1000)
            nitn = config.get('nitn', 2)
            alpha = config.get('alpha', 0.5)
            dw_default = config.get('delta_width', 0.01)

            # Get adaptive widths (or initialize from default)
            key = process_name
            if key not in self.adaptive_widths:
                self.adaptive_widths[key] = {}
            if r_index not in self.adaptive_widths[key]:
                self.adaptive_widths[key][r_index] = {
                    'forward': dw_default,
                    'backward': dw_default
                }

            dw_f = self.adaptive_widths[key][r_index]['forward']
            dw_b = self.adaptive_widths[key][r_index]['backward']

            batch_f = self.collision_integrand_batch(
                process_name, species, r_target, dw_f, mode='forward', t=t)
            batch_b = self.collision_integrand_batch(
                process_name, species, r_target, dw_b, mode='backward', t=t)

            result_f = integrator_f(batch_f, nitn=nitn, neval=neval, alpha=alpha)
            result_b = integrator_b(batch_b, nitn=nitn, neval=neval, alpha=alpha)

            # Adapt widths based on rel_err
            for result, mode in [(result_f, 'forward'), (result_b, 'backward')]:
                if result.mean != 0:
                    rel_err = result.sdev / abs(result.mean)
                    dw_cur = self.adaptive_widths[key][r_index][mode]
                    if rel_err > 0.1:
                        self.adaptive_widths[key][r_index][mode] = dw_cur * 2.0
                    elif rel_err < 0.0001:
                        self.adaptive_widths[key][r_index][mode] = dw_cur * 0.5

            rate_contrib = result_b.mean - result_f.mean

            for result, rname in [(result_f, 'forward'), (result_b, 'backward')]:
                if result.mean != 0:
                    rel_err = result.sdev / abs(result.mean)
                    if rel_err > 5.0 and self.world_rank == 0:
                        print(f"      Warning: Large Vegas error at "
                            f"r={r_target:.3f} for "
                            f"{process_name}-{rname}: "
                            f"rel_err={rel_err:.2f}")
                    if rel_err > 1.0:
                        self.error_stats['dropped'] += 1
                        rate_contrib = 0.0
                else:
                    self.error_stats['neglected'] += 1

            total_rate += rate_contrib
        return total_rate

    # ------------------------------------------------------------------
    # Rate computation over grid (Vegas)
    # ------------------------------------------------------------------
    def _compute_rates_single_pass(self, active_processes, t=0.0,
                                    species_filter=None):
        """Compute rates for a single target side configuration.
        
        species_filter: if provided, only compute for these species.
        """
        for s, func in self.species_mass_func.items():
            self.species_mass[s] = func(t)
        species_rates = {}
        compute_list = species_filter if species_filter else self.species_list
        for species in compute_list:
            t_sp = time.time()
            if self.world_rank == 0:
                stat = self.species_config[species]
                mass = self.species_mass.get(species, 0.0)
                print(f"\n  Computing C[f] for {species} "
                      f"({stat}, m={mass}, a={self.scale_factor(t):.4f}):")
                print(f"    Grid points: {len(self.r_grids[species])}")

            n_r = len(self.r_grids[species])
            rates_local = np.zeros_like(self.r_grids[species])

            block_size = n_r // self.n_r_parallel
            remainder = n_r % self.n_r_parallel
            start_idx = self.color * block_size + min(self.color, remainder)
            end_idx = (start_idx + block_size
                       + (1 if self.color < remainder else 0))
            my_r_indices = list(range(start_idx, end_idx))
            if self.step_count % 2 == 1:
                my_r_indices = my_r_indices[::-1]

            for idx, i in enumerate(my_r_indices):
                r = self.r_grids[species][i]
                if r < self.cutoff_zero:
                    continue
                rates_local[i] = self.compute_collision_rate(
                    r, species, active_processes, r_index=i, t=t)
                if (self.sub_rank == 0
                        and (idx + 1) % max(1, len(my_r_indices) // 5) == 0):
                    elapsed = time.time() - t_sp
                    rate_calc = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = ((len(my_r_indices) - idx - 1) / rate_calc
                           if rate_calc > 0 else 0)
                    if self.world_rank == 0:
                        print(f"    [Vegas] Group {self.color}: "
                              f"{(idx+1)*100//len(my_r_indices)}% "
                              f"({idx+1}/{len(my_r_indices)}), "
                              f"ETA={eta:.0f}s", flush=True)

            if self.sub_rank != 0:
                rates_local[:] = 0.0

            rates = np.zeros_like(rates_local)
            self.world_comm.Allreduce(rates_local, rates, op=MPI.SUM)
            species_rates[species] = rates

            if self.world_rank == 0:
                print(f"    [Vegas] done in {time.time()-t_sp:.1f}s, "
                      f"max|C|={np.max(np.abs(rates)):.3e}")

        return species_rates

    def _compute_rates_vegas(self, active_processes, t=0.0):
            """Compute collision rates with slot summation for n_in != n_out.

            For each process and each species, counts how many times the
            species appears on input vs output side. If counts differ,
            computes C_in and C_out separately and combines:
                C_total = n_in_s * C_in + n_out_s * C_out
            If counts are equal, computes once and multiplies by
                (n_in_s + n_out_s).
            """
            # Update masses at time t
            for s, func in self.species_mass_func.items():
                self.species_mass[s] = func(t)

            species_rates = {sp: np.zeros_like(self.r_grids[sp])
                            for sp in self.species_list}

            for process_name in active_processes:
                config = self.process_configs[process_name]
                input_species = config['input']
                output_species = config['output']

                for species in self.species_list:
                    n_in_s = input_species.count(species)
                    n_out_s = output_species.count(species)

                    if n_in_s == 0 and n_out_s == 0:
                        continue


                    if n_in_s == n_out_s:
                        self._force_target_side = None
                        k = self._compute_rates_single_pass(
                            [process_name], t=t, species_filter=[species])
                        symmetric = sorted(input_species) == sorted(output_species)
                        mult = n_in_s if symmetric else (n_in_s + n_out_s)
                        species_rates[species] += mult * k[species]

                    else:
                        if n_in_s > 0:
                            self._integrator_suffix = '_in'
                            self._force_target_side = 'input'
                            k_in = self._compute_rates_single_pass(
                                [process_name], t=t, species_filter=[species])
                            species_rates[species] += n_in_s * k_in[species]

                        if n_out_s > 0:
                            self._integrator_suffix = '_out'
                            self._force_target_side = 'output'
                            k_out = self._compute_rates_single_pass(
                                [process_name], t=t, species_filter=[species])
                            species_rates[species] += n_out_s * k_out[species]

                        self._force_target_side = None
                        self._integrator_suffix = ''

            return species_rates


    # ------------------------------------------------------------------
    # Rate computation (Analytical, 2->2 only)
    # ------------------------------------------------------------------
    def _compute_rates_all_species(self, process_name, n_F):
        config = self.process_configs[process_name]
        M_squared = config['coupling'] ** 2
        input_species = config['input']
        output_species = config['output']
        species_rates = {}
        all_sp = config['input'] + config['output']
        masses = [self.species_mass.get(s, 0.0) for s in all_sp]

        for species in self.species_list:
            t_sp = time.time()
            stat = self.species_config[species]
            n_r = len(self.r_grids[species])

            if self.world_rank == 0:
                print(f"\n  Computing C[f] for {species} ({stat}):")
                print(f"    Grid points: {n_r}")

            key = (species, M_squared)
            if key not in self._analytical_integrators:
                self._analytical_integrators[key] = \
                    CollisionIntegral2to2Analytical(
                        self.q_min, self.q_max, M_squared,
                        masses=masses, n_F=n_F, grid='log')
            ci = self._analytical_integrators[key]

            rates_local = np.zeros(n_r)
            my_r_indices = list(range(self.world_rank, n_r, self.world_size))

            for idx, i in enumerate(my_r_indices):
                r = self.r_grids[species][i]
                if r < self.cutoff_zero:
                    continue
                rates_local[i] = ci.compute_rate(
                    r, self.interpolators[species], stat)

                if (self.world_rank == 0
                        and (idx + 1) % max(1, len(my_r_indices) // 5) == 0):
                    elapsed = time.time() - t_sp
                    rate_calc = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = ((len(my_r_indices) - idx - 1) / rate_calc
                           if rate_calc > 0 else 0)
                    print(f"    [Analytical] "
                          f"{(idx+1)*100//len(my_r_indices)}% "
                          f"({idx+1}/{len(my_r_indices)}), "
                          f"ETA={eta:.0f}s", flush=True)

            rates = np.zeros(n_r)
            self.world_comm.Allreduce(rates_local, rates, op=MPI.SUM)
            n_in_s = input_species.count(species)
            n_out_s = output_species.count(species)
            symmetric = sorted(input_species) == sorted(output_species)
            mult = n_in_s if symmetric else (n_in_s + n_out_s)
            species_rates[species] = mult * rates

            if self.world_rank == 0:
                print(f"    [Analytical] done in {time.time()-t_sp:.1f}s, "
                      f"max|C|={np.max(np.abs(rates)):.3e}")

        return species_rates

    # ------------------------------------------------------------------
    # Time stepping (Vegas)
    # ------------------------------------------------------------------
    def evolve_step(self, dt, active_processes=None,
                    adapt_dt=True, method='heun'):
        if active_processes is None:
            active_processes = list(self.process_configs.keys())
        t_step_start = time.time()
        self.error_stats = {'dropped': 0, 'neglected': 0}

        if self.world_rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Step {self.step_count} | t = {self.current_time:.3e} "
                  f"| dt = {dt:.3e} | method = {method}")
            print(f"{'=' * 60}")

        self.r_grids = self.world_comm.bcast(self.r_grids, root=0)
        self.distributions_1d = self.world_comm.bcast(
            self.distributions_1d, root=0)
        self.interpolators = self.world_comm.bcast(self.interpolators, root=0)

        # k1
        k1 = self._compute_rates_vegas(active_processes, t=self.current_time)

        # Adaptive dt
        dt_actual = dt
        if adapt_dt and self.world_rank == 0:
            for species in self.species_list:
                f = self.distributions_1d[species]
                f_safe = np.maximum(f, 1e-30)
                dlogf_max = np.max(np.abs(dt * k1[species] / f_safe))
                if dlogf_max > 0.3:
                    ratio = 0.3 / dlogf_max
                    dt_actual = min(dt_actual,
                                    dt * 10**np.floor(np.log10(ratio)))
            if dt_actual < dt:
                print(f"    dt adapted: {dt:.3e} -> {dt_actual:.3e}")
        dt_actual = self.world_comm.bcast(dt_actual, root=0)

        # Time integration
        if method == 'euler':
            if self.world_rank == 0:
                for species in self.species_list:
                    self.distributions_1d[species] = euler_update(
                        self.distributions_1d[species],
                        k1[species], dt_actual)
                    self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

        elif method == 'heun':
            f_original = {}
            for species in self.species_list:
                f_original[species] = self.distributions_1d[species].copy()

            if self.world_rank == 0:
                for species in self.species_list:
                    self.distributions_1d[species] = euler_update(
                        self.distributions_1d[species],
                        k1[species], dt_actual)
                    self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

            self.distributions_1d = self.world_comm.bcast(
                self.distributions_1d, root=0)
            self.interpolators = self.world_comm.bcast(
                self.interpolators, root=0)

            if self.world_rank == 0:
                print(f"    [Heun] Computing corrector k2...")

            f_predictor = {}
            for species in self.species_list:
                f_predictor[species] = self.distributions_1d[species].copy()

            k2 = self._compute_rates_vegas(active_processes, t=self.current_time + dt_actual)

            if self.world_rank == 0:
                for species in self.species_list:
                    self.distributions_1d[species] = heun_update(
                        f_original[species], k1[species], k2[species],
                        dt_actual, f_pred=f_predictor[species])
                    self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

        # Broadcast final state
        self.distributions_1d = self.world_comm.bcast(
            self.distributions_1d, root=0)
        self.interpolators = self.world_comm.bcast(self.interpolators, root=0)

        self.current_time += dt_actual
        self.step_count += 1

        if self.world_rank == 0:
            print(f"\nStep completed in {time.time()-t_step_start:.1f}s")
            print(f"    Vegas errors - Dropped: "
                  f"{self.error_stats['dropped']}")
            print(f"    Vegas errors - Neglected: "
                  f"{self.error_stats['neglected']}")
            print(f"{'=' * 60}")

        return k1

    # ------------------------------------------------------------------
    # Time stepping (Analytical, 2->2 only)
    # ------------------------------------------------------------------
    def evolve_step_analytical(self, dt, process_name, n_F=80, adapt_dt=True,
                               method='heun'):
        t_step_start = time.time()

        if self.world_rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Analytical Step {self.step_count} | "
                  f"t = {self.current_time:.3e} "
                  f"| dt = {dt:.3e} | method = {method}")
            print(f"{'=' * 60}")

        self.r_grids = self.world_comm.bcast(self.r_grids, root=0)
        self.distributions_1d = self.world_comm.bcast(
            self.distributions_1d, root=0)
        self.interpolators = self.world_comm.bcast(self.interpolators, root=0)

        # k1
        k1 = self._compute_rates_all_species(process_name, n_F)

        # Adaptive dt
        dt_actual = dt
        if adapt_dt and self.world_rank == 0:
            for species in self.species_list:
                f = self.distributions_1d[species]
                f_safe = np.maximum(f, 1e-30)
                dlogf_max = np.max(np.abs(dt * k1[species] / f_safe))
                if dlogf_max > 0.3:
                    ratio = 0.3 / dlogf_max
                    dt_actual = min(dt_actual,
                                    dt * 10**np.floor(np.log10(ratio)))
            if dt_actual < dt:
                print(f"    dt adapted: {dt:.3e} -> {dt_actual:.3e}")
        dt_actual = self.world_comm.bcast(dt_actual, root=0)

        # Time integration
        if method == 'euler':
            if self.world_rank == 0:
                for species in self.species_list:
                    self.distributions_1d[species] = euler_update(
                        self.distributions_1d[species],
                        k1[species], dt_actual)
                    self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

        elif method == 'heun':
            f_original = {}
            for species in self.species_list:
                f_original[species] = self.distributions_1d[species].copy()

            if self.world_rank == 0:
                for species in self.species_list:
                    self.distributions_1d[species] = euler_update(
                        self.distributions_1d[species],
                        k1[species], dt_actual)
                    self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

            self.distributions_1d = self.world_comm.bcast(
                self.distributions_1d, root=0)
            self.interpolators = self.world_comm.bcast(
                self.interpolators, root=0)

            if self.world_rank == 0:
                print(f"    [Heun] Computing corrector k2...")

            f_predictor = {}
            for species in self.species_list:
                f_predictor[species] = self.distributions_1d[species].copy()
            k2 = self._compute_rates_all_species(process_name, n_F)

            if self.world_rank == 0:
                for species in self.species_list:
                    self.distributions_1d[species] = heun_update(
                        f_original[species], k1[species], k2[species],
                        dt_actual, f_pred=f_predictor[species])
                    self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))

        # Broadcast
        self.distributions_1d = self.world_comm.bcast(
            self.distributions_1d, root=0)
        self.interpolators = self.world_comm.bcast(self.interpolators, root=0)

        self.current_time += dt_actual
        self.step_count += 1

        if self.world_rank == 0:
            print(f"\nStep completed in {time.time()-t_step_start:.1f}s")

        return k1

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------
    def compute_moments(self):
        moments = {}
        for species in self.species_list:
            f = self.distributions_1d[species]
            r_grid = self.r_grids[species]
            m = self.species_mass.get(species, 0.0)
            E_grid = np.sqrt(r_grid**2 + m**2)
            n = np.trapezoid(f * r_grid**2, r_grid) * 4 * np.pi
            e = np.trapezoid(f * r_grid**2 * E_grid, r_grid) * 4 * np.pi
            moments[species] = {'n': n, 'e': e}
        return moments

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------
    def save_checkpoint(self, filename, history=None):
        if self.world_rank != 0:
            return
        process_configs_ser = {}
        for name, config in self.process_configs.items():
            cc = config.copy()
            cc['matrix_element_name'] = config['matrix_element'].__name__
            del cc['matrix_element']
            process_configs_ser[name] = cc

        state = {
            'species_config': self.species_config,
            'species_mass': self.species_mass,
            'q_min': self.q_min, 'q_max': self.q_max, 'n_grid': self.n_grid,
            'distributions_1d': self.distributions_1d,
            'r_grids': self.r_grids,
            'current_time': self.current_time,
            'step_count': self.step_count,
            'process_configs': process_configs_ser,
            'history': history,
            'vegas_integrators': self.vegas_integrators,
            'adaptive_widths': self.adaptive_widths,
        }
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filename, 'wb') as fh:
            pickle.dump(state, fh)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename, matrix_elements=None):
        """Load checkpoint. Pass matrix_elements={'name': func} to restore."""
        if self.world_rank == 0:
            print(f"Loading checkpoint: {filename}")
            with open(filename, 'rb') as fh:
                state = pickle.load(fh)
        else:
            state = None
        state = self.world_comm.bcast(state, root=0)

        self.species_config = state['species_config']
        self.species_mass = state.get('species_mass', {})
        self.species_list = list(self.species_config.keys())
        self.q_min = state['q_min']
        self.q_max = state['q_max']
        self.n_grid = state['n_grid']
        self.distributions_1d = state['distributions_1d']
        self.r_grids = state.get('r_grids', {})
        self.current_time = state['current_time']
        self.step_count = state['step_count']
        self.vegas_integrators = state.get('vegas_integrators', {})
        self.adaptive_widths = state.get('adaptive_widths', {})

        # Lookup for matrix element restoration:
        # 1) user-provided dict, 2) caller's globals, 3) best.py globals
        import inspect
        caller_globals = inspect.stack()[1][0].f_globals
        me_lookup = matrix_elements or {}

        self.process_configs = {}
        for name, config in state['process_configs'].items():
            cc = config.copy()
            func_name = cc.get('matrix_element_name')
            func = (me_lookup.get(func_name)
                    or caller_globals.get(func_name)
                    or globals().get(func_name))
            if func:
                cc['matrix_element'] = func
                del cc['matrix_element_name']
                self.process_configs[name] = cc
                if self.world_rank == 0:
                    print(f"  Restored process: {name}")
            else:
                if self.world_rank == 0:
                    print(f"  Warning: Could not restore {name} "
                          f"(function '{func_name}' not found)")

        if not self.r_grids:
            for species in self.species_list:
                self.r_grids[species] = np.linspace(
                    self.q_min, self.q_max, self.n_grid)
        for species in self.species_list:
            self.interpolators[species] = ExtrapolatingInterp(
                self.r_grids[species], self.distributions_1d[species],
                mass=self.species_mass.get(species, 0.0),
                stat=self.species_config.get(species, 'boson'))
            if species not in self.species_mass:
                self.species_mass[species] = 0.0

        return state.get('history', None)


"""
Updated analytical 2->2 collision integral with massive particle support.
Replaces the massless-only version in besthep.py.

Based on Ala-Mattinen et al., Phys. Rev. D 105, 123005 (2022), Appendix A.
Equations (A8)-(A18) with general masses m1, m2, m3, m4.
"""

# ======================================================================
# Angular integral kernels (general mass 2->2)
# ======================================================================
def _F_backward_single(p1, p3, p4, M_squared, masses):
    """
    Backward kinematic function F(p1, p3, p4) for 12->34.
    
    Energy conservation: E1 + E2 = E3 + E4, so E2 = E3 + E4 - E1.
    Then p2 = sqrt(E2^2 - m2^2).
    
    masses = [m1, m2, m3, m4]
    """
    m1, m2, m3, m4 = masses
    
    E1 = np.sqrt(p1**2 + m1**2)
    E3 = np.sqrt(p3**2 + m3**2)
    E4 = np.sqrt(p4**2 + m4**2)
    E2 = E3 + E4 - E1
    
    if E2 <= m2 or E2 <= 0:
        return 0.0
    p2 = np.sqrt(E2**2 - m2**2)
    if p2 <= 1e-30 or p1 <= 1e-30:
        return 0.0
    
    Q = m1**2 - m2**2 + m3**2 + m4**2
    gamma = E3 * E4 - E1 * E3 - E1 * E4
    kappa = p1**2 + p3**2

    def integrand(cos_theta):
        sin_theta_sq = max(0.0, 1.0 - cos_theta**2)
        eps = p1 * p3 * cos_theta
        
        a = p4**2 * (-4.0 * kappa + 8.0 * eps)
        if a >= 0:
            return 0.0
        b = p4 * (-p1 + eps / p1) * (8.0 * gamma + 4.0 * Q + 8.0 * eps)
        c = 4.0 * p3**2 * p4**2 * sin_theta_sq - (2.0 * (gamma + eps) + Q)**2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return 0.0
        return M_squared * np.pi / np.sqrt(-a)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = quad(integrand, -1, 1, limit=50,
                         epsabs=1e-12, epsrel=1e-8)
    return max(0.0, result)


def _F_forward_single(p1, p2, p3, M_squared, masses):
    """
    Forward kinematic function F'(p1, p2, p3) for 12->34.
    
    Energy conservation: E4 = E1 + E2 - E3.
    Then p4 = sqrt(E4^2 - m4^2).
    
    masses = [m1, m2, m3, m4]
    """
    m1, m2, m3, m4 = masses
    
    E1 = np.sqrt(p1**2 + m1**2)
    E2 = np.sqrt(p2**2 + m2**2)
    E3 = np.sqrt(p3**2 + m3**2)
    E4 = E1 + E2 - E3
    
    if E4 <= m4 or E4 <= 0:
        return 0.0
    p4 = np.sqrt(E4**2 - m4**2)
    if p4 <= 1e-30 or p1 <= 1e-30:
        return 0.0
    
    Q_prime = m1**2 - m4**2 + m2**2 + m3**2
    gamma_p = E1 * E2 - E1 * E3 - E2 * E3
    kappa_p = p1**2 + p3**2

    def integrand(cos_theta):
        sin_theta_sq = max(0.0, 1.0 - cos_theta**2)
        eps = p1 * p3 * cos_theta
        
        a = p2**2 * (-4.0 * kappa_p + 8.0 * eps)
        if a >= 0:
            return 0.0
        b = p2 * (p1 - eps / p1) * (8.0 * gamma_p + 4.0 * Q_prime + 8.0 * eps)
        c = 4.0 * p2**2 * p3**2 * sin_theta_sq - (2.0 * (gamma_p + eps) + Q_prime)**2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return 0.0
        return M_squared * np.pi / np.sqrt(-a)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = quad(integrand, -1, 1, limit=50,
                         epsabs=1e-12, epsrel=1e-8)
    return max(0.0, result)


# ======================================================================
# Analytical 2->2 collision integral (general mass)
# ======================================================================
class CollisionIntegral2to2Analytical:
    """Analytical 2->2 collision integral with exact energy conservation.
    
    Supports general masses m1, m2, m3, m4.
    For identical particles (e.g. phi phi -> phi phi), pass masses=[m,m,m,m].
    """

    def __init__(self, q_min, q_max, M_squared, masses=None,
                 n_F=80, grid='log'):
        self.q_min = q_min
        self.q_max = q_max
        self.M_squared = M_squared
        self.masses = masses if masses is not None else [0.0, 0.0, 0.0, 0.0]
        self.n_F = n_F
        if grid == 'log':
            self.p_grid = np.logspace(np.log10(q_min), np.log10(q_max), n_F)
        else:
            self.p_grid = np.linspace(q_min, q_max, n_F)
        self._F_BW = {}
        self._F_FW = {}

    def _energy(self, p, mass_idx):
        m = self.masses[mass_idx]
        return np.sqrt(p**2 + m**2)

    def _ensure_F_backward(self, p1):
        """Precompute F(p1, p3, p4) on grid.
        Backward: integrate over (p3, p4), E2 = E3+E4-E1."""
        p1_key = round(p1, 10)
        if p1_key in self._F_BW:
            return self._F_BW[p1_key]
        
        m1, m2, m3, m4 = self.masses
        E1 = np.sqrt(p1**2 + m1**2)
        
        # Integration variables: (p3, p2) on grid
        # p4 determined by energy conservation: E4 = E1 + E2 - E3
        # But we want F(p1, p3, p4), integrating over (p3, p4)
        # with p2 = sqrt((E3+E4-E1)^2 - m2^2)
        F = np.zeros((self.n_F, self.n_F))
        for i, p3 in enumerate(self.p_grid):
            for j, p4 in enumerate(self.p_grid):
                E3 = np.sqrt(p3**2 + m3**2)
                E4 = np.sqrt(p4**2 + m4**2)
                E2 = E3 + E4 - E1
                if E2 <= m2 or E2 <= 0:
                    continue
                p2 = np.sqrt(E2**2 - m2**2)
                if p2 <= 0:
                    continue
                F[i, j] = _F_backward_single(p1, p3, p4,
                                             self.M_squared, self.masses)
        self._F_BW[p1_key] = F
        return F

    def _ensure_F_forward(self, p1):
        """Precompute F'(p1, p2, p3) on grid.
        Forward: integrate over (p2, p3), E4 = E1+E2-E3."""
        p1_key = round(p1, 10)
        if p1_key in self._F_FW:
            return self._F_FW[p1_key]
        
        m1, m2, m3, m4 = self.masses
        
        F = np.zeros((self.n_F, self.n_F))
        for i, p2 in enumerate(self.p_grid):
            for j, p3 in enumerate(self.p_grid):
                E1 = np.sqrt(p1**2 + m1**2)
                E2 = np.sqrt(p2**2 + m2**2)
                E3 = np.sqrt(p3**2 + m3**2)
                E4 = E1 + E2 - E3
                if E4 <= m4 or E4 <= 0:
                    continue
                p4 = np.sqrt(E4**2 - m4**2)
                if p4 <= 0:
                    continue
                F[i, j] = _F_forward_single(p1, p2, p3,
                                            self.M_squared, self.masses)
        self._F_FW[p1_key] = F
        return F

    def compute_rate_BW(self, p1, f_interp, species_stat='boson'):
        if p1 < 1e-30:
            return 0.0
        F_table = self._ensure_F_backward(p1)
        
        m1, m2, m3, m4 = self.masses
        E1 = np.sqrt(p1**2 + m1**2)
        f1 = float(f_interp(p1))
        stat_f1 = (1.0 + f1) if species_stat == 'boson' else (1.0 - f1)
        prefactor = 2.0 / (2.0 * np.pi)**4 / (2.0 * E1)

        integrand = np.zeros((self.n_F, self.n_F))
        for i, p3 in enumerate(self.p_grid):
            for j, p4 in enumerate(self.p_grid):
                if F_table[i, j] == 0:
                    continue
                E3 = np.sqrt(p3**2 + m3**2)
                E4 = np.sqrt(p4**2 + m4**2)
                E2 = E3 + E4 - E1
                if E2 <= m2 or E2 <= 0:
                    continue
                p2 = np.sqrt(E2**2 - m2**2)
                
                f3 = float(f_interp(p3))
                f4 = float(f_interp(p4))
                f2 = float(f_interp(p2))
                stat_f2 = ((1.0 + f2) if species_stat == 'boson'
                           else (1.0 - f2))
                
                # Phase space: p3^2/(2E3) * p4^2/(2E4) * dp3 dp4
                # The p3^2, p4^2 come from d^3p in spherical coords
                # dp3, dp4 handled by trapezoid
                integrand[i, j] = ((p3**2 / (2.0 * E3))
                                   * (p4**2 / (2.0 * E4))
                                   * F_table[i, j]
                                   * f3 * f4 * stat_f1 * stat_f2)

        result = np.trapezoid(
            np.trapezoid(integrand, x=self.p_grid, axis=1),
            x=self.p_grid)
        return prefactor * result

    def compute_rate_FW(self, p1, f_interp, species_stat='boson'):
        if p1 < 1e-30:
            return 0.0
        F_table = self._ensure_F_forward(p1)
        
        m1, m2, m3, m4 = self.masses
        E1 = np.sqrt(p1**2 + m1**2)
        f1 = float(f_interp(p1))
        prefactor = 2.0 / (2.0 * np.pi)**4 / (2.0 * E1)

        integrand = np.zeros((self.n_F, self.n_F))
        for i, p2 in enumerate(self.p_grid):
            for j, p3 in enumerate(self.p_grid):
                if F_table[i, j] == 0:
                    continue
                E2 = np.sqrt(p2**2 + m2**2)
                E3 = np.sqrt(p3**2 + m3**2)
                E4 = E1 + E2 - E3
                if E4 <= m4 or E4 <= 0:
                    continue
                p4 = np.sqrt(E4**2 - m4**2)
                
                f2 = float(f_interp(p2))
                f3 = float(f_interp(p3))
                f4 = float(f_interp(p4))
                stat_f3 = ((1.0 + f3) if species_stat == 'boson'
                           else (1.0 - f3))
                stat_f4 = ((1.0 + f4) if species_stat == 'boson'
                           else (1.0 - f4))
                
                integrand[i, j] = ((p2**2 / (2.0 * E2))
                                   * (p3**2 / (2.0 * E3))
                                   * F_table[i, j]
                                   * f1 * f2 * stat_f3 * stat_f4)

        result = np.trapezoid(
            np.trapezoid(integrand, x=self.p_grid, axis=1),
            x=self.p_grid)
        return prefactor * result

    def compute_rate(self, p1, f_interp, species_stat='boson'):
        return (self.compute_rate_BW(p1, f_interp, species_stat)
                - self.compute_rate_FW(p1, f_interp, species_stat))

    def clear_cache(self):
        self._F_BW.clear()
        self._F_FW.clear()