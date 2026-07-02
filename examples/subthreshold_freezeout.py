"""
DRAKE 4.3 sub-threshold freeze-out -- run script (exponential Rosenbrock-Euler).

Uses the solver's built-in exponential Rosenbrock-Euler stepper:

    solver.evolve_step(dt, method='exprb')

'exprb' is the diagonal (W-type) exponential Rosenbrock-Euler method built into
besthep [Hochbruck, Ostermann & Schweitzer (2009); Hochbruck & Ostermann (2010),
Acta Numerica 19, eq. (2.47)]. Per momentum mode it integrates df/dt = A - Gamma*f
exactly over the step, with A (production) and Gamma = -dC/df (diagonal Jacobian,
= loss/f - eta*A) assembled PER SLOT inside _compute_rates_vegas so that
A - Gamma*f == C holds even when the observed species sits on the output side.
It is single-stage (one Vegas pass per step, like Euler) and positivity-
preserving (A >= 0 => f_new >= 0 for any sign of Gamma) -- no clamp on
Gamma, no floor on f_new; first order under the diagonal-Jacobian
approximation. Reduces to forward Euler as Gamma->0; preserves
equilibria (C=0 => f unchanged). evolve_step asserts the A - Gamma*f == k1 identity
each step (MPI-safe) so a per-slot mis-assignment halts loudly rather than running
on silently.

phi2 is a prescribed thermal bath (PhiTwoBath), refreshed each step by
prescribe_phi2(). Ordering in the loop matters:
  - prescribe_phi2() is called ONCE before the loop, so the step-0 rates see
    phi2 at t0;
  - inside the loop it is evolve_step -> prescribe_phi2() -> record, so each step
    computes rates against phi2 at the CURRENT time, then refreshes phi2 to the
    new time for recording and for the next step's rates.

NOTE (deliberate, for generality/cleanliness): evolve_step advances ALL species,
so it also computes rates for phi2 and updates it -- that update is immediately
overwritten by prescribe_phi2(), i.e. phi2's rate work is wasted, and phi2's
(near-equilibrium) change also enters the adaptive-dt max. Accepted here in
exchange for using the solver's standard, self-consistent API (no hand-rolled
stepper, no forward/backward vs gain/loss name-mapping to get wrong).

COSMOLOGY: T(a) and a(t) are built from the Drees et al. dof table by enforcing
entropy conservation (heff T^3 a^3 = const -> a(T)) and Friedmann with the
heff-variation correction folded into g_*^{1/2} (Sqrt[g*] column):
    dT/dt = -(pi/sqrt(90)) (T^3/M_Pl) (heff / g_*^{1/2}),   reduced M_Pl.
The {T, a, t} table is built once and injected as solver.scale_factor.
delta_width sets the sqrt(s)~2 m2 threshold resolution.
"""
import os, sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from besthep import BEST


# --------------------------- scenario --------------------------------------
m1       = 1.0
r        = 1.1
m2       = r * m1
x_init   = 5.0
T0_bath  = m1 / x_init
coupling = 1.0          # = lambda directly (symmetry factor 1/4 is inside matrix_element)

# --------------------------- cosmology -------------------------------------
m1_phys       = 100.0           # GeV, sets T_phys = m1_phys * T_code for dof lookup
M_Pl_red_phys = 2.435e18        # GeV, reduced Planck mass
a0            = 1.0
x_stop        = 30.0
dof_file      = "dof_Drees_etal.dat"


def build_cosmology(dof_file, m1_phys, x_init, x_stop, a0=1.0,
                    M_Pl_red_phys=2.435e18, x_margin=1.15, n_pts=4000):
    """Build T(a), a(t) from the dof table (entropy conservation + Friedmann).

    Returns (scale_factor, T_of_a, t0) in CODE units (T_code = T_phys/m1_phys).
    dof columns: T[GeV], heff, Sqrt[geff], Sqrt[g*]. Reduced M_Pl convention.
    Verified (numpy): T(a0)=T0, heff*T^3*a^3 const, T_of_a(scale_factor(t)) ==
    T_of_t(t), constant-dof limit -> a ~ t^(1/2), T ~ 1/a.
    """
    T, heff, sgeff, sgstar = [], [], [], []
    with open(dof_file) as fh:
        for line in fh:
            s = line.split()
            if len(s) < 4:
                continue
            try:
                v = [float(x) for x in s[:4]]
            except ValueError:
                continue                       # header / non-numeric
            T.append(v[0]); heff.append(v[1]); sgeff.append(v[2]); sgstar.append(v[3])
    T = np.array(T); heff = np.array(heff)
    sgeff = np.array(sgeff); sgstar = np.array(sgstar)
    o = np.argsort(T)
    T, heff, sgeff, sgstar = T[o], heff[o], sgeff[o], sgstar[o]
    lgT = np.log(T)
    heff_of   = lambda Tp: np.interp(np.log(Tp), lgT, heff)
    sgeff_of  = lambda Tp: np.interp(np.log(Tp), lgT, sgeff)
    sgstar_of = lambda Tp: np.interp(np.log(Tp), lgT, sgstar)

    M_Pl = M_Pl_red_phys / m1_phys             # code units
    T0c  = 1.0 / x_init
    Tfin = 1.0 / (x_stop * x_margin)           # margin -> no extrapolation at the end
    Tp   = lambda Tc: Tc * m1_phys             # code -> physical (GeV) for lookup

    if Tp(T0c) > T.max() or Tp(Tfin) < T.min():
        raise ValueError(f"dof table [{T.min():.3g},{T.max():.3g}] GeV does not "
                         f"cover run range [{Tp(Tfin):.3g},{Tp(T0c):.3g}] GeV")

    Tg = np.logspace(np.log10(T0c), np.log10(Tfin), n_pts)   # decreasing
    heff0 = heff_of(Tp(T0c))
    a_g = a0 * (heff0 * T0c**3 / (heff_of(Tp(Tg)) * Tg**3))**(1.0/3.0)
    integrand = (np.sqrt(90) / np.pi) * (M_Pl / Tg**3) \
                * (sgstar_of(Tp(Tg)) / heff_of(Tp(Tg)))
    dt_cum = cumulative_trapezoid(-integrand, Tg, initial=0.0)   # Tg down -> >0
    H0 = sgeff_of(Tp(T0c)) * np.pi / np.sqrt(90) * T0c**2 / M_Pl
    t0 = 1.0 / (2.0 * H0)
    t_g = t0 + dt_cum

    order = np.argsort(t_g)
    t_s, a_s, T_s = t_g[order], a_g[order], Tg[order]

    def scale_factor(t):
        return (float(np.interp(t, t_s, a_s)) if np.isscalar(t)
                else np.interp(t, t_s, a_s))
    ao = np.argsort(a_s)
    a_asc, T_by_a = a_s[ao], T_s[ao]
    def T_of_a(a):
        return (float(np.interp(a, a_asc, T_by_a)) if np.isscalar(a)
                else np.interp(a, a_asc, T_by_a))
    return scale_factor, T_of_a, t0


# --------------------------- numerics --------------------------------------
q_min, q_max, n_grid = 0.1, 10.0, 40
neval    = int(2e5)
dt_frac  = 0.05
n_steps  = 500
delta_width = 0.05
adapt_width = True
max_rel_err, min_rel_err, max_rel_change = 0.1, 0.01, 0.3
checkpoint_file = "checkpoint.pkl"


# --------------------------- physics ---------------------------------------
def matrix_element(momenta, coupling):
    # |M|^2 = lambda^2 for phi1 phi1 -> phi2 phi2 (contact vertex -i*lambda).
    # besthep does NOT add identical-particle symmetry factors, so divide by
    # 1/(2!*2!) = 1/4 (two identical phi1 in, two identical phi2 out) here.
    # => coupling IS lambda directly (coupling=1 is lambda=1).
    return np.full(momenta.shape[2], coupling**2 / 4.0)

def be_on_grid(q_grid, mass, T, a, mu=0.0):
    p = np.asarray(q_grid, float) / a
    E = np.sqrt(p**2 + mass**2)
    x = np.clip((E - mu) / T, -700, 700)
    return 1.0 / np.expm1(x)

class PhiTwoBath:
    def __init__(self, mass, mu=0.0):
        self.mass, self.mu, self.a, self.T = mass, mu, 1.0, 1.0
    def set_state(self, a, T):
        self.a, self.T = a, T
    def __call__(self, q):
        p = np.asarray(q, float) / self.a
        E = np.sqrt(p**2 + self.mass**2)
        x = np.clip((E - self.mu) / self.T, -700, 700)
        return 1.0 / np.expm1(x)

def init_phi1(q):
    return be_on_grid(q, m1, T0_bath, a0, mu=0.0)
def init_phi2(q):
    return be_on_grid(q, m2, T0_bath, a0, mu=0.0)

def comoving_number(f, q):
    return np.trapezoid(f * q**2, q) / (2 * np.pi**2)


# --------------------------- build cosmology -------------------------------
scale_factor, T_of_a, t0 = build_cosmology(
    dof_file, m1_phys, x_init, x_stop, a0=a0, M_Pl_red_phys=M_Pl_red_phys)


# --------------------------- setup -----------------------------------------
solver = BEST(q_min=q_min, q_max=q_max, n_grid=n_grid,
              max_rel_change=max_rel_change, adapt_width=adapt_width,
              max_rel_err=max_rel_err, min_rel_err=min_rel_err)

solver.scale_factor = scale_factor       # full-treatment cosmology;
                                          

# --- resume from an existing checkpoint if one is present ------------------
# Same pattern as the elastic example: on resume, load_checkpoint restores the
# species, the process (incl. matrix_element by name), distributions, r_grids,
# current_time, step_count, history, vegas integrators and adaptive widths --
# so initialize_species / add_process are SKIPPED. The resume decision
# is broadcast from rank 0 so all ranks call the collective load together.
resume = os.path.exists(checkpoint_file) and solver.world_rank == 0
resume = solver.world_comm.bcast(resume, root=0)
if resume:
    history = solver.load_checkpoint(
        checkpoint_file, matrix_elements={'matrix_element': matrix_element})
    solver.scale_factor = scale_factor 
else:
    solver.initialize_species('phi1', init_phi1, stat='boson', mass=m1)
    solver.initialize_species('phi2', init_phi2, stat='boson', mass=m2)
    solver.add_process('ann', ['phi1', 'phi1'], ['phi2', 'phi2'],
                       matrix_element, coupling=coupling,
                       neval=neval, delta_width=delta_width)
    solver.current_time = t0
    history = solver.init_history()

# always (phi2_bath is stateless setup)
phi2_bath = PhiTwoBath(m2, mu=0.0)


def prescribe_phi2():
    """Overwrite phi2 with the thermal bath at the current time."""
    a = solver.scale_factor(solver.current_time)
    phi2_bath.set_state(a, T_of_a(a))
    solver.interpolators['phi2'] = phi2_bath
    solver.distributions_1d['phi2'] = phi2_bath(solver.r_grids['phi2'])


# --------------------------- evolution -------------------------------------
prescribe_phi2()                     # phi2 at the current time (t0 or resumed)
q1 = solver.r_grids['phi1']
# Reference values are the INITIAL-condition ones (deterministic), NOT the
# current distribution -- so n1/n1_0 stays relative to the true start even on
# a resumed run. At t0: a=a0, T=T0_bath, and phi1 starts in equilibrium.
_a_t0  = scale_factor(t0)
n1_0   = comoving_number(be_on_grid(q1, m1, T_of_a(_a_t0), _a_t0), q1)
neq_0  = n1_0

if solver.world_rank == 0:
    print("\nmode = exponential Rosenbrock-Euler (diagonal / W-type), method='exprb'")
    print(f"cosmology = full treatment from {dof_file} (reduced M_Pl)")
    print(f"t0 = {t0:.3e}   dt = {dt_frac}*t  (Hubble-paced)")
    print(f"{'step':>4} | {'t':>11} | {'x=m1/T':>7} | {'n1/n1_0':>9} | "
          f"{'neq/neq_0':>9} | {'n1/neq':>8}")
    print("-" * 66)

for step in range(n_steps):
    # already at/after the stop (e.g. resuming a finished run)? do nothing.
    if m1 / T_of_a(solver.scale_factor(solver.current_time)) > x_stop:
        if solver.world_rank == 0:
            print(f"already past x_stop={x_stop} (x="
                  f"{m1/T_of_a(solver.scale_factor(solver.current_time)):.3f}); nothing to do.")
        break

    dt = dt_frac * solver.current_time

    # phi1 (and phi2, discarded) advanced by one exprb step; sees phi2 at the
    # current time. evolve_step advances current_time by dt internally.
    solver.evolve_step(dt, method='exprb', adapt_dt=False)

    prescribe_phi2()                 # refresh phi2 to the new time
    m = solver.record(history)

    a = solver.scale_factor(solver.current_time)
    T_code = T_of_a(a)
    x = m1 / T_code
    n1   = comoving_number(solver.distributions_1d['phi1'], q1)
    neq  = comoving_number(be_on_grid(q1, m1, T_code, a), q1)

    if solver.world_rank == 0 and (solver.step_count % 5 == 0 or x > x_stop):
        print(f"{solver.step_count:>4} | {solver.current_time:>11.4e} | {x:>7.3f} | "
              f"{n1/n1_0:>9.4e} | {neq/neq_0:>9.4e} | {n1/max(neq,1e-300):>8.3f}")
    solver.save_checkpoint(checkpoint_file, history=history)
    if x > x_stop:
        break

if solver.world_rank == 0:
    print("-" * 66)
    print("READ: exponential Rosenbrock-Euler (method='exprb') -- linear decay per")
    print("mode integrated exactly, equilibria preserved (C=0 -> f unchanged),")
    print("Gamma->0 -> forward Euler, positivity structural (no clamp/floor).")
    print("Cosmology from the dof table (entropy + Friedmann, reduced M_Pl); Y(x)=n/s")
    print("with s=(2pi^2/45)heff(T)T^3 from the same table is the relic observable")
    print("(plot_yield.py). n1/neq>1 => forbidden relic above equilibrium.")
