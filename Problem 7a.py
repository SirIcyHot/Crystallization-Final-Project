"""
Two-Stage Continuous Crystallizer Design and Optimization
==========================================================

- Two CSTRs in series
- Optimize for:
    (1) Maximum mean crystal size
    (2) Minimum mean crystal size
- Steady state approximated via long-time integration
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---- Parameters ----
k_b = 7.8e19
k_g = 0.017
b = 6.2
g = 1.5

k_v = 0.24
rho_c = 1.296e6

C0 = 0.05  # feed concentration

# ---- Constraints ----
tau_max = 45.0
T_min, T_max = 273, 323

# ---- Saturation concentration ----
def C_sat(T):
    return 1.5846e-5*T**2 - 9.0567e-3*T + 1.3066

# ---- Single CSTR stage ----
def cstr_stage(y, y_in, T, tau):
    mu0, mu1, mu2, mu3, C = y
    mu0_in, mu1_in, mu2_in, mu3_in, C_in = y_in

    Cs = C_sat(T)
    S = max(C - Cs, 0)

    B = k_b * S**b
    G = k_g * S**g

    dmu0 = B + (mu0_in - mu0)/tau
    dmu1 = G*mu0 + (mu1_in - mu1)/tau
    dmu2 = 2*G*mu1 + (mu2_in - mu2)/tau
    dmu3 = 3*G*mu2 + (mu3_in - mu3)/tau
    dC   = -3*k_v*rho_c*G*mu2 + (C_in - C)/tau

    return np.array([dmu0, dmu1, dmu2, dmu3, dC])

# ---- Two-stage system ----
def two_stage_odes(t, y, T1, T2, tau1, tau2):
    y1 = y[:5]
    y2 = y[5:]

    y_feed = [0, 0, 0, 0, C0]

    dy1 = cstr_stage(y1, y_feed, T1, tau1)
    dy2 = cstr_stage(y2, y1, T2, tau2)

    return np.concatenate([dy1, dy2])

# ---- Mean crystal size ----
def mean_size(mu0, mu1):
    return mu1 / mu0 if mu0 > 1e-12 else 0

# ---- Objective function ----
def objective(x, mode="max"):
    T1, T2, tau1, tau2 = x

    if tau1 <= 0 or tau2 <= 0:
        return 1e6

    y0_full = [0]*10

    sol = solve_ivp(
        two_stage_odes,
        (0, 500),   # long time to reach steady state
        y0_full,
        args=(T1, T2, tau1, tau2)
    )

    if not sol.success:
        return 1e6

    # steady-state values (last point)
    mu0_2 = sol.y[5, -1]
    mu1_2 = sol.y[6, -1]

    d = mean_size(mu0_2, mu1_2)
    if not np.isfinite(d):
        return 1e6

    return -d if mode == "max" else d

# ---- Initial guess & bounds ----
x0 = [290, 280, 10, 30]
bounds = [(T_min, T_max), (T_min, T_max), (1, tau_max), (1, tau_max)]
constraints = [{"type": "ineq", "fun": lambda x: tau_max - (x[2] + x[3])}]

def optimize_mode(mode, n_starts=20, seed=42):
    rng = np.random.default_rng(seed)
    best_res = None

    for i in range(n_starts):
        if i == 0:
            x_init = np.array(x0, dtype=float)
        else:
            T1_init = rng.uniform(T_min, T_max)
            T2_init = rng.uniform(T_min, T_max)
            tau1_init = rng.uniform(1, tau_max - 1)
            tau2_max = max(1.0, tau_max - tau1_init)
            tau2_init = rng.uniform(1, tau2_max)
            x_init = np.array([T1_init, T2_init, tau1_init, tau2_init], dtype=float)

        res = minimize(
            objective,
            x_init,
            args=(mode,),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={"maxiter": 300, "ftol": 1e-9}
        )

        if not res.success:
            continue

        if best_res is None or res.fun < best_res.fun:
            best_res = res

    return best_res

# ---- Solve optimization ----
res_max = optimize_mode("max", n_starts=20, seed=42)
res_min = optimize_mode("min", n_starts=20, seed=123)

# ---- Function to evaluate final design ----
def evaluate_solution(x):
    T1, T2, tau1, tau2 = x

    sol = solve_ivp(
        two_stage_odes,
        (0, 500),
        [0]*10,
        args=(T1, T2, tau1, tau2)
    )

    mu0_2 = sol.y[5, -1]
    mu1_2 = sol.y[6, -1]

    d = mean_size(mu0_2, mu1_2)
    return d

# ---- Results ----
print("\n===== DESIGN 1: MAXIMIZE CRYSTAL SIZE =====")
if res_max is None:
    print("No feasible optimum found for max-size objective.")
else:
    print("Optimal variables:", res_max.x)
    d_max = evaluate_solution(res_max.x)
    print(f"Mean size: {d_max:.6e}")

print("\n===== DESIGN 2: MINIMIZE CRYSTAL SIZE =====")
if res_min is None:
    print("No feasible optimum found for min-size objective.")
else:
    print("Optimal variables:", res_min.x)
    d_min = evaluate_solution(res_min.x)
    print(f"Mean size: {d_min:.6e}")

# ---- Compact comparison table ----
print("\n===== COMPARISON TABLE =====")
header = f"{'Design':<12} {'T1 (K)':>10} {'T2 (K)':>10} {'tau1 (min)':>12} {'tau2 (min)':>12} {'Mean size':>14}"
print(header)
print("-" * len(header))

if res_max is None:
    print(f"{'Max size':<12} {'-':>10} {'-':>10} {'-':>12} {'-':>12} {'-':>14}")
else:
    T1, T2, tau1, tau2 = res_max.x
    d_max = evaluate_solution(res_max.x)
    print(f"{'Max size':<12} {T1:10.2f} {T2:10.2f} {tau1:12.2f} {tau2:12.2f} {d_max:14.6e}")

if res_min is None:
    print(f"{'Min size':<12} {'-':>10} {'-':>10} {'-':>12} {'-':>12} {'-':>14}")
else:
    T1, T2, tau1, tau2 = res_min.x
    d_min = evaluate_solution(res_min.x)
    print(f"{'Min size':<12} {T1:10.2f} {T2:10.2f} {tau1:12.2f} {tau2:12.2f} {d_min:14.6e}")