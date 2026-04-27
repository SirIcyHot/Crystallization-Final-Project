import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---- Parameters ----
k_b = 7.8e19      # nucleation constant (example)
k_g = 0.017    # growth constant (example)
b = 6.2          # nucleation exponent
g = 1.5          # growth exponent

k_v = 0.24     # shape factor
rho_c = 1.296e6  # crystal density (g/m^3)

T = 298  # temperature (K), constant

# ---- Saturation concentration ----
def C_sat(T):
    return 1.5846e-5*T**2 - 9.0567e-3*T + 1.3066

# ---- ODE system ----
def crystallization_odes(t, y):
    mu0, mu1, mu2, mu3, C = y
    
    Cs = C_sat(T)
    supersat = max(C - Cs, 0)  # avoid negative supersaturation
    
    B = k_b * supersat**b
    G = k_g * supersat**g
    
    dmu0_dt = B
    dmu1_dt = G * mu0
    dmu2_dt = 2 * G * mu1
    dmu3_dt = 3 * G * mu2
    dC_dt   = -3 * k_v * rho_c * G * mu2
    
    return [dmu0_dt, dmu1_dt, dmu2_dt, dmu3_dt, dC_dt]

# ---- Initial conditions ----
mu0_0 = 0
mu1_0 = 0
mu2_0 = 0
mu3_0 = 0
C0 = 0.0256  # initial concentration

y0 = [mu0_0, mu1_0, mu2_0, mu3_0, C0]

# ---- Time span ----
t_span = (0, 80)  # minutes
t_eval = np.linspace(0, 80, 200)

# ---- Solve ODE ----
sol = solve_ivp(crystallization_odes, t_span, y0, t_eval=t_eval, method='RK45')

# ---- Extract results ----
t = sol.t
mu0, mu1, mu2, mu3, C = sol.y

# ---- Derived quantities ----
d_mean = mu1 / mu0  # mean size (handle divide-by-zero carefully)
CV = np.sqrt(mu2 * mu0 / mu1**2 - 1)

# ---- Plot ----
plt.figure()
plt.plot(t, C)
plt.xlabel("Time (min)")
plt.ylabel("Concentration")
plt.title("Concentration vs Time")

plt.figure()
plt.plot(t, d_mean)
plt.xlabel("Time (min)")
plt.ylabel("Mean crystal size")

plt.show()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---- time grid ----
N = 10  # number of control intervals
t_grid = np.linspace(0, 80, N)

#Final temperature meets 273 as specified in #4
# ---- bounds ----
T_min, T_max = 273, 315

#this part done by Wyatt in previous problems but simplified here for multiobjective optimization
# ---- objective function ----
def evaluate_profile(T_profile):
    def T_func(t):
        idx = np.searchsorted(t_grid, t) - 1
        idx = np.clip(idx, 0, N-1)
        return T_profile[idx]

    def odes(t, y):
        mu0, mu1, mu2, mu3, C = y
        T = T_func(t)

        Cs = 1.5846e-5*T**2 - 9.0567e-3*T + 1.3066
        S = max(C - Cs, 0)

        B = k_b * S**b
        G = k_g * S**g

        return [
            B,
            G * mu0,
            2 * G * mu1,
            3 * G * mu2,
            -3 * k_v * rho_c * G * mu2
        ]

    sol = solve_ivp(odes, (0,80), y0, t_eval=[80])
    mu0, mu1, mu2, mu3, C = sol.y[:, -1]

    d = mu1 / mu0 if mu0 > 0 else 0
    Y = (C0 - C) / C0
    return d, Y

# Multiobjective optimization idea:
# We optimize two competing goals at final time t=80 min:
# 1) d: mean crystal size (maximize)
# 2) Y: yield fraction (maximize)
# A weighted-sum scalarization is used:
# maximize  weight_size*d + (1 - weight_size)*Y
# where weight_size in [0,1].
def objective(T_profile, weight_size):
    d, Y = evaluate_profile(T_profile)
    combined_score = weight_size * d + (1 - weight_size) * Y

    return -combined_score  # minimize negative -> equivalent to maximizing score

# ---- solve for different weights ----
pareto = []

for weight_size in np.linspace(0, 1, 10):
    res = minimize(
        objective,
        x0=np.ones(N)*300,
        args=(weight_size,),
        bounds=[(T_min, T_max)]*N,
        method='SLSQP'
    )

    T_opt = res.x
    d, Y = evaluate_profile(T_opt)
    pareto.append((weight_size, d, Y))

# ---- report and visualize trade-off curve ----
print("\nMultiobjective optimization results (weighted sum):")
print("weight_size   mean_size(d)   yield(Y)")
for weight_size, d, Y in pareto:
    print(f"{weight_size:10.2f}   {d:12.6e}   {Y:8.4f}")

plt.figure()
plt.plot([p[2] for p in pareto], [p[1] for p in pareto], "o-")
plt.xlabel("Yield, Y")
plt.ylabel("Mean crystal size, d")
plt.title("Pareto Trade-off: Yield vs Mean Size")
plt.grid(True)
plt.show()