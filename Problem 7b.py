#Propose a two-stage continuous crystalliser and optimise its design and steady state
#performance. It is highly recommended to propose 2 alternative designs and
#operating strategies: one for the maximisation of the mean crystal size and one for
#the minimisation of the mean crystal size, both at steady state. Assume that the
#kinetics and model parameters of the batch system are still valid, but you need to add
#the required terms to capture the impact of the inputs and outputs to all relevant equations.
#Tips:
#• The maximum total volume of the two-stage crystalliser must not exceed 5 L and the total residence time must not exceed 45 min.
#• The temperature bounds are 323 K (UB) and 273 K (LB).
#• Other key performance constraints may be required.
#• Provide any relevant assumptions

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