import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#test
# -----------------------------
# Problem constants
# -----------------------------
kv = 0.24
rho_c = 1.296e6          # g/m^3

b = 6.2                  # nucleation exponent
g = 1.5                  # growth exponent

T0 = 315                 # K
cooling_rate = -0.4      # K/min
C0 = 0.0256              # g/g water

t_final = 80             # min
t_eval = np.linspace(0, t_final, 300)

# -----------------------------
# Random kinetic parameters
# -----------------------------
np.random.seed(1)  # makes random values repeatable

kb = np.random.uniform(2.0e18, 5.0e20)
kg = np.random.uniform(0.001, 0.5)

print(f"Random kb = {kb:.3e}")
print(f"Random kg = {kg:.4f}")

# -----------------------------
# Temperature and solubility
# -----------------------------
def temperature(t):
    return T0 + cooling_rate * t

def Csat(T):
    return 1.5846e-5*T**2 - 9.0567e-3*T + 1.3066

# -----------------------------
# ODE model
# y = [mu0, mu1, mu2, mu3, C]
# -----------------------------
def crystallization_odes(t, y):
    mu0, mu1, mu2, mu3, C = y

    T = temperature(t)
    C_saturation = Csat(T)

    supersat = C - C_saturation

    # Avoid negative supersaturation causing invalid powers
    if supersat <= 0:
        B = 0
        G = 0
    else:
        B = kb * supersat**b
        G = kg * supersat**g

    dmu0dt = B
    dmu1dt = G * mu0
    dmu2dt = 2 * G * mu1
    dmu3dt = 3 * G * mu2
    dCdt = -3 * kv * rho_c * G * mu2

    return [dmu0dt, dmu1dt, dmu2dt, dmu3dt, dCdt]

# -----------------------------
# Initial conditions
# -----------------------------
mu0_0 = 0
mu1_0 = 0
mu2_0 = 0
mu3_0 = 0

y0 = [mu0_0, mu1_0, mu2_0, mu3_0, C0]

# -----------------------------
# Solve ODE system
# -----------------------------
solution = solve_ivp(
    crystallization_odes,
    [0, t_final],
    y0,
    t_eval=t_eval,
    method="BDF",        # good for potentially stiff crystallization systems
    rtol=1e-6,
    atol=1e-12
)

t = solution.t
mu0 = solution.y[0]
mu1 = solution.y[1]
mu2 = solution.y[2]
mu3 = solution.y[3]
C = solution.y[4]

T = temperature(t)
C_saturation = Csat(T)
supersaturation = C - C_saturation

# Avoid division by zero
mean_size = np.zeros_like(t)
CV = np.zeros_like(t)

valid = mu0 > 0
mean_size[valid] = mu1[valid] / mu0[valid]

valid_cv = (mu1 > 0) & (mu0 > 0)
CV[valid_cv] = np.sqrt((mu2[valid_cv] * mu0[valid_cv]) / (mu1[valid_cv]**2) - 1)

# -----------------------------
# Print final values
# -----------------------------
print("\nFinal results:")
print(f"Final temperature = {T[-1]:.2f} K")
print(f"Final concentration = {C[-1]:.5f} g/g water")
print(f"Final Csat = {C_saturation[-1]:.5f} g/g water")
print(f"Final supersaturation = {supersaturation[-1]:.5e} g/g water")
print(f"Final mean crystal size = {mean_size[-1]:.5e} m")
print(f"Final CV = {CV[-1]:.4f}")

# -----------------------------
# Plot concentration
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(t, C, label="Concentration")
plt.plot(t, C_saturation, "--", label="Saturation concentration")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (g/g water)")
plt.title("Concentration and Solubility vs Time")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Plot supersaturation
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(t, supersaturation)
plt.xlabel("Time (min)")
plt.ylabel("Supersaturation, C - Csat")
plt.title("Supersaturation vs Time")
plt.grid(True)
plt.show()

# -----------------------------
# Plot mean crystal size
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(t, mean_size)
plt.xlabel("Time (min)")
plt.ylabel("Mean crystal size, d (m)")
plt.title("Mean Crystal Size vs Time")
plt.grid(True)
plt.show()

# -----------------------------
# Plot moments
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(t, mu0, label="mu0")
plt.plot(t, mu1, label="mu1")
plt.plot(t, mu2, label="mu2")
plt.plot(t, mu3, label="mu3")
plt.xlabel("Time (min)")
plt.ylabel("Moment value")
plt.title("Crystal Population Moments vs Time")
plt.legend()
plt.grid(True)
plt.show()
