import numpy as np
import matplotlib.pyplot as plt

# Constants
q = 1.0       # Charge
m = 1.0       # Mass
gamma = 0.1   # Friction coefficient
B = np.array([0, 0, 1.0])  # Magnetic field in z-direction

# Cross-product for v x B
def v_cross_B(v, B):
    return np.cross(v, B)

# Right-hand side of the ODE system
def acceleration(v, B, q, m, gamma):
    return (q/m) * v_cross_B(v, B) - (gamma/m) * v

# RK4 implementation
def rk4(v, B, q, m, gamma, delta_t):
    k1 = acceleration(v, B, q, m, gamma)
    k2 = acceleration(v + 0.5 * delta_t * k1, B, q, m, gamma)
    k3 = acceleration(v + 0.5 * delta_t * k2, B, q, m, gamma)
    k4 = acceleration(v + delta_t * k3, B, q, m, gamma)
    
    return v + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Parameters for simulation
delta_t = 0.01  # Time step
t_max = 10.0    # Maximum time
v0 = np.array([1.0, 0.0, 0.0])  # Initial velocity

# Time array
time = np.arange(0, t_max, delta_t)

# Initialize velocity array
v_values = np.zeros((len(time), 3))
v_values[0] = v0

# Time-stepping loop
for i in range(1, len(time)):
    v_values[i] = rk4(v_values[i-1], B, q, m, gamma, delta_t)

# Plot results
plt.plot(time, v_values[:, 0], label='v_x')
plt.plot(time, v_values[:, 1], label='v_y')
plt.plot(time, v_values[:, 2], label='v_z')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity of a Charged Particle in a Magnetic Field')
plt.legend()
plt.grid(True)
plt.show()

