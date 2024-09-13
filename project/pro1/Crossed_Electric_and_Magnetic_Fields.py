import numpy as np
import matplotlib.pyplot as plt

# Constants
q = 1.0       # Charge of the particle
m = 1.0       # Mass of the particle
gamma = 0.1   # Friction coefficient
B = np.array([0.0, 0.0, 1.0])  # Magnetic field vector (along z-axis)
E = np.array([1.0, 0.0, 0.0])  # Electric field vector (along x-axis)

# Function to compute acceleration
def acceleration(v, E, B, q, m, gamma):
    # Lorentz force equation: a = (q/m)*(E + v x B) - (gamma/m)*v
    return (q / m) * (E + np.cross(v, B)) - (gamma / m) * v

# RK4 integration method
def rk4(v, E, B, q, m, gamma, delta_t):
    k1 = acceleration(v, E, B, q, m, gamma)
    k2 = acceleration(v + 0.5 * delta_t * k1, E, B, q, m, gamma)
    k3 = acceleration(v + 0.5 * delta_t * k2, E, B, q, m, gamma)
    k4 = acceleration(v + delta_t * k3, E, B, q, m, gamma)
    return v + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Simulation parameters
delta_t = 0.01   # Time step size
t_max = 20.0     # Maximum simulation time
v0 = np.array([0.0, 0.0, 0.0])  # Initial velocity
x0 = np.array([0.0, 0.0, 0.0])  # Initial position

# Time array
time = np.arange(0, t_max, delta_t)

# Arrays to store velocity and position
v_values = np.zeros((len(time), 3))
x_values = np.zeros((len(time), 3))
v_values[0] = v0
x_values[0] = x0

# Time-stepping loop
for i in range(1, len(time)):
    v_values[i] = rk4(v_values[i - 1], E, B, q, m, gamma, delta_t)
    x_values[i] = x_values[i - 1] + v_values[i - 1] * delta_t

# Plotting velocity components over time
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, v_values[:, 0], label='v_x')
plt.plot(time, v_values[:, 1], label='v_y')
plt.plot(time, v_values[:, 2], label='v_z')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity Components over Time')
plt.legend()
plt.grid(True)

# Plotting particle trajectory in the xy-plane
plt.subplot(2, 1, 2)
plt.plot(x_values[:, 0], x_values[:, 1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Particle Trajectory in the xy-plane')
plt.grid(True)

plt.tight_layout()
plt.show()
