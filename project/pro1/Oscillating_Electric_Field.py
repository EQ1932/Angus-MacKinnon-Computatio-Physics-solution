import numpy as np
import matplotlib.pyplot as plt


""" omega_list = [0.1, 0.5, 1.0, 2.0, 5.0]

for omega in omega_list:
    # Repeat the simulation with the current omega
    # (Include the simulation code here)
    # Plot or save the results for analysis """

# Constants
q = 1.0       # Charge of the particle
m = 1.0       # Mass of the particle
gamma = 0.1   # Friction coefficient
B = np.array([0.0, 0.0, 1.0])  # Magnetic field vector (along z-axis)
E0 = 1.0      # Amplitude of the electric field
omega = 1.0   # Angular frequency of the electric field

# Function to compute acceleration
def acceleration(v, t, E0, omega, B, q, m, gamma):
    # Time-dependent electric field
    E = np.array([E0 * np.cos(omega * t), 0.0, 0.0])
    # Lorentz force equation: a = (q/m)*(E + v x B) - (gamma/m)*v
    return (q / m) * (E + np.cross(v, B)) - (gamma / m) * v

# RK4 integration method
def rk4(v, t, delta_t, E0, omega, B, q, m, gamma):
    k1 = acceleration(v, t, E0, omega, B, q, m, gamma)
    k2 = acceleration(v + 0.5 * delta_t * k1, t + 0.5 * delta_t, E0, omega, B, q, m, gamma)
    k3 = acceleration(v + 0.5 * delta_t * k2, t + 0.5 * delta_t, E0, omega, B, q, m, gamma)
    k4 = acceleration(v + delta_t * k3, t + delta_t, E0, omega, B, q, m, gamma)
    return v + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Simulation parameters
delta_t = 0.01   # Time step size
t_max = 50.0     # Maximum simulation time
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
    t = time[i - 1]
    v_values[i] = rk4(v_values[i - 1], t, delta_t, E0, omega, B, q, m, gamma)
    x_values[i] = x_values[i - 1] + v_values[i - 1] * delta_t

# Plotting velocity components over time
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, v_values[:, 0], label='v_x')
plt.xlabel('Time')
plt.ylabel('v_x')
plt.title('Velocity Component v_x over Time')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, v_values[:, 1], label='v_y', color='orange')
plt.xlabel('Time')
plt.ylabel('v_y')
plt.title('Velocity Component v_y over Time')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, v_values[:, 2], label='v_z', color='green')
plt.xlabel('Time')
plt.ylabel('v_z')
plt.title('Velocity Component v_z over Time')
plt.grid(True)

plt.tight_layout()
plt.show()



# Plotting particle trajectory in the xy-plane
plt.figure(figsize=(8, 6))
plt.plot(x_values[:, 0], x_values[:, 1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Particle Trajectory in the xy-plane')
plt.grid(True)
plt.show()

