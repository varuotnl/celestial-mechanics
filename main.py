
import matplotlib.pyplot as plt
import numpy as np

from integrator import verlet_step

# Simulation parameters (SI units)
G = 6.67430e-11
MASS_1 = 1.0e24
MASS_2 = 1.0e24
SEPARATION = 1.0e7
N_STEPS = 2000

masses = np.array([MASS_1, MASS_2])
total_mass = masses.sum()

# Circular relative orbital speed and period
relative_speed = np.sqrt(G * total_mass / SEPARATION)
period = 2 * np.pi * SEPARATION / relative_speed
dt = period / N_STEPS

# Both bodies orbit their common centre of mass
positions = np.array([
    [-SEPARATION / 2, 0.0],
    [SEPARATION / 2, 0.0]
])

velocities = np.array([
    [0.0, -relative_speed / 2],
    [0.0, relative_speed / 2]
])

# Store trajectories
trajectories = np.zeros((N_STEPS + 1, 2, 2))
trajectories[0] = positions

# Run simulation
for step in range(N_STEPS):
    positions, velocities = verlet_step(
        positions, velocities, masses, dt
    )
    trajectories[step + 1] = positions

# Plot trajectories
plt.figure(figsize=(7, 7))

plt.plot(
    trajectories[:, 0, 0],
    trajectories[:, 0, 1],
    label="Body 1"
)

plt.plot(
    trajectories[:, 1, 0],
    trajectories[:, 1, 1],
    label="Body 2"
)

plt.scatter(0, 0, color="black", marker="+", label="Centre of mass")

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Two-body circular orbit using Velocity Verlet")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()

print(f"Orbital period: {period:.2f} seconds")
print(f"Timestep: {dt:.2f} seconds")
