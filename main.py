
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import total_energy
from integrator import verlet_step

# Simulation parameters (SI units)
G = 6.67430e-11
MASS_1 = 1.0e24
MASS_2 = 1.0e24
SEPARATION = 1.0e7
N_STEPS = 2000

masses = np.array([MASS_1, MASS_2])
total_mass = masses.sum()

# Calculate circular orbital speed and period
relative_speed = np.sqrt(G * total_mass / SEPARATION)
period = 2 * np.pi * SEPARATION / relative_speed
dt = period / N_STEPS

# Initial positions around the centre of mass
positions = np.array([
    [-SEPARATION / 2, 0.0],
    [SEPARATION / 2, 0.0]
])

# Initial velocities
velocities = np.array([
    [0.0, -relative_speed / 2],
    [0.0, relative_speed / 2]
])

# Arrays to store trajectories and energies
trajectories = np.zeros((N_STEPS + 1, 2, 2))
energies = np.zeros(N_STEPS + 1)

trajectories[0] = positions
energies[0] = total_energy(positions, velocities, masses)

# Run the simulation
for step in range(N_STEPS):
    positions, velocities = verlet_step(
        positions, velocities, masses, dt
    )

    trajectories[step + 1] = positions
    energies[step + 1] = total_energy(
        positions, velocities, masses
    )

# Plot 1: Orbital trajectories
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

plt.scatter(
    0, 0,
    color="black",
    marker="+",
    label="Centre of mass"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Two-body circular orbit using Velocity Verlet")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()

# Calculate relative energy error
relative_error = (energies - energies[0]) / abs(energies[0])

# Time values
times = np.arange(N_STEPS + 1) * dt

# Plot 2: Energy conservation
plt.figure(figsize=(8, 4))
plt.plot(times, relative_error)

plt.xlabel("Time (s)")
plt.ylabel("Relative energy error")
plt.title("Energy conservation - Velocity Verlet")
plt.grid(True)
plt.savefig("results/two_body.png", dpi=300, bbox_inches="tight")
plt.show()

# Print numerical results
print(f"Orbital period: {period:.2f} seconds")
print(f"Timestep: {dt:.2f} seconds")
print(
    f"Maximum relative energy error: "
    f"{np.max(np.abs(relative_error)):.3e}"
)
