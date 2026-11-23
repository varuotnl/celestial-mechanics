
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import total_energy
from gravity import G
from integrator import verlet_step

# Simulation parameters (SI units)
MASS_1 = 1.0e24
MASS_2 = 1.0e24
SEPARATION = 1.0e7
N_STEPS = 4000

masses = np.array([MASS_1, MASS_2])
total_mass = masses.sum()

# Circular and escape speeds
circular_speed = np.sqrt(G * total_mass / SEPARATION)
escape_speed = np.sqrt(2) * circular_speed

period = 2 * np.pi * SEPARATION / circular_speed
dt = period / 2000

orbit_cases = {
    "Elliptical": 0.8 * circular_speed,
    "Parabolic": escape_speed,
    "Hyperbolic": 1.5 * escape_speed,
}

# Characteristic energy scale for normalization
energy_scale = G * MASS_1 * MASS_2 / SEPARATION

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for orbit_name, relative_speed in orbit_cases.items():

    positions = np.array([
        [-SEPARATION / 2, 0.0],
        [SEPARATION / 2, 0.0]
    ])

    velocities = np.array([
        [0.0, -relative_speed / 2],
        [0.0, relative_speed / 2]
    ])

    relative_positions = np.zeros((N_STEPS + 1, 2))
    energies = np.zeros(N_STEPS + 1)

    relative_positions[0] = positions[1] - positions[0]
    energies[0] = total_energy(positions, velocities, masses)

    for step in range(N_STEPS):
        positions, velocities = verlet_step(
            positions, velocities, masses, dt
        )

        relative_positions[step + 1] = (
            positions[1] - positions[0]
        )

        energies[step + 1] = total_energy(
            positions, velocities, masses
        )

    # Analytical classification from initial conditions
    specific_energy = (
        0.5 * relative_speed**2
        - G * total_mass / SEPARATION
    )

    # Energy conservation error
    energy_error = (energies - energies[0]) / energy_scale

    max_error = np.max(np.abs(energy_error))

    print(f"\n{orbit_name}")
    print(f"Initial specific energy: {specific_energy:.4e} J/kg")
    print(f"Maximum normalized energy error: {max_error:.4e}")

    axes[0].plot(
        relative_positions[:, 0] / SEPARATION,
        relative_positions[:, 1] / SEPARATION,
        label=orbit_name
    )

    times = np.arange(N_STEPS + 1) * dt

    axes[1].plot(
        times / period,
        energy_error,
        label=orbit_name
    )

# Plot 1: Orbital trajectories
axes[0].scatter(0, 0, color="black", marker="*")
axes[0].set_xlabel("Relative X / initial separation")
axes[0].set_ylabel("Relative Y / initial separation")
axes[0].set_title("Two-body orbital trajectories")
axes[0].axis("equal")
axes[0].grid(True)
axes[0].legend()

# Plot 2: Energy conservation
axes[1].set_xlabel("Time / circular orbital period")
axes[1].set_ylabel("Energy change / characteristic energy")
axes[1].set_title("Numerical energy conservation")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("results/two_body_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()