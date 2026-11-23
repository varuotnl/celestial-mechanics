
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import total_energy
from gravity import G
from integrator import verlet_step

# Simulation parameters (SI units)
MASS = 1.0e24
SIDE_LENGTH = 1.0e7
N_STEPS = 4000
N_ORBITS = 3

masses = np.array([MASS, MASS, MASS])

# Distance from triangle centre to each vertex
radius = SIDE_LENGTH / np.sqrt(3)

# Initial angular positions of the three bodies
angles = np.array([
    0.0,
    2 * np.pi / 3,
    4 * np.pi / 3
])

# Initial positions: equilateral triangle
positions = radius * np.column_stack((
    np.cos(angles),
    np.sin(angles)
))

# Angular velocity for the rotating triangle solution
omega = np.sqrt(3 * G * MASS / SIDE_LENGTH**3)

# Tangential velocities
velocities = omega * np.column_stack((
    -positions[:, 1],
    positions[:, 0]
))

# Orbital period and timestep
period = 2 * np.pi / omega
dt = N_ORBITS * period / N_STEPS

# Store trajectories and energy
trajectories = np.zeros((N_STEPS + 1, 3, 2))
energies = np.zeros(N_STEPS + 1)

trajectories[0] = positions
energies[0] = total_energy(positions, velocities, masses)

# Numerical integration
for step in range(N_STEPS):
    positions, velocities = verlet_step(
        positions, velocities, masses, dt
    )

    trajectories[step + 1] = positions
    energies[step + 1] = total_energy(
        positions, velocities, masses
    )

# Plot trajectories
plt.figure(figsize=(7, 7))

for i in range(3):
    plt.plot(
        trajectories[:, i, 0],
        trajectories[:, i, 1],
        label=f"Body {i + 1}"
    )

    plt.scatter(
        trajectories[0, i, 0],
        trajectories[0, i, 1],
        marker="o",
        s=35
    )

plt.scatter(
    0, 0,
    color="black",
    marker="+",
    label="Centre of mass"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Three-body rotating triangle simulation")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/three_body.png", dpi=300, bbox_inches="tight")
plt.show()

# Energy conservation
relative_error = (
    (energies - energies[0]) / abs(energies[0])
)

times = np.arange(N_STEPS + 1) * dt

plt.figure(figsize=(8, 4))
plt.plot(times / period, relative_error)

plt.xlabel("Time / orbital period")
plt.ylabel("Relative energy error")
plt.title("Three-body energy conservation")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/three_body_energy.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Orbital period: {period:.2f} seconds")
print(f"Maximum relative energy error: "
      f"{np.max(np.abs(relative_error)):.4e}")
