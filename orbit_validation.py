
import matplotlib.pyplot as plt
import numpy as np

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

# Characteristic circular orbital period
period = 2 * np.pi * SEPARATION / circular_speed
dt = period / 2000

# Different initial relative speeds
orbit_cases = {
    "Elliptical": 0.8 * circular_speed,
    "Parabolic": escape_speed,
    "Hyperbolic": 1.5 * escape_speed,
}

plt.figure(figsize=(9, 7))

for orbit_name, relative_speed in orbit_cases.items():

    # Positions relative to the centre of mass
    positions = np.array([
        [-SEPARATION / 2, 0.0],
        [SEPARATION / 2, 0.0]
    ])

    # Opposite velocities keep total momentum zero
    velocities = np.array([
        [0.0, -relative_speed / 2],
        [0.0, relative_speed / 2]
    ])

    # Store relative positions (body 2 relative to body 1)
    relative_positions = np.zeros((N_STEPS + 1, 2))
    relative_positions[0] = positions[1] - positions[0]

    # Numerical simulation
    for step in range(N_STEPS):
        positions, velocities = verlet_step(
            positions, velocities, masses, dt
        )

        relative_positions[step + 1] = (
            positions[1] - positions[0]
        )

    plt.plot(
        relative_positions[:, 0] / SEPARATION,
        relative_positions[:, 1] / SEPARATION,
        label=orbit_name
    )

# Origin represents body 1 in relative coordinates
plt.scatter(
    0, 0,
    color="black",
    marker="*",
    s=100,
    label="Relative origin"
)

plt.xlabel("Relative X position / initial separation")
plt.ylabel("Relative Y position / initial separation")
plt.title("Two-body orbital trajectories")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Circular relative speed: {circular_speed:.2f} m/s")
print(f"Escape relative speed: {escape_speed:.2f} m/s")
