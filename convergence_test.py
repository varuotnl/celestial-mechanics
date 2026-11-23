
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import total_energy
from gravity import G
from integrator import verlet_step

# Physical parameters (SI units)
MASS_1 = 1.0e24
MASS_2 = 1.0e24
SEPARATION = 1.0e7

masses = np.array([MASS_1, MASS_2])
total_mass = masses.sum()

# Initial elliptical orbit
circular_speed = np.sqrt(G * total_mass / SEPARATION)
relative_speed = 0.8 * circular_speed

period = 2 * np.pi * SEPARATION / circular_speed

initial_positions = np.array([
    [-SEPARATION / 2, 0.0],
    [SEPARATION / 2, 0.0]
])

initial_velocities = np.array([
    [0.0, -relative_speed / 2],
    [0.0, relative_speed / 2]
])

# Use the same physical duration for every test
simulation_time = 2 * period

# Different timestep resolutions
steps_per_period = [500, 1000, 2000, 4000]

energy_scale = G * MASS_1 * MASS_2 / SEPARATION

errors = []

for resolution in steps_per_period:

    dt = period / resolution
    n_steps = 2 * resolution

    # Reset initial conditions for each run
    positions = initial_positions.copy()
    velocities = initial_velocities.copy()

    initial_energy = total_energy(
        positions, velocities, masses
    )

    max_error = 0.0

    for step in range(n_steps):

        positions, velocities = verlet_step(
            positions, velocities, masses, dt
        )

        current_energy = total_energy(
            positions, velocities, masses
        )

        error = abs(
            (current_energy - initial_energy) / energy_scale
        )

        max_error = max(max_error, error)

    errors.append(max_error)

    print(
        f"Steps/period: {resolution:4d} | "
        f"dt: {dt:.4f} s | "
        f"Max error: {max_error:.6e}"
    )

# Plot convergence
timesteps = period / np.array(steps_per_period)

plt.figure(figsize=(8, 5))

plt.loglog(
    timesteps,
    errors,
    "o-",
    label="Measured energy error"
)

# Reference line for second-order convergence
reference = errors[0] * (timesteps / timesteps[0])**2

plt.loglog(
    timesteps,
    reference,
    "--",
    label="Second-order reference"
)

plt.xlabel("Timestep (s)")
plt.ylabel("Maximum normalized energy error")
plt.title("Velocity Verlet timestep convergence")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

# Calculate observed error reduction
print("\nError reduction when timestep is halved:")

for i in range(len(errors) - 1):
    ratio = errors[i] / errors[i + 1]

    print(
        f"{steps_per_period[i]} -> "
        f"{steps_per_period[i + 1]}: "
        f"{ratio:.2f}x"
    )
