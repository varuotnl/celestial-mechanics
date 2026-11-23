
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import total_energy
from gravity import G
from integrator import verlet_step

# --------------------------------------------------
# 1. Simulation parameters
# --------------------------------------------------

# Fixed random seed for reproducibility
rng = np.random.default_rng(42)

N_BODIES = 100
MASS = 1.0e20
CLOUD_RADIUS = 1.0e7
SOFTENING = 2.0e5
N_STEPS = 2000

masses = np.full(N_BODIES, MASS)

# --------------------------------------------------
# 2. Generate initial particle positions
# --------------------------------------------------

# Uniform random distribution inside a circular disk
angles = rng.uniform(0, 2 * np.pi, N_BODIES)

radii = CLOUD_RADIUS * np.sqrt(
    rng.uniform(0, 1, N_BODIES)
)

positions = np.column_stack((
    radii * np.cos(angles),
    radii * np.sin(angles)
))

# Shift centre of mass to the origin
positions -= np.mean(positions, axis=0)

# Save initial positions for visualization
initial_positions = positions.copy()

# --------------------------------------------------
# 3. Assign initial rotational velocities
# --------------------------------------------------

total_mass = masses.sum()

# Approximate angular velocity
omega = np.sqrt(
    G * total_mass / CLOUD_RADIUS**3
)

# Tangential velocities
velocities = omega * np.column_stack((
    -positions[:, 1],
    positions[:, 0]
))

# Remove net momentum
velocities -= np.mean(velocities, axis=0)

# --------------------------------------------------
# 4. Define simulation timestep
# --------------------------------------------------

dynamical_time = np.sqrt(
    CLOUD_RADIUS**3 / (G * total_mass)
)

dt = dynamical_time / 400

# --------------------------------------------------
# 5. Initialize energy storage
# --------------------------------------------------

energies = np.zeros(N_STEPS + 1)

energies[0] = total_energy(
    positions,
    velocities,
    masses,
    softening=SOFTENING
)

# --------------------------------------------------
# 6. Run the N-body simulation
# --------------------------------------------------

for step in range(N_STEPS):

    positions, velocities = verlet_step(
        positions,
        velocities,
        masses,
        dt,
        softening=SOFTENING
    )

    # Calculate total energy at each timestep
    energies[step + 1] = total_energy(
        positions,
        velocities,
        masses,
        softening=SOFTENING
    )

# --------------------------------------------------
# 7. Plot initial and final configurations
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    initial_positions[:, 0] / CLOUD_RADIUS,
    initial_positions[:, 1] / CLOUD_RADIUS,
    s=15
)

axes[0].set_title("Initial rotating cloud")

axes[1].scatter(
    positions[:, 0] / CLOUD_RADIUS,
    positions[:, 1] / CLOUD_RADIUS,
    s=15
)

axes[1].set_title("Final cloud configuration")

for ax in axes:
    ax.set_xlabel("X / initial cloud radius")
    ax.set_ylabel("Y / initial cloud radius")
    ax.axis("equal")
    ax.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Calculate relative energy error
# --------------------------------------------------

initial_energy = energies[0]

relative_error = (
    (energies - initial_energy) / abs(initial_energy)
)

times = np.arange(N_STEPS + 1) * dt

# --------------------------------------------------
# 9. Plot energy conservation
# --------------------------------------------------

plt.figure(figsize=(8, 4))

plt.plot(
    times / dynamical_time,
    relative_error
)

plt.xlabel("Time / dynamical timescale")
plt.ylabel("Relative energy error")
plt.title("100-body simulation: Energy conservation")

plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. Print numerical results
# --------------------------------------------------

print(f"Number of bodies: {N_BODIES}")
print(f"Softening length: {SOFTENING:.2e} m")
print(f"Total simulated time: {N_STEPS * dt:.2f} s")

print(
    f"Initial total energy: "
    f"{initial_energy:.4e} J"
)

print(
    f"Final total energy: "
    f"{energies[-1]:.4e} J"
)

print(
    f"Maximum relative energy error: "
    f"{np.max(np.abs(relative_error)):.4e}"
)
