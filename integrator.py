
import numpy as np

from gravity import gravitational_force


def calculate_accelerations(positions, masses):
    """Calculate gravitational acceleration for every body."""

    n = len(masses)
    accelerations = np.zeros_like(positions, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            force = gravitational_force(
                positions[i], positions[j], masses[i], masses[j]
            )

            accelerations[i] += force / masses[i]
            accelerations[j] -= force / masses[j]

    return accelerations


def verlet_step(positions, velocities, masses, dt):
    """Advance all bodies by one velocity Verlet timestep."""

    old_accelerations = calculate_accelerations(positions, masses)

    new_positions = (
        positions
        + velocities * dt
        + 0.5 * old_accelerations * dt**2
    )

    new_accelerations = calculate_accelerations(new_positions, masses)

    new_velocities = (
        velocities
        + 0.5 * (old_accelerations + new_accelerations) * dt
    )

    return new_positions, new_velocities
