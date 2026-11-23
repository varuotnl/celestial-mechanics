
import numpy as np

from gravity import G


def total_energy(positions, velocities, masses, softening=0.0):
    """Calculate total kinetic and gravitational potential energy."""

    kinetic = 0.5 * np.sum(
        masses[:, None] * velocities**2
    )

    potential = 0.0
    n = len(masses)

    for i in range(n):
        for j in range(i + 1, n):
            displacement = positions[j] - positions[i]
            distance_squared = np.dot(displacement, displacement)

            if distance_squared == 0 and softening == 0:
                raise ValueError("Two bodies occupy the same position.")

            potential -= (
                G * masses[i] * masses[j]
                / np.sqrt(distance_squared + softening**2)
            )

    return kinetic + potential
