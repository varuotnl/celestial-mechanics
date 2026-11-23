
import numpy as np

from gravity import G


def total_energy(positions, velocities, masses):
    """Calculate total kinetic and gravitational potential energy."""

    kinetic = 0.5 * np.sum(
        masses[:, None] * velocities**2
    )

    potential = 0.0
    n = len(masses)

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(
                positions[j] - positions[i]
            )
            if distance == 0:
                raise ValueError("Two bodies occupy the same position.")

            potential -= G * masses[i] * masses[j] / distance

    return kinetic + potential
