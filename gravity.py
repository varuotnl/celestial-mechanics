
import numpy as np

# Gravitational constant (m^3 kg^-1 s^-2)
G = 6.67430e-11


def gravitational_force(position1, position2, mass1, mass2):
    """Calculate the gravitational force acting on body 1 due to body 2."""

    displacement = position2 - position1
    distance = np.linalg.norm(displacement)

    if distance == 0:
        raise ValueError("Two bodies cannot occupy the same position.")

    force = G * mass1 * mass2 * displacement / distance**3

    return force
