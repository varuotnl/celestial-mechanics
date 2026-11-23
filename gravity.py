
import numpy as np

G = 6.67430e-11


def gravitational_force(
    position1, position2, mass1, mass2, softening=0.0
):
    """Calculate gravitational force on body 1 due to body 2."""

    displacement = position2 - position1
    distance_squared = np.dot(displacement, displacement)

    if distance_squared == 0 and softening == 0:
        raise ValueError("Two bodies occupy the same position.")

    softened_distance = (
        distance_squared + softening**2
    ) ** 1.5

    force = (
        G * mass1 * mass2
        * displacement / softened_distance
    )

    return force
