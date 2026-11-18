
import numpy as np
import matplotlib.pyplot as plt

# Define the initial positions of two celestial bodies
body_1 = np.array([0.0, 0.0])
body_2 = np.array([3.0, 2.0])

# Plot their positions
plt.scatter(body_1[0], body_1[1], label="Body 1")
plt.scatter(body_2[0], body_2[1], label="Body 2")

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Initial positions of two celestial bodies")
plt.axis("equal")
plt.legend()
plt.show()
