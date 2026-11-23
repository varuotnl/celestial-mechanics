
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from gravity import G
from integrator import verlet_step

# --------------------------------------------------
# 1. Simulation parameters
# --------------------------------------------------

rng = np.random.default_rng(42)

N_BODIES = 100
MASS = 1.0e20
CLOUD_RADIUS = 1.0e7
SOFTENING = 2.0e5

N_STEPS = 2000
FRAME_INTERVAL = 10

masses = np.full(N_BODIES, MASS)

# --------------------------------------------------
# 2. Initial positions
# --------------------------------------------------

angles = rng.uniform(0, 2 * np.pi, N_BODIES)
radii = CLOUD_RADIUS * np.sqrt(
    rng.uniform(0, 1, N_BODIES)
)

positions = np.column_stack((
    radii * np.cos(angles),
    radii * np.sin(angles)
))

positions -= np.mean(positions, axis=0)

# --------------------------------------------------
# 3. Initial velocities
# --------------------------------------------------

total_mass = masses.sum()

omega = np.sqrt(
    G * total_mass / CLOUD_RADIUS**3
)

velocities = omega * np.column_stack((
    -positions[:, 1],
    positions[:, 0]
))

velocities -= np.mean(velocities, axis=0)

# --------------------------------------------------
# 4. Timestep and trajectory storage
# --------------------------------------------------

dynamical_time = np.sqrt(
    CLOUD_RADIUS**3 / (G * total_mass)
)

dt = dynamical_time / 400

# Store one configuration every 10 timesteps
frames = [positions.copy()]

# --------------------------------------------------
# 5. Run the simulation
# --------------------------------------------------

for step in range(N_STEPS):

    positions, velocities = verlet_step(
        positions,
        velocities,
        masses,
        dt,
        softening=SOFTENING
    )

    if (step + 1) % FRAME_INTERVAL == 0:
        frames.append(positions.copy())

frames = np.array(frames)

print(f"Stored animation frames: {len(frames)}")

# --------------------------------------------------
# 6. Create animation
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

# Use fixed axis limits for the entire animation
max_extent = np.max(np.abs(frames)) / CLOUD_RADIUS
plot_limit = max(1.2, 1.1 * max_extent)

scatter = ax.scatter([], [], s=18)

ax.set_xlim(-plot_limit, plot_limit)
ax.set_ylim(-plot_limit, plot_limit)
ax.set_aspect("equal")

ax.set_xlabel("X / initial cloud radius")
ax.set_ylabel("Y / initial cloud radius")
ax.grid(True)

title = ax.set_title("100-body gravitational simulation")


def update(frame_number):
    """Update particle positions for one animation frame."""

    current_positions = (
        frames[frame_number] / CLOUD_RADIUS
    )

    scatter.set_offsets(current_positions)

    elapsed_time = (
        frame_number * FRAME_INTERVAL * dt
        / dynamical_time
    )

    title.set_text(
        f"100-body simulation | Time = "
        f"{elapsed_time:.2f} dynamical times"
    )

    return scatter, title


animation = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False
)

# --------------------------------------------------
# 7. Save animation
# --------------------------------------------------

animation.save(
    "hundred_body_animation.gif",
    writer=PillowWriter(fps=20)
)

plt.close(fig)

print("Animation saved as hundred_body_animation.gif")
