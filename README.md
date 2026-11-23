
# Celestial Mechanics — Python Project

Final project for Introduction to Programming in Python,
University of Twente, Fall 2026.

## Overview

This project simulates gravitational interactions between
multiple bodies using Newton's law of gravitation and
the Velocity Verlet numerical integration method.

## Simulations

- Two-body circular orbit
- Elliptical, parabolic, and hyperbolic trajectories
- Three-body rotating triangle
- 100-body rotating gravitational cloud
- Energy conservation and timestep convergence tests
- Animation of the 100-body simulation

## Project files

- `main.py` — Two-body simulation
- `gravity.py` — Gravitational acceleration calculations
- `integrator.py` — Velocity Verlet integration
- `diagnostics.py` — Energy calculations
- `hundred_body.py` — 100-body simulation
- `animation.py` — Animated 100-body simulation
- `pyproject.toml` — Project dependencies

Additional Python scripts contain the orbital
classification, three-body, and convergence experiments.

## Requirements

- Python 3.14 or later
- NumPy
- Matplotlib
- Pillow

## Running the project

Open the project folder in Positron.

Install the project dependencies in the active virtual
environment:

```powershell
python -m pip install -e .
```

Run individual simulations, for example:

```powershell
python main.py
python hundred_body.py
python animation.py
```

## Reproducibility

The 100-body simulation uses a fixed random seed (42)
to reproduce the initial particle distribution.

Numerical parameters are defined in the simulation scripts.

## GitHub repository

https://github.com/varuotnl/celestial-mechanics
