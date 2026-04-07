import numpy as np
from Backend.Particles import Particles
from Backend.Simulation import Simulation
from Frontend.Visualize import Visualize

if __name__ == '__main__':
    N: int = 2_000

    x: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size=N)
    y: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size=N)
    types_n: np.ndarray = np.random.randint(0, 5, size=N)

    lookup = np.array([2, 2, 2, 2, 2], dtype=np.float64)

    forces = lookup[types_n]
    force = np.column_stack((forces, forces))

    particles: Particles = Particles(
        2,
        force,
        x,
        y,
        velocity_x=np.zeros(N),
        velocity_y=np.zeros(N),
        types=types_n,
        radius=15
    )

    simulation: Simulation = Simulation(particles=particles)
    visualize: Visualize = Visualize(x, y, particles, simulation)
    visualize.start()


