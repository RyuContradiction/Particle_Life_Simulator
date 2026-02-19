import numpy as np
from Backend.Particles import Particles
from Backend.Simulation import Simulation
from Frontend.Visualize import Visualize



if __name__ == '__main__':
    N: int = 1_000
    x: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size= N)
    y: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size= N)
    particles: Particles = Particles(x, y, velocity_x=np.zeros(N), velocity_y=np.zeros(N), types=np.random.randint(0, 5, size=N), radius=20)
    simulation: Simulation = Simulation(particles=particles)
    visualize: Visualize = Visualize(x, y, particles, simulation)
    visualize.start()




