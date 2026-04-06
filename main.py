import numpy as np
from numba.typed import Dict
from numba.core import types
from Backend.Particles import Particles
from Backend.Simulation import Simulation
from Frontend.Visualize import Visualize



if __name__ == '__main__':
    N: int = 1_000
    x: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size= N)
    y: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size= N)
    types_n = np.random.randint(0, 5, size=N)
    mapping = Dict.empty(key_type=types.int64, value_type=types.int64)
    mapping[1] = 2.0
    mapping[2] = 2.2
    mapping[3] = 5.0
    mapping[4] = 1.2
    mapping[0] = 3.4
    forces=np.array([mapping.get(x, x) for x in types_n])
    force = np.column_stack((forces,forces))


    
    particles: Particles = Particles(5, force, x, y, velocity_x=np.zeros(N), velocity_y=np.zeros(N), types=types_n, radius=20 )
    simulation: Simulation = Simulation(particles=particles)
    visualize: Visualize = Visualize(x, y, particles, simulation)
    visualize.start()




