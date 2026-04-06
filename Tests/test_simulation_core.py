# tests/test_simulation_core.py
import numpy as np
import pytest
from numba.typed import Dict
from numba.core import types

from Backend.Particles import Particles
from Backend.Simulation import Simulation


def make_small_sim():
    """
    3 Partikel in 1D Linie:
      p0 at (0,0)
      p1 at (5,0)
      p2 at (100,0)  -> weit weg
    radius=10 sollte p0<->p1 als Nachbarn finden, p2 nicht.
    """
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
    # einfache Interaktionsmatrix (5x5 wie bei dir), damit indexing sicher ist
    W = np.array([
        [1, -1,  1, -1,  1],
        [-1, 1, -1,  1, -1],
        [1, -1,  1,  1, -1],
        [-1, 1,  1, -1,  1],
        [1, -1, -1,  1,  1],
    ], dtype=np.float64)

    sim = Simulation(interactionmatrix=W, particles=particles)
    return sim


#def test_check_interactions_no_neighbours_returns_empty():
    sim = make_small_sim()
    # p2 ist weit weg -> bei radius=10 sollte es keine Nachbarn haben
    px, py = sim._particles.x[2], sim._particles.y[2]
    neighbours_x, neighbours_y, interactions, filtered_indices = sim.check_interactions(px, py, sim._particles.radius, 2)

    assert neighbours_x.shape == (0,)
    assert neighbours_y.shape == (0,)
    assert interactions.shape == (0, 2)
    assert filtered_indices.shape == (0,)


def test_check_interactions_finds_neighbours_and_excludes_self():
    sim = make_small_sim()
    # p0 sollte p1 finden
    px, py = sim._particles.x[0], sim._particles.y[0]
    neighbours_x, neighbours_y, interactions, filtered_indices = sim.check_interactions(px, py, sim._particles.radius, 0)

    # Nur p1
    assert filtered_indices.tolist() == [1]
    assert neighbours_x.tolist() == [sim._particles.x[1]]
    assert neighbours_y.tolist() == [sim._particles.y[1]]

    # interactions: [[curr_type, neigh_type]] = [[type(0), type(1)]]
    assert interactions.shape == (1, 2)
    assert interactions[0, 0] == sim._particles.types[0]
    assert interactions[0, 1] == sim._particles.types[1]

    # Selbst darf nicht drin sein:
    assert 0 not in filtered_indices


def test_check_interactions_respects_checked_particles_filter():
    sim = make_small_sim()

    # Markiere Paar (index=0, neighbour=1) als "schon gecheckt"
    # Achtung: check_interactions liest self._checked_particles[index, neighbour]
    sim._checked_particles[0, 1] = True

    px, py = sim._particles.x[0], sim._particles.y[0]
    neighbours_x, neighbours_y, interactions, filtered_indices = sim.check_interactions(px, py, sim._particles.radius, 0)

    # Jetzt sollte p1 rausgefiltert sein
    assert filtered_indices.shape == (0,)
    assert neighbours_x.shape == (0,)
    assert interactions.shape == (0, 2)


def test_calc_velocity_no_neighbours_no_change():
    sim = make_small_sim()
    i = 0

    x0 = sim._particles.x.copy()
    y0 = sim._particles.y.copy()
    vx0 = sim._particles.velocity_x.copy()
    vy0 = sim._particles.velocity_y.copy()

    # N==0
    sim.calc_velocity(
        position_x=float(sim._particles.x[i]),
        position_y=float(sim._particles.y[i]),
        neighbours_x=np.empty(0, dtype=np.float64),
        neighbours_y=np.empty(0, dtype=np.float64),
        interactions=np.empty((0, 2), dtype=np.int64),
        index=i,
        filtered_indices=np.empty(0, dtype=np.int64),
    )

    assert np.all(sim._particles.x == x0)
    assert np.all(sim._particles.y == y0)
    assert np.all(sim._particles.velocity_x == vx0)
    assert np.all(sim._particles.velocity_y == vy0)


def test_calc_velocity_updates_positions_and_marks_checked():
    sim = make_small_sim()

    # Wir testen das Paar i=0 mit neighbour=1
    i = 0
    px, py = float(sim._particles.x[i]), float(sim._particles.y[i])

    neighbours_x = np.array([sim._particles.x[1]], dtype=np.float64)
    neighbours_y = np.array([sim._particles.y[1]], dtype=np.float64)

    interactions = np.array([[sim._particles.types[i], sim._particles.types[1]]], dtype=np.int64)
    filtered_indices = np.array([1], dtype=np.int64)

    x_before = sim._particles.x.copy()
    y_before = sim._particles.y.copy()
    vx_before = sim._particles.velocity_x.copy()
    vy_before = sim._particles.velocity_y.copy()

    sim.calc_velocity(px, py, neighbours_x, neighbours_y, interactions, i, filtered_indices)

    # 1) checked_particles muss gesetzt werden (dein Code setzt [filtered_indices, index])
    assert sim._checked_particles[1, 0]

    # 2) Irgendwas muss sich geändert haben (bei nicht-null Kraft)
    moved = (sim._particles.x != x_before) | (sim._particles.y != y_before)
    vel_changed = (sim._particles.velocity_x != vx_before) | (sim._particles.velocity_y != vy_before)

    assert moved.any() or vel_changed.any()

def test_calc_motion():
    sim = make_small_sim()
    px, py = sim._particles.x[0], sim._particles.y[0]
    neighbours_x, neighbours_y, interactions, filtered_indices = sim.check_interactions(px, py, sim._particles.radius, 0)
    force_direction = sim.calc_motion(simulation.check_interactions)

    assert force_direction.shape == (neighbours_x.shape[0,2])

