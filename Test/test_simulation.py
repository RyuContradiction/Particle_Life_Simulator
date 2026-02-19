from Backend.Simulation import Simulation
import numpy as np
INTERACTION_RADIUS, INTERACTION_MATRIX = 100, np.array([[1, -1, 1, -1, 1],
                                                           [-1, -1, -1, 1, -1],
                                                           [1, -1, 1, 1, -1],
                                                           [-1, 1, 1, -1, 1],
                                                           [1, -1, -1, 1, 1]], dtype=float)


def test_check_interactions():
    sim = Simulation()
    sim._particles.x = np.array([0.0, 1.0, 2.0, 3.0])
    sim._particles.y = np.array([0.0, 1.0, 2.0, 3.0])
    sim._particles.types = np.array([0, 1, 2, 3])
    
    result = sim.check_interactions(0.0, 0.0, radius=1.5, index=0)
    assert result is not None
    neighbours_x, neighbours_y, interactions, indices = result
    assert len(neighbours_x) == 1
    assert len(neighbours_y) == 1
    assert len(interactions) == 1
    assert len(indices) == 1
    assert neighbours_x[0] == 1.0
    assert neighbours_y[0] == 1.0
    # interactions is (N, 2) array with [type_i, type_j]
    expected_interaction_value = sim._interactionmatrix[interactions[0, 0], interactions[0, 1]]
    assert expected_interaction_value == sim._interactionmatrix[0, 1]


def test_calc_velocity():
    sim = Simulation()
    sim._particles.x = np.array([0.0, 1.0])
    sim._particles.y = np.array([0.0, 1.0])
    sim._particles.types = np.array([0, 1])
    sim._particles.velocity_x = np.zeros(2)
    sim._particles.velocity_y = np.zeros(2)
    
    neighbours_x = np.array([1.0])
    neighbours_y = np.array([1.0])
    # interactions should be (N, 2) with [type_i, type_j] pairs
    interactions = np.array([[0, 1]])  # particle type 0 interacting with particle type 1
    
    sim.calc_velocity(0.0, 0.0, neighbours_x, neighbours_y, interactions, index=0, filtered_indices=np.array([1]))
    
    assert True


def run_performance_test():
    import timeit

    print("\n" + "-" * 60)
    print("LEISTUNGSTEST")
    print("-" * 60)

    sim = Simulation()
    time_step = timeit.timeit(lambda: sim.step(), number=10)
    print(f"\nstep() x10:                 {time_step:.4f}s  ({time_step/10*1000:.2f}ms pro step)")
    px, py = sim._particles.x[0], sim._particles.y[0]
    time_check = timeit.timeit(
        lambda: sim.check_interactions(px, py, INTERACTION_RADIUS, 0),
        number=1000
    )
    print(f"check_interactions() x1000: {time_check:.4f}s  ({time_check/1000*1000:.3f}ms pro Aufruf)")
    time_force = timeit.timeit(lambda: sim.calc_force(0), number=1000)
    print(f"calc_force() x1000:         {time_force:.4f}s ({time_force/1000*1000:.3f}ms pro Aufruf)")

    time_friction = timeit.timeit(
        lambda: sim.calc_friction(sim._particles.velocity_x, sim._particles.velocity_y),
        number=10000
    )
    print(f"calc_friction() x10000:     {time_friction:.4f}s  ({time_friction/10000*1000:.4f}ms pro Aufruf)")


def test_diffuse():
    sim = Simulation()
    sim._particles.x = np.array([0.0, 1.0])
    sim._particles.y = np.array([0.0, 1.0])
    sim._particles.types = np.array([0, 1])
    sim._particles.velocity_x = np.zeros(2)
    sim._particles.velocity_y = np.zeros(2)
    
    new_x, new_y = sim.diffuse()
    
    assert new_x is not None
    assert new_y is not None