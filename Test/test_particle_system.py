from Backend.particle_system import generate_particle

def test_generate_particle():
    positions, velocities, types = generate_particle()
    
    assert positions.shape[0] == 100, "Should generate 100 particles"
    assert positions.shape[1] == 2, "Each position should have 2 coordinates (x, y)"
    
    assert velocities.shape[0] == 100, "Should generate 100 particles"
    assert velocities.shape[1] == 2, "Each velocity should have 2 components (vx, vy)"
    
    assert types.shape[0] == 100, "Should generate 100 particles"
    assert all(t in [0, 1, 2, 3] for t in types), "Particle types should be between 0 and 3"