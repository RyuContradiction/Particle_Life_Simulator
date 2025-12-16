import numpy as np
from Backend.config import NUM_PARTICLES, NUM_TYPES, WIDTH, HEIGHT


class ParticleSystem:
    """
    Verwaltet alle Partikel als NumPy Arrays.
    
    """
    
    def __init__(self, num_particles=NUM_PARTICLES, num_types=NUM_TYPES):
        """
        Erstellt ein neues Partikelsystem.
        
        Args:
            num_particles: Anzahl der Partikel (default aus config.py)
            num_types: Anzahl der Typen, z.B. 4 für Rot/Grün/Blau/Gelb
        """
        # Position: Zufällig im Fenster verteilt (0 bis WIDTH/HEIGHT)
        self.x = np.random.rand(num_particles) * WIDTH
        self.y = np.random.rand(num_particles) * HEIGHT
        
        # Geschwindigkeit: Startet bei 0 für alle Partikel
        self.velocity_x = np.zeros(num_particles)
        self.velocity_y = np.zeros(num_particles)
        
        # Typ: Zufällig zwischen 0 und num_types-1 (bestimmt Farbe & Verhalten)
        self.types = np.random.randint(0, num_types, num_particles)