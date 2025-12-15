import numpy as np
from Config.config import FRICTION

class Environment:

    def __init__(Self):
        pass


    def calc_friction(self, velocity_x, velocity_y) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate friction forces for all particles.
        Returns:
            Tuple of numpy arrays representing friction forces in x and y directions.

        friction_x = -FRICTION (coefficient) * velocity_x
        """

        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

        friction_x = FRICTION * self.velocity_x
        friction_y = FRICTION * self.velocity_y

        return friction_x, friction_y


    def calc_force(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Berechnet die resultierenden Kräfte (Fx_total, Fy_total) auf alle Partikel.
        Dies beinhaltet Interaktionskräfte und externe Kräfte (z.B. Reibung).

        :return: Ein Tupel (Fx_total, Fy_total) der Gesamtkräfte.
        """
        pass