import numpy as np
from Backend.Particles import Particles


class Simulation: 
    def __init__(self, 
                 interactionmatrix: np.ndarray = np.array([[1, -1, 1, -1, 1],
                                                           [-1, -1, -1, 1, -1],
                                                           [1, -1, 1, 1, -1],
                                                           [-1, 1, 1, -1, 1],
                                                           [1, -1, -1, 1, 1]]),
                 particles: Particles = Particles()):

        self._interactionmatrix: np.ndarray = interactionmatrix 
        self._particles: Particles = particles
        self._checked_particles: np.ndarray = np.zeros((self._particles.x.shape[0], self._particles.x.shape[0]), dtype= bool)

    @property
    def interactionmatrix(self) -> np.ndarray:
        return self._interactionmatrix

    def check_interactions(self, position_x, position_y, radius, index) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # AABB um den Punkt (symmetrisch)
        maske_x = (self._particles.x >= position_x - radius) & (self._particles.x <= position_x + radius)
        maske_y = (self._particles.y >= position_y - radius) & (self._particles.y <= position_y + radius)
        maske_n = maske_x & maske_y

        # sich selbst ausschließen
        maske_n[index] = False

        # Original-Indizes der Nachbarn
        indices = np.where(maske_n)[0]
        if indices.size == 0:
            # Immer gleiches Rückgabeformat (leer)
            empty = np.empty(0, dtype=self._particles.x.dtype)
            empty_i = np.empty((0, 2), dtype=np.int64)
            return empty, empty, empty_i, indices

        neighbours_x = self._particles.x[indices]
        neighbours_y = self._particles.y[indices]

        neighbours = self._checked_particles[index]
        checked_neighbours = neighbours[indices]
        filter_n = checked_neighbours == False
        neighbours_x = neighbours_x[filter_n]
        neighbours_y = neighbours_y[filter_n]



        # Typen der Nachbarn
        neigh_types = self._particles.types[indices]            # (N,)
        neigh_types = neigh_types[filter_n]
        curr_type = self._particles.types[index]                # Skalar

        # interactions als (N,2): [ [curr_type, neigh_type], ... ]
        interactions = np.empty((np.sum(filter_n), 2), dtype=np.int64)
        interactions[:, 0] = curr_type
        interactions[:, 1] = neigh_types

        filtered_indices = indices[filter_n]
        

        return neighbours_x, neighbours_y, interactions, filtered_indices

    def calc_velocity(
        self,
        position_x: float,
        position_y: float,
        neighbours_x: np.ndarray,     # (N,)
        neighbours_y: np.ndarray,     # (N,)
        interactions: np.ndarray,     # (N,2) -> [current_type, neighbour_type]
        index: int,
        filtered_indices: np.ndarray
        ) -> None:
        # Konstanten (bei dir ggf. als Attribute speichern)
        k: float = 1.0
        m1: float = 1.0
        m2: float = 1.0
        t: float = 0.01
        gamma: float = 0.001
        eps: float = 1e-12

        N:int = neighbours_x.shape[0]
        if N == 0:
            return

        


        # --- Geometrie (alles vektorisiert) ---
        # Vektoren vom aktuellen Partikel zu allen Nachbarn
        dx: np.ndarray = position_x - neighbours_x                # (N,)
        dy: np.ndarray = position_y - neighbours_y                # (N,)
        r2: np.ndarray = dx * dx + dy * dy + eps                  # (N,)   (Abstand^2)
        r_abs: np.ndarray = np.sqrt(r2)                           # (N,)

        # Einheitsrichtungen (N,2)
        r_hat: np.ndarray = np.column_stack((dx / r_abs, dy / r_abs))  # (N,2)

        # --- k_ij holen (vektorisiert) ---
        kij: np.ndarray = self._interactionmatrix[interactions[:, 0], interactions[:, 1]]  # (N,)

        # --- Kräfte pro Nachbar (N,2) ---
        # inverse-square: 1/r^2 (hier r2 ist schon Abstand^2)
        F_pairs: np.ndarray = (k * kij / r2)[:, None] * r_hat     # (N,2)

        # Gesamtkraft auf das aktuelle Partikel (2,)
        F1: np.ndarray = F_pairs.sum(axis=0)

        # Reibung auf aktuelles Partikel
        v1: np.ndarray = np.array([self._particles.velocity_x[index], self._particles.velocity_y[index]], dtype=np.float64)
        F1 = F1 - gamma * v1

        # Beschleunigung + Update für aktuelles Partikel
        a1: np.ndarray = F1 / m1
        self._particles.velocity_x[index] += a1[0] * t
        self._particles.velocity_y[index] += a1[1] * t
        self._particles.x[index] += self._particles.velocity_x[index] * t
        self._particles.y[index] += self._particles.velocity_y[index] * t

        # --- Gegenkräfte auf Nachbarn (ohne Loop) ---
        # Newton III: Nachbar bekommt -F_pair
        F2_pairs: np.ndarray = -F_pairs                            # (N,2)

        # Reibung auf Nachbarn (N,2)
        v2: np.ndarray = np.column_stack((
            self._particles.velocity_x[filtered_indices],
            self._particles.velocity_y[filtered_indices],
        )).astype(np.float64)                           # (N,2)
        F2_pairs = F2_pairs - gamma * v2

        # Beschleunigung + Update für Nachbarn (vektorisiert)
        a2: np.ndarray = F2_pairs / m2                              # (N,2)
        self._particles.velocity_x[filtered_indices] += a2[:, 0] * t
        self._particles.velocity_y[filtered_indices] += a2[:, 1] * t
        self._particles.x[filtered_indices] += self._particles.velocity_x[filtered_indices] * t
        self._particles.y[filtered_indices] += self._particles.velocity_y[filtered_indices] * t

        #Um doppelt berechnungen der Geschwindigkeit eines Partikel im zusammenhang eines anderen zu vermeiden, werden diese in einem Array vermerkt
        index_array: np.ndarray = np.full(filtered_indices.shape, index)
        self._checked_particles[filtered_indices, index_array] = True
                
    def diffuse(self):
        for i in range(self._particles.x.shape[0]):
            check = self.check_interactions(self._particles.x[i], self._particles.y[i], self._particles.radius, i)
            if check == 0:
                continue
            neighbours_x, neighbours_y, interactions, indices = check
            self.calc_velocity(self._particles.x[i], self._particles.y[i],neighbours_x, neighbours_y, interactions, i, indices)
        self._checked_particles.fill(False)
        return (self._particles.x, self._particles.y)

