import numpy as np
import numba
from numba import njit, prange
from numba.core import types
from Backend.Particles import Particles


class Simulation: 
    def __init__(self, 
                 particles: Particles,
                 interactionmatrix: np.ndarray = np.array([[1, -1, 1, -1, 1],
                                                           [-1, -1, -1, 1, -1],
                                                           [1, -1, 1, 1, -1],
                                                           [-1, 1, 1, -1, 1],
                                                           [1, -1, -1, 1, 1]])
                 ):

        self._interactionmatrix: np.ndarray = interactionmatrix 
        self._particles: Particles = particles
        self._checked_particles: np.ndarray = np.zeros((self._particles.x.shape[0], self._particles.x.shape[0]), dtype= bool)
        self._friction = 0.15
        self._position = np.array([0, -1, 1, -1, 1, -1, -1, -1])

    @property
    def interactionmatrix(self) -> np.ndarray:
        return self._interactionmatrix
    


    # Kein checked weil Kraft immer nur einmal berechnet wird und nicht direkt das Paar, und Orange benutzten
    #Force ist ein Vektor 
    #Maske_p besteht aus o u l r m1 m2
    #Dict mi
    #Irgendwie das ganze abhängig machen von der Distanz 
    #Der Radius bekommt Schichten oder wir pro 1 oder 0.1 Näherung wird Kraft verstärkt um * 1,5
    #Wenn Partikel fast auf anderes Partikel, dann können diese sich auch abstoßen obwohl sie anziehen wenn genug Geschwindigkeit 
    #Das wird als erstes überprüft 
    #Und das abgestoßen Partikel bekommt 90% der Geschwindigkeit 
    #Bei bestimmter Geschwindigkeit Random drift




    def check_interactions(self, position_x, position_y, radius, index) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        maske_x = (self._particles.x >= position_x - radius) & (self._particles.x <= position_x + radius)
        maske_y = (self._particles.y >= position_y - radius) & (self._particles.y <= position_y + radius)
        maske_n = maske_x & maske_y

        maske_n[index] = False

        indices = np.where(maske_n)[0]
        if indices.size == 0:

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



        neigh_types = self._particles.types[indices]            
        neigh_types = neigh_types[filter_n]
        curr_type = self._particles.types[index]                

        interactions = np.empty((np.sum(filter_n), 2), dtype=np.int64)
        interactions[:, 0] = curr_type
        interactions[:, 1] = neigh_types

        filtered_indices = indices[filter_n]
        

        return neighbours_x, neighbours_y, interactions, filtered_indices

    #wird als erstes immer aufgerufen, musss dann check interactions zwe-, nein ich mach einfach weiter mit selben nachbern kann also entweder als erstes oder als letztes wobei letzeres mehr sinn ergibt damit die nachbern sich whärend der simulation nicht ändern, ein partikelpaar wird sonst sich gegenseitig nicht mehr beeinflussen koennen und nur der erste abgestossene wirkt oder garkeiner im schlimmsten fall
    def calc_collison(self, position_x: float, position_y: float, neighbours_x: np.ndarray, neighbours_y: np.ndarray, index: int, filtered_indices: np.ndarray):
        pass


    def calc_force_direction(self, position_x: float, position_y: float, neighbours_x: np.ndarray, neighbours_y: np.ndarray, index: int, filtered_indices: np.ndarray) -> np.ndarray:
        size: int = self._particles.size
        force_direction: np.ndarray

        
        #sieben mögliche Positionen wo sich die anderen Partikel befinden: oben = o, unten = u, links = l, rechts = r, mitte = m -> or, ol, mr, ml, ur, ul und mm welcher der bereich von der grösse des Partikels ist

        mask_o = neighbours_y >= position_y + size
        mask_u = neighbours_y <= position_y - size
        mask_m1 = (neighbours_x <= position_x + size) & (neighbours_x >= position_x - size)
        

        mask_r = neighbours_x >= position_x + size
        mask_l = neighbours_x <= position_x - size
        mask_m2 = (neighbours_y <= position_y + size) & (neighbours_y >= position_y - size)
        
        mask_m = mask_m1 & mask_m2
        mask_om = mask_m1 & mask_o
        mask_um = mask_m1 & mask_u
        n = mask_o.shape[0]

        mask_p = np.empty(n,9)
        
        mask_p[:,0] = mask_o 
        mask_p[:,1] = mask_u
        mask_p[:,2] = mask_r
        mask_p[:,3] = mask_l
        mask_p[:,4] = mask_m
        mask_p[:,5] = mask_om
        mask_p[:,6] = mask_um
        
        keys = mask_p.astype(int32)
        
        keys[:,0] *= 1
        keys[:,1] *= 2
        keys[:,2] *= 3
        keys[:,3] *= 4
        keys[:,4] *= 5
        keys[:,5] *= 6
        keys[:,6] *= 7

        positions = np.sum(keys,axis=1)

        force_direction = self._position[positions]
        return force_direction

    def calc_motion(self, position_x: float, position_y: float, neighbours_x: np.ndarray, neighbours_y: np.ndarray, index: int, filtered_indices: np.ndarray) -> None:
        force_direction: np.ndarray = self.calc_force_direction(position_x, position_y, neighbours_x, neighbours_y)
        force: np.ndarray = self._particles.force[filtered_indices]
        radius: int = self._particles.radius
        eps = 1e-6

        #force und velocity berechnen, erst force * force direction auf jeweils x und y, dann mit der formel fuer distanz zum betrachteten verechnen, mit reibung verrechnen, dann dass auf die geschwindigkeit addieren und das auf den positionen x und y addieren
        force[:,0] *= force_direction
        force[:,1] *= force_direction

        particel_distance_x = np.abs(neighbours_x - position_x) + eps
        particel_distance_y = np.abs(neighbours_y - position_y) + eps

        force[:, 0] *= 1 - (particel_distance_x / radius) + eps
        force[:, 1] *= 1 - (particel_distance_y / radius) + eps


        self._particles.velocity_x[filtered_indices] += force[:, 0] 
        self._particles.velocity_y[filtered_indices] += force[:, 1]

        #Reibung
        self._particles.velocity_x[filtered_indices] *= 0.9
        self._particles.velocity_y[filtered_indices] *= 0.9


        self._particles.x[filtered_indices] += self._particles.velocity_x[filtered_indices]
        self._particles.y[filtered_indices] += self._particles.velocity_y[filtered_indices]


        








    # Kein checked weil Kraft immer nur einmal berechnet wird und nicht direkt das Paar, und Orange benutzten
    #Force ist ein Vektor 
    #Maske_p besteht aus o u l r m1 m2
    #Dict mi
    #Irgendwie das ganze abhängig machen von der Distanz 
    #Der Radius bekommt Schichten oder wir pro 1 oder 0.1 Näherung wird Kraft verstärkt um * 1,5
    #Wenn Partikel fast auf anderes Partikel, dann können diese sich auch abstoßen obwohl sie anziehen wenn genug Geschwindigkeit 
    #Das wird als erstes überprüft 
    #Und das abgestoßen Partikel bekommt 90% der Geschwindigkeit 
    #Bei bestimmter Geschwindigkeit Random drift



        

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
        #random drift hinzufuegen
        for i in prange(self._particles.x.shape[0]):
            check = self.check_interactions(self._particles.x[i], self._particles.y[i], self._particles.radius, i)
            if check == 0:
                continue
            neighbours_x, neighbours_y, interactions, indices = check
            self.calc_velocity(self._particles.x[i], self._particles.y[i],neighbours_x, neighbours_y, interactions, i, indices)
        self._checked_particles.fill(False)
        return (self._particles.x, self._particles.y)

