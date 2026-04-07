import numpy as np
import numba
from numba import njit, prange
from numba.core import types
from Backend.Particles import Particles


class Simulation: 
    def __init__(self, 
                 particles: Particles,
                interactionmatrix = np.array([
                    [ 1.0,  -1.2,   1.0,  -0.9,   1.3],
                    [-1.2,  -1.3,  -1.44,  1.15, -0.7],
                    [ 1.0,  -1.44,  1.5,   1.6,  -1.25],
                    [-0.9,   1.15,  1.6,  -1.6,   1.2],
                    [ 1.3,  -0.7,  -1.25,  1.2,   1.5]
                ])                 ):

        self._interactionmatrix: np.ndarray = interactionmatrix 
        self._particles: Particles = particles
        self._friction = 0.15
        
        self._position_x = np.array([
                                    [[0, -1],
                                     [1, 0]],

                                    [[-1, 0],
                                     [0, 0]]
                                ])
        self._position_y = np.array([
                                    [[0, -1],
                                     [1, 0]],

                                    [[-1, 0],
                                     [0, 0]]
                                ])


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

        neigh_types = self._particles.types[indices]            
        curr_type = self._particles.types[index]                

        interactions = np.empty((indices.shape[0], 2), dtype=np.int64)
        interactions[:, 0] = curr_type
        interactions[:, 1] = neigh_types


        return neighbours_x, neighbours_y, interactions, indices

    #wird als erstes immer aufgerufen, musss dann check interactions zwe-, nein ich mach einfach weiter mit selben nachbern kann also entweder als erstes oder als letztes wobei letzeres mehr sinn ergibt damit die nachbern sich whärend der simulation nicht ändern, ein partikelpaar wird sonst sich gegenseitig nicht mehr beeinflussen koennen und nur der erste abgestossene wirkt oder garkeiner im schlimmsten fall
    def calc_collison(self, position_x: float, position_y: float, neighbours_x: np.ndarray, neighbours_y: np.ndarray, index: int, filtered_indices: np.ndarray):
        pass


    def calc_force_direction(self, position_x: float, position_y: float, neighbours_x: np.ndarray, neighbours_y: np.ndarray) -> np.ndarray:
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

        mask_p = np.empty((n,7))
        mask_x = np.empty((n,3))
        mask_y = np.empty((n,3))
        
        mask_p[:,0] = mask_o 
        mask_p[:,1] = mask_u
        mask_p[:,2] = mask_r
        mask_p[:,3] = mask_l
        mask_p[:,4] = mask_m
        mask_p[:,5] = mask_om
        mask_p[:,6] = mask_um
        
        mask_y[:,0] = mask_o 
        mask_y[:,1] = mask_u
        mask_x[:,0] = mask_r
        mask_x[:,1] = mask_l
        mask_y[:,2] = mask_m2
        mask_x[:,2] = mask_m1

        key_x = mask_x.astype(np.int32)
        key_y = mask_y.astype(np.int32)

        gx = self._position_x[key_x[:, 0], key_x[:, 1], key_x[:, 2]]

        gy = self._position_y[key_y[:, 0], key_y[:, 1], key_y[:, 2]] 
        
        force_direction = np.column_stack((gx, gy))
        return force_direction

    def calc_interaction(self, position_x: float, position_y: float, neighbours_x: np.ndarray, neighbours_y: np.ndarray, interactions: np.ndarray, index: int, filtered_indices: np.ndarray) -> None:
        force_direction: np.ndarray = self.calc_force_direction(position_x, position_y, neighbours_x, neighbours_y)
        force: np.ndarray = self._particles.force[filtered_indices]
        radius: int = self._particles.radius
        eps = 1e-6
        kij: np.ndarray = self._interactionmatrix[interactions[:, 0], interactions[:, 1]]  # (N,)

        #force und velocity berechnen, erst force * force direction auf jeweils x und y, dann mit der formel fuer distanz zum betrachteten verechnen, mit reibung verrechnen, dann dass auf die geschwindigkeit addieren und das auf den positionen x und y addieren
        force[:,0] *= force_direction[:,0]
        force[:,1] *= force_direction[:,1]

        particel_distance_x = np.abs(neighbours_x - position_x) + eps
        particel_distance_y = np.abs(neighbours_y - position_y) + eps

        force[:, 0] *= 1 - (particel_distance_x / radius) + eps
        force[:, 1] *= 1 - (particel_distance_y / radius) + eps

        #interactions attribute verrechen i.e. Anziehen und Abstoßen
        force[:, 0] *= kij 
        force[:, 1] *= kij

        self._particles.velocity_x[filtered_indices] += force[:, 0] * 0.99
        self._particles.velocity_y[filtered_indices] += force[:, 1] * 0.99
        self._particles.x[filtered_indices] += self._particles.velocity_x[filtered_indices]
        self._particles.y[filtered_indices] += self._particles.velocity_y[filtered_indices]

        self._particles.direction[filtered_indices] = force_direction




        








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
                
    def diffuse(self):
        #random drift hinzufuegen

        mask_xv = np.abs(self._particles.velocity_x) <= 0.1
        mask_yv = np.abs(self._particles.velocity_y) <= 0.1

        mask_v = mask_xv & mask_yv

        noise_x = np.random.normal(0, 0.01, size=self._particles.x.shape[0])
        noise_y = np.random.normal(0, 0.01, size=self._particles.y.shape[0])

        self._particles.x[mask_v] += noise_x[mask_v]
        self._particles.y[mask_v] += noise_y[mask_v]
        for i in prange(self._particles.x.shape[0]):
            check = self.check_interactions(self._particles.x[i], self._particles.y[i], self._particles.radius, i)
            if check == 0:
                continue
            neighbours_x, neighbours_y, interactions, indices = check
            self.calc_interaction(self._particles.x[i], self._particles.y[i],neighbours_x, neighbours_y, interactions, i, indices)
        return (self._particles.x, self._particles.y)

