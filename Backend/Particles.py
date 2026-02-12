import numpy as np

class Particles:
    def __init__(self, 
                 x: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size=1000),
                 y: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size=1000), 
                 velocity_x: np.ndarray = np.zeros(1000), 
                 velocity_y: np.ndarray = np.zeros(1000),  
                 types: np.ndarray = np.clip(np.rint(np.random.normal(loc=0, scale=1.0, size=1000)), 0, 4).astype(int),
                 radius: int = 15):
            # keine Parameter
        self._x = x
        self._y = y
        self._velocity_x = velocity_x
        self._velocity_y = velocity_y
        self._types = types 
        self._radius = radius
        
            # mit property
    @property
    def x(self) -> np.ndarray: 
                return self._x
    @x.setter
    def x(self, value: np.ndarray) -> None: 
          self._x = value
        
    @property
    def y(self) -> np.ndarray: 
        return self._y
    @y.setter
    def y(self, value: np.ndarray) -> None: 
        self._y = value
    
    @property
    def velocity_x(self) -> np.ndarray: 
        return self._velocity_x
    @velocity_x.setter
    def velocity_x(self, value: np.ndarray) -> None: 
        self._velocity_x = value
        
    @property
    def velocity_y(self) -> np.ndarray: 
        return self._velocity_y
    @velocity_y.setter
    def velocity_y(self, value: np.ndarray) -> None: 
        self._velocity_y = value
        
    @property
    def types(self) -> np.ndarray: 
        return self._types
    @types.setter
    def types(self, value: np.ndarray) -> None: 
        self._types = value

    @property
    def radius(self) -> int:
        return self._radius
    @radius.setter
    def radius(self, value: int) -> None:
        self._radius = value
