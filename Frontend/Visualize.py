import numpy as np
from vispy import app, scene
from Backend.Simulation import Simulation
from Backend.Particles import Particles


class Visualize:

    def __init__(self, x: np.ndarray, y: np.ndarray, particles: Particles, simulation: Simulation):
        self._x: np.ndarray = x
        self._y: np.ndarray = y
        self._particles: Particles = particles
        self._simulation: Simulation = simulation
        app.use_app("pyqt5")  # <-- VOR canvas
        self._canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
        self._grid = self._canvas.central_widget.add_grid(margin=10)
        self._grid.spacing = 10

        self._view_sim = self._grid.add_view(row=0, col=0, col_span=4)
        self._view_mat = self._grid.add_view(row=0, col=4)

        self._view_sim.camera = scene.cameras.PanZoomCamera(aspect=1)
        self._view_mat.camera = scene.cameras.PanZoomCamera(aspect=1)
        self._view_mat.camera.flip = (0, 1, 0)   # y flip
        self._palette = np.array([
            [1,0,0,1],   # rot
            [0,1,0,1],   # grün
            [0,0,1,1],   # blau
            [1,1,0,1],   # gelb
            [1,0,1,1],   # magenta
        ], dtype=float)

        self._colors = self._palette[particles.types]
        # --- Scatter (Simulation) ---
        self._scatter = scene.visuals.Markers(parent=self._view_sim.scene)
        self._pos0 = np.c_[x, y]
        self._scatter.set_data(self._pos0, face_color=self._colors, size=10)

        self._Lx, self._Ly = 200, 200
        self._view_sim.camera.set_range(x=(-self._Lx/2, self._Lx/2), y=(-self._Ly/2, self._Ly/2))

        # --- Matrix (statisch) ---
        self._W = np.asarray(simulation.interactionmatrix)
        self._img = scene.visuals.Image(self._W, parent=self._view_mat.scene, cmap="RdBu",
                          interpolation="nearest", clim=(-1, 1))

        self._h, self._w = self._W.shape
        self._view_mat.camera.set_range(x=(0, self._w), y=(0, self._h))

        # optional: Matrix nicht zoombar/pannbar
        self._view_mat.camera.interactive = False

    def update(self,event):
        x_new, y_new = self._simulation.diffuse()

        # wrap-around
        x_wrapped = ((x_new + self._Lx/2) % self._Lx) - self._Lx/2
        y_wrapped = ((y_new + self._Ly/2) % self._Ly) - self._Ly/2

        # ✅ WICHTIG: Zustand zurück in particles schreiben
        self._particles.x[:] = x_wrapped
        self._particles.y[:] = y_wrapped

        pos = np.c_[self._particles.x, self._particles.y]
        colors_now = self._palette[self._particles.types]

        self._scatter.set_data(pos, face_color=colors_now, size=10)

    def start(self) -> None:
        timer = app.Timer(interval=0.016, connect=self.update, start=True)
        app.run()

