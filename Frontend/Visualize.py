import numpy as np
from vispy import app, scene, color
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

        self._img = None 

    def update(self,event) -> None:
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

    def make_matrix_with_color_header(self, W: np.ndarray):
        T = W.shape[0]
        W = W.astype(np.float32, copy=False)

        cmap = color.get_colormap("RdBu")
        maxabs = float(np.max(np.abs(W))) if W.size else 1.0
        if maxabs == 0:
            maxabs = 1.0

        norm = (W / maxabs) * 0.5 + 0.5
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)

        # ✅ wichtig: map -> (T*T,4) und dann reshape
        rgba_core = cmap.map(norm.ravel()).reshape(T, T, 4).astype(np.float32, copy=False)

        rgba_img = np.zeros((T + 1, T + 1, 4), dtype=np.float32)
        rgba_img[1:, 1:, :] = rgba_core
        rgba_img[0, 1:, :] = self._palette[:T]
        rgba_img[1:, 0, :] = self._palette[:T]
        rgba_img[0, 0, :] = np.array([0, 0, 0, 1], dtype=np.float32)
        print(type(rgba_img))
        return rgba_img


    def start(self) -> None:
        W = np.asarray(self._simulation.interactionmatrix)   # (5,5)
        rgba_img = self.make_matrix_with_color_header(W)

        self._img = scene.visuals.Image(
            rgba_img,
            parent=self._view_mat.scene,
            interpolation="nearest"
        )
        h, w = rgba_img.shape[:2]
        self._view_mat.camera.set_range(x=(0, w), y=(0, h))
        self._view_mat.camera.interactive = False

        timer = app.Timer(interval=0.016, connect=self.update, start=True)
        app.run()

