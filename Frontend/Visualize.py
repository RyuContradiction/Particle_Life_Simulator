import numpy as np
from vispy import app, scene, color


class Visualize:
    def __init__(self, simulation, particles, domain=((0.0, 1.0), (0.0, 1.0)), interval=0.05):
        self._simulation = simulation
        self._particles = particles

        (self._xmin, self._xmax), (self._ymin, self._ymax) = domain
        self._Lx = float(self._xmax - self._xmin)
        self._Ly = float(self._ymax - self._ymin)
        if self._Lx <= 0 or self._Ly <= 0:
            raise ValueError("domain muss xmin<xmax und ymin<ymax haben.")

        # Canvas + Layout: Simulation groß, Matrix klein
        self._canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
        self._canvas.size = (1200, 700)

        grid = self._canvas.central_widget.add_grid(margin=10)
        grid.spacing = 0

        self._view_particles = grid.add_view(row=0, col=0, col_span=5)
        self._view_matrix = grid.add_view(row=0, col=5)

        self._view_particles.camera = "panzoom"
        self._view_particles.camera.aspect = 1
        self._view_particles.camera.set_range(x=(self._xmin, self._xmax), y=(self._ymin, self._ymax))

        self._view_matrix.camera = "panzoom"
        self._view_matrix.camera.aspect = 1

        # Partikel visual
        self._scatter = scene.visuals.Markers(parent=self._view_particles.scene)

        pos0 = np.c_[np.asarray(particles.x), np.asarray(particles.y)]
        pos0 = self._wrap_periodic(pos0)

        # Farben pro Typ (robust: explizites RGBA pro Punkt)
        face0 = self._colors_for_types(np.asarray(getattr(particles, "types", None)))
        self._scatter.set_data(
            pos0.astype(np.float32, copy=False),
            face_color=face0,
            edge_color=None,
            size=float(getattr(particles, "radius", 6)),
        )

        # Interaktionsmatrix visual
        W0 = self._get_interaction_matrix_for_display()
        self._img = scene.visuals.Image(
            W0,
            parent=self._view_matrix.scene,
            cmap="RdBu",               # diverging: gut für -/+
            interpolation="nearest",
            clim=self._auto_clim(W0),  # sinnvolle Skalierung
        )
        self._fit_matrix_camera(W0)

        self._timer = app.Timer(interval=interval, connect=self.update, start=True)

    # ---------- helpers ----------

    def _wrap_periodic(self, pos):
        pos = np.asarray(pos, dtype=float)
        pos[:, 0] = ((pos[:, 0] - self._xmin) % self._Lx) + self._xmin
        pos[:, 1] = ((pos[:, 1] - self._ymin) % self._Ly) + self._ymin
        return pos

    def _colors_for_types(self, types):
        palette = np.array([
            [1,0,0,1],   # rot
            [0,1,0,1],   # grün
            [0,0,1,1],   # blau
            [1,1,0,1],   # gelb
            [1,0,1,1],   # magenta
        ], dtype=np.float32)

        types = np.asarray(types) % len(palette)
        return palette[types]

    def _get_interaction_matrix_for_display(self):
        """
        Deine Simulation.interactionmatrix enthält offenbar Labels in erster Zeile/Spalte:
        [[0,1,2,3,4],
         [1, 1,-1,-1, 1],
         ...]
        -> Für Heatmap wollen wir nur den Kern W[1:,1:].
        """
        W = np.asarray(self._simulation.interactionmatrix)
        if W.ndim != 2:
            raise ValueError("simulation.interactionmatrix muss 2D sein.")

        # Heuristik: erste Zeile == 0..n-1 UND erste Spalte == 0..n-1 (Labels)
        # Dann schneiden wir sie weg.
        if W.shape[0] == W.shape[1] and W.shape[0] >= 3:
            n = W.shape[0]
            row0 = W[0, :]
            col0 = W[:, 0]
            if np.array_equal(row0, np.arange(n)) and np.array_equal(col0, np.arange(n)):
                W = W[1:, 1:]

        return W.astype(np.float32, copy=False)

    def _auto_clim(self, W):
        """
        Für Interaktionsmatrizen ist oft -1..1 sinnvoll.
        Falls andere Werte drin sind: symmetrisch um 0 skalieren.
        """
        W = np.asarray(W)
        maxabs = float(np.max(np.abs(W))) if W.size else 1.0
        if maxabs == 0:
            maxabs = 1.0
        return (-maxabs, maxabs)

    def _fit_matrix_camera(self, W):
        h, w = W.shape
        self._view_matrix.camera.set_range(x=(0, w), y=(0, h))

    # ---------- update loop ----------

    def update(self, event):
        # Sim step
        x, y = self._simulation.diffuse()
        pos = np.c_[np.asarray(x), np.asarray(y)]
        pos = self._wrap_periodic(pos)

        # Farben ggf. live (falls types sich ändern könnten)
        face = self._colors_for_types(np.asarray(getattr(self._particles, "types", None)))

        self._scatter.set_data(
            pos.astype(np.float32, copy=False),
            face_color=face,
            edge_color=None,
            size=float(getattr(self._particles, "radius", 6)),
        )

        # Matrix refresh + clim passend halten
        W = self._get_interaction_matrix_for_display()
        self._img.set_data(W)
        self._img.clim = self._auto_clim(W)
        # Wenn die Matrixgröße sich ändern kann:
        # self._fit_matrix_camera(W)

    def start(self):
        app.run()

