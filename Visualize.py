import numpy as np
from vispy import app, scene
from Backend.Simulation import Simulation
from Backend.Particles import Particles

app.use_app("pyqt5")  # <-- VOR canvas

x = np.random.normal(loc=0.0, scale=10.0, size=10000)
y = np.random.normal(loc=0.0, scale=10.0, size=10000)

particles = Particles(
    x, y,
    velocity_x=np.zeros(10000),
    velocity_y=np.zeros(10000),
    types=np.random.randint(0, 5, size=10000),
    radius=20
)



simulation = Simulation(particles=particles)

canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="black")
grid = canvas.central_widget.add_grid(margin=10)
grid.spacing = 10

view_sim = grid.add_view(row=0, col=0, col_span=4)
view_mat = grid.add_view(row=0, col=4)

view_sim.camera = scene.cameras.PanZoomCamera(aspect=1)
view_mat.camera = scene.cameras.PanZoomCamera(aspect=1)
view_mat.camera.flip = (0, 1, 0)   # y flip
palette = np.array([
    [1,0,0,1],   # rot
    [0,1,0,1],   # grün
    [0,0,1,1],   # blau
    [1,1,0,1],   # gelb
    [1,0,1,1],   # magenta
], dtype=float)

colors = palette[particles.types]
# --- Scatter (Simulation) ---
scatter = scene.visuals.Markers(parent=view_sim.scene)
pos0 = np.c_[x, y]
scatter.set_data(pos0, face_color=colors, size=10)

Lx, Ly = 100, 100
view_sim.camera.set_range(x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2))

# --- Matrix (statisch) ---
W = np.asarray(simulation.interactionmatrix)
print(W)
# falls Header-Zeile/Spalte vorhanden: entfernen
if W.ndim == 2 and W.shape[0] == W.shape[1]:
    n = W.shape[0]
    if np.array_equal(W[0, :], np.arange(n)) and np.array_equal(W[:, 0], np.arange(n)):
        W = W[1:, 1:]
print(W)
img = scene.visuals.Image(W, parent=view_mat.scene, cmap="RdBu",
                          interpolation="nearest", clim=(-1, 1))

h, w = W.shape
view_mat.camera.set_range(x=(0, w), y=(0, h))

# optional: Matrix nicht zoombar/pannbar
view_mat.camera.interactive = False

def update(event):
    x_new, y_new = simulation.diffuse()

    # wrap-around
    x_wrapped = ((x_new + Lx/2) % Lx) - Lx/2
    y_wrapped = ((y_new + Ly/2) % Ly) - Ly/2

    pos = np.c_[x_wrapped, y_wrapped]
    N = pos.shape[0]
    colors_now = palette[particles.types[:N]]

    scatter.set_data(pos, face_color=colors_now, size=10)
    canvas.update()

timer = app.Timer(interval=0.016, connect=update, start=True)
app.run()
