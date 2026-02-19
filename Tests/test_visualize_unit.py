import numpy as np
import pytest

from Backend.Particles import Particles
from Backend.Simulation import Simulation

# >>> Passe diesen Import an deinen echten Pfad an:
# z.B. from Frontend.Visualize import Visualize
from Frontend.Visualize import Visualize


# -------------------------
# Kleine Dummy-Objekte fürs Mocking
# -------------------------

class DummyScatter:
    def __init__(self):
        self.last_pos = None
        self.last_face_color = None
        self.last_size = None

    def set_data(self, pos, face_color=None, size=None, **kwargs):
        self.last_pos = np.asarray(pos)
        self.last_face_color = np.asarray(face_color) if face_color is not None else None
        self.last_size = size


class DummyCamera:
    def __init__(self):
        self.range_args = None
        self.interactive = True

    def set_range(self, x=None, y=None):
        self.range_args = {"x": x, "y": y}


class DummyView:
    def __init__(self):
        self.camera = DummyCamera()
        self.scene = object()


class DummyImage:
    def __init__(self, data, parent=None, interpolation=None):
        self.data = np.asarray(data)
        self.parent = parent
        self.interpolation = interpolation


class DummyTimer:
    def __init__(self, interval, connect, start):
        self.interval = interval
        self.connect = connect
        self.start = start


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def particles_and_sim():
    # 2 Partikel, damit update() leicht zu prüfen ist
    x = np.array([49.0, -49.0], dtype=np.float64)
    y = np.array([0.0, 0.0], dtype=np.float64)
    vx = np.zeros(2, dtype=np.float64)
    vy = np.zeros(2, dtype=np.float64)
    types = np.array([0, 1], dtype=np.int64)
    particles = Particles(x=x, y=y, velocity_x=vx, velocity_y=vy, types=types, radius=10)

    W = np.array([
        [ 1, -1,  1, -1,  1],
        [-1,  1, -1,  1, -1],
        [ 1, -1,  1,  1, -1],
        [-1,  1,  1, -1,  1],
        [ 1, -1, -1,  1,  1],
    ], dtype=np.float32)

    sim = Simulation(interactionmatrix=W, particles=particles)
    return particles, sim


def make_visualize_without_gui(monkeypatch, particles, sim):
    """
    Erzeugt ein Visualize-Objekt ohne echte VisPy-GUI:
    Wir umgehen __init__ und setzen nur die Felder, die deine Methoden brauchen.
    """
    viz = Visualize.__new__(Visualize)

    viz._simulation = sim
    viz._particles = particles

    # Weltgröße (wie in deinem Code)
    viz._Lx = 100.0
    viz._Ly = 100.0

    # Palette (5 Typen) wie bei dir
    viz._palette = np.array([
        [1,0,0,1],   # rot
        [0,1,0,1],   # grün
        [0,0,1,1],   # blau
        [1,1,0,1],   # gelb
        [1,0,1,1],   # magenta
    ], dtype=np.float32)

    # Dummy views + scatter
    viz._view_mat = DummyView()
    viz._scatter = DummyScatter()

    # monkeypatch: Image und Timer und app.run ersetzen, damit kein Fenster aufgeht
    import Frontend.Visualize as viz_module  # muss zu deinem Modulnamen passen

    monkeypatch.setattr(viz_module.scene.visuals, "Image", DummyImage, raising=True)
    monkeypatch.setattr(viz_module.app, "Timer", DummyTimer, raising=True)
    monkeypatch.setattr(viz_module.app, "run", lambda: None, raising=True)

    return viz


# -------------------------
# 1) Test: update() wrap + state write-back + scatter.set_data
# -------------------------

def test_update_wraps_and_updates_scatter(monkeypatch, particles_and_sim):
    particles, sim = particles_and_sim
    viz = make_visualize_without_gui(monkeypatch, particles, sim)

    # diffuse() soll absichtlich Werte außerhalb [-50,50) liefern
    def fake_diffuse():
        return np.array([60.0, -60.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(sim, "diffuse", fake_diffuse)

    viz.update(event=None)

    # wrap-around für Lx=100: 60 -> -40, -60 -> 40
    assert np.allclose(viz._particles.x, np.array([-40.0, 40.0]))
    assert np.allclose(viz._particles.y, np.array([0.0, 0.0]))

    # scatter wurde mit den gewrappten Positionen gefüttert
    assert viz._scatter.last_pos.shape == (2, 2)
    assert np.allclose(viz._scatter.last_pos[:, 0], viz._particles.x)
    assert np.allclose(viz._scatter.last_pos[:, 1], viz._particles.y)

    # Farben nach types
    expected_colors = viz._palette[viz._particles.types]
    assert np.allclose(viz._scatter.last_face_color, expected_colors)


# -------------------------
# 2) Test: make_matrix_with_color_header() liefert korrektes RGBA (T+1,T+1,4)
# -------------------------

def test_make_matrix_with_color_header_shape_and_header_colors(monkeypatch, particles_and_sim):
    particles, sim = particles_and_sim
    viz = make_visualize_without_gui(monkeypatch, particles, sim)

    W = np.asarray(sim.interactionmatrix)  # (5,5)
    rgba = viz.make_matrix_with_color_header(W)

    # shape
    assert rgba.shape == (6, 6, 4)
    assert rgba.dtype == np.float32

    # header row/col Farben müssen Palette entsprechen
    T = W.shape[0]
    assert np.allclose(rgba[0, 1:T+1, :], viz._palette[:T])  # obere Headerzeile
    assert np.allclose(rgba[1:T+1, 0, :], viz._palette[:T])  # linke Headerspalte

    # Ecke oben links schwarz
    assert np.allclose(rgba[0, 0, :], np.array([0, 0, 0, 1], dtype=np.float32))


# -------------------------
# 3) Test: start() erstellt Image, setzt Kamera-Range, deaktiviert Interaktivität
# -------------------------

def test_start_creates_image_and_sets_camera(monkeypatch, particles_and_sim):
    particles, sim = particles_and_sim
    viz = make_visualize_without_gui(monkeypatch, particles, sim)

    # start() soll _img setzen und camera konfigurieren, ohne echte loop zu starten
    viz.start()

    assert hasattr(viz, "_img")
    assert isinstance(viz._img, DummyImage)

    # Image-data muss RGBA sein und (6,6,4)
    assert viz._img.data.shape == (6, 6, 4)

    # Kamera-Range gesetzt
    assert viz._view_mat.camera.range_args is not None
    assert viz._view_mat.camera.range_args["x"] == (0, 6)
    assert viz._view_mat.camera.range_args["y"] == (0, 6)

    # Matrix nicht interaktiv
    assert viz._view_mat.camera.interactive is False

