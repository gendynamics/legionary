import numpy as np
import pytest
from quantum_optimizer import gl3d, viewer


def test_diamond_geometry():
    v, e = gl3d.shape_edges("diamond")
    assert v.shape == (6, 3) and len(e) == 12 and len(set(map(frozenset, e))) == 12
    deg = np.bincount(np.array(e).ravel(), minlength=6)
    assert list(deg) == [4] * 6  # octahedron: every vertex has 4 edges
    assert v[:, 1].max() > np.abs(v[:, 0]).max()  # vertically elongated


def test_cube_geometry_and_same_mesh_size():
    v, e = gl3d.shape_edges("cube")
    assert v.shape == (8, 3) and len(e) == 12
    assert gl3d.edge_mesh("diamond").shape == gl3d.edge_mesh("cube").shape == (72, 8)
    with pytest.raises(ValueError):
        gl3d.shape_edges("sphere")


def test_shape_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(gl3d, "run", lambda a, d: seen.setdefault("shape", a.shape) and 0)
    for argv, want in ([], "diamond"), (["--shape", "cube"], "cube"):
        seen.clear()
        viewer.main(["--demo", "--director-backend", "off", *argv])
        assert seen["shape"] == want
    with pytest.raises(SystemExit):
        viewer.main(["--demo", "--shape", "pyramid"])
