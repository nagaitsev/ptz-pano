from __future__ import annotations

import numpy as np

from ptz_pano.stitching.feature_graph_compositor import FeatureEdge, place_by_strongest_tree


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def test_place_by_strongest_tree_composes_affine_edges() -> None:
    edges = [
        FeatureEdge(0, 1, _translation(10, 0), matches=80, inliers=70, mean_error_px=0.4),
        FeatureEdge(1, 2, _translation(0, 20), matches=70, inliers=60, mean_error_px=0.5),
    ]

    transforms = place_by_strongest_tree(frame_count=3, edges=edges)

    assert transforms[0] is not None
    assert transforms[1] is not None
    assert transforms[2] is not None
    np.testing.assert_allclose(transforms[0], _translation(0, 0))
    np.testing.assert_allclose(transforms[1], _translation(10, 0))
    np.testing.assert_allclose(transforms[2], _translation(10, 20))


def test_place_by_strongest_tree_prefers_edges_with_more_inliers() -> None:
    weak_direct = FeatureEdge(
        0,
        2,
        _translation(100, 0),
        matches=100,
        inliers=12,
        mean_error_px=3.0,
    )
    strong_step_a = FeatureEdge(
        0,
        1,
        _translation(10, 0),
        matches=80,
        inliers=60,
        mean_error_px=0.4,
    )
    strong_step_b = FeatureEdge(
        1,
        2,
        _translation(10, 0),
        matches=75,
        inliers=55,
        mean_error_px=0.5,
    )

    transforms = place_by_strongest_tree(
        frame_count=3,
        edges=[weak_direct, strong_step_a, strong_step_b],
    )

    assert transforms[2] is not None
    np.testing.assert_allclose(transforms[2], _translation(20, 0))
