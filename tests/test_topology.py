"""Pin msl.data.landmarks to MediaPipe's own topology tables.

`landmarks.py` hand-copies MediaPipe's landmark indices so that it stays
importable with no MediaPipe installed -- config.py imports it, and Milestone 0
runs on a machine with neither the library nor a model bundle. A hand-copied
table drifts silently, so this module checks the copy against the installed
package's declared topology and is skipped when that package is absent.

This is not a style test. It found a real defect: LEFT_EYEBROW_INDICES and
RIGHT_EYEBROW_INDICES were inverted. Both tuples held genuine eyebrow points in
correct chain order, so every downstream consumer looked healthy -- the overlay
would have drawn two plausible eyebrows, the feature vector would have had the
right shape, and training would have converged. The only symptom was that a
constant named "left" addressed the signer's right.

What this does NOT establish: that the bundled facemesh2_lite tflite emits its
478 points in the order these tables assume. That link is closed only by
rendering the overlay on a real face (spec S10.1).
"""

from __future__ import annotations

import pytest

from msl.data import landmarks as lm

mp_vision = pytest.importorskip(
    "mediapipe.tasks.python.vision",
    reason="MediaPipe not installed; topology constants cannot be cross-checked",
)


def _vertices(connections) -> set[int]:
    """Vertex set of a MediaPipe connection table."""
    points: set[int] = set()
    for connection in connections:
        points.add(connection.start)
        points.add(connection.end)
    return points


def _edges(connections) -> set[frozenset[int]]:
    """Undirected edge set, so traversal direction is not under test."""
    return {frozenset((c.start, c.end)) for c in connections}


# --- Face ------------------------------------------------------------------


def test_lips_subset_of_mediapipe_lips():
    table = _vertices(mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
    assert set(lm.LIPS_OUTER_INDICES) <= table


def test_lips_form_a_closed_ring():
    """Every consecutive pair in the tuple is a real MediaPipe lip edge.

    This is the check that makes the overlay's polyline meaningful. Membership
    alone would pass for a scrambled ordering, which would draw a star rather
    than a mouth -- and a star is easy to mistake for "the face subset is just
    noisy" when glancing at an overlay.
    """
    table = _edges(mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
    ring = lm.LIPS_OUTER_INDICES
    for a, b in zip(ring, ring[1:] + ring[:1]):
        assert frozenset((a, b)) in table, f"{a}-{b} is not a MediaPipe lip edge"


@pytest.mark.parametrize(
    ("ours", "table_name"),
    [
        (lm.LEFT_EYEBROW_INDICES, "FACE_LANDMARKS_LEFT_EYEBROW"),
        (lm.RIGHT_EYEBROW_INDICES, "FACE_LANDMARKS_RIGHT_EYEBROW"),
    ],
)
def test_eyebrow_indices_match_their_declared_side(ours, table_name):
    """The regression test for the inversion.

    Subject-relative naming: MediaPipe's LEFT_EYEBROW is the signer's own left,
    consistent with PoseLandmark.LEFT_SHOULDER. On an unmirrored frame that
    eyebrow appears on the viewer's right, which is how the names came to be
    swapped in the first place.
    """
    table = _vertices(getattr(mp_vision.FaceLandmarksConnections, table_name))
    assert set(ours) <= table


@pytest.mark.parametrize(
    ("ours", "table_name"),
    [
        (lm.LEFT_EYEBROW_INDICES, "FACE_LANDMARKS_LEFT_EYEBROW"),
        (lm.RIGHT_EYEBROW_INDICES, "FACE_LANDMARKS_RIGHT_EYEBROW"),
    ],
)
def test_eyebrow_indices_form_a_connected_chain(ours, table_name):
    table = _edges(getattr(mp_vision.FaceLandmarksConnections, table_name))
    for a, b in zip(ours, ours[1:]):
        assert frozenset((a, b)) in table, f"{a}-{b} is not a MediaPipe edge"


def test_eyebrows_are_not_the_same_side():
    """Cheap symmetry guard: the two tuples must not both name one eyebrow."""
    assert set(lm.LEFT_EYEBROW_INDICES).isdisjoint(lm.RIGHT_EYEBROW_INDICES)


def test_eye_corners_split_two_per_eye():
    left = _vertices(mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
    right = _vertices(mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
    corners = set(lm.EYE_CORNER_INDICES)
    assert len(corners & left) == 2
    assert len(corners & right) == 2
    assert corners <= (left | right)


def test_face_indices_avoid_the_iris_block():
    """Iris points (468-477) are appended by facemesh2_lite and not wanted here.

    Iris position tracks gaze at a finer grain than this project uses, and
    including it would make the feature vector depend on whether a future bundle
    ships the iris sub-model at all.
    """
    iris = _vertices(
        mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
    ) | _vertices(mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
    assert set(lm.FACE_INDICES).isdisjoint(iris)
    assert max(lm.FACE_INDICES) < min(iris)


# --- Pose ------------------------------------------------------------------


def test_shoulder_positions_address_the_shoulders():
    """The normaliser's origin and scale both hang off these two positions."""
    landmark = mp_vision.PoseLandmark
    assert lm.POSE_INDICES[lm.LEFT_SHOULDER_POS] == int(landmark.LEFT_SHOULDER)
    assert lm.POSE_INDICES[lm.RIGHT_SHOULDER_POS] == int(landmark.RIGHT_SHOULDER)


def test_pose_subset_excludes_everything_below_the_hips():
    """Signing happens above the waist (spec S4)."""
    landmark = mp_vision.PoseLandmark
    below_waist = {
        int(landmark.LEFT_KNEE), int(landmark.RIGHT_KNEE),
        int(landmark.LEFT_ANKLE), int(landmark.RIGHT_ANKLE),
        int(landmark.LEFT_HEEL), int(landmark.RIGHT_HEEL),
        int(landmark.LEFT_FOOT_INDEX), int(landmark.RIGHT_FOOT_INDEX),
    }
    assert set(lm.POSE_INDICES).isdisjoint(below_waist)
    assert int(landmark.LEFT_HIP) in lm.POSE_INDICES
    assert int(landmark.RIGHT_HIP) in lm.POSE_INDICES


def test_pose_connections_are_real_mediapipe_edges():
    """Our edges, mapped back to raw indices, must exist in MediaPipe's table."""
    table = _edges(mp_vision.PoseLandmarksConnections.POSE_LANDMARKS)
    for a, b in lm.POSE_CONNECTIONS:
        raw = frozenset((lm.POSE_INDICES[a], lm.POSE_INDICES[b]))
        assert raw in table, f"{sorted(raw)} is not a MediaPipe pose edge"


def test_pose_connections_cover_every_kept_edge():
    """No upper-body edge is silently missing from the overlay.

    A dropped edge would not break extraction -- the points are still in the
    feature vector -- but it would leave a gap in the drawn skeleton, which is
    exactly the kind of thing that gets read as a tracking failure.
    """
    kept = set(lm.POSE_INDICES)
    expected = {
        frozenset((c.start, c.end))
        for c in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
        if c.start in kept and c.end in kept
    }
    ours = {
        frozenset((lm.POSE_INDICES[a], lm.POSE_INDICES[b]))
        for a, b in lm.POSE_CONNECTIONS
    }
    assert ours == expected


# --- Hands -----------------------------------------------------------------


def test_hand_connections_match_exactly():
    """Hands keep all 21 points, so our table should equal MediaPipe's."""
    table = _edges(mp_vision.HandLandmarksConnections.HAND_CONNECTIONS)
    ours = {frozenset(edge) for edge in lm.HAND_CONNECTIONS}
    assert ours == table
