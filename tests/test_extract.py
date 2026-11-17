"""Normalisation logic in `extract.py`, on synthetic landmark arrays.

No MediaPipe, no model bundle, no video file. Everything here builds arrays by
hand, which is the point: normalisation is the part of extraction that can be
silently wrong. A broken MediaPipe setup announces itself in the overlay; a
wrong scale divisor or a mask written before translation instead of after
produces plausible-looking numbers that quietly poison training.

Geometry convention in these tests: "physical" coordinates are in units of
image *height*, so x and y are directly comparable. `physical_frame` converts
them into the per-axis-normalised space MediaPipe actually emits, for a given
aspect ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from msl.config import DataConfig, FeatureConfig
from msl.data import landmarks as lm
from msl.data.extract import (
    ClipLandmarks,
    FrameLandmarks,
    apply_aspect,
    cache_key,
    cache_path,
    normalise_frame,
    pack_clip,
    save_cached,
)

SHOULDER_SPAN = 0.30  # units of image height
BODY_CENTRE = (0.50, 0.40)


def physical_frame(
    *,
    aspect: float = 1.0,
    centre: tuple[float, float] = BODY_CENTRE,
    span: float = SHOULDER_SPAN,
    scale: float = 1.0,
    left_hand: bool = True,
    right_hand: bool = True,
    face: bool = True,
) -> FrameLandmarks:
    """Build a frame in MediaPipe's normalised space from physical geometry.

    MediaPipe divides x by frame width and y by frame height, so a physical
    x-offset of X (in image heights) arrives as X/aspect. Undoing that is
    exactly what `apply_aspect` does.
    """
    cx, cy = centre
    half = span * scale / 2.0

    def place(points: np.ndarray) -> np.ndarray:
        """physical (n,3) -> MediaPipe-normalised (n,3).

        `scale` models camera distance, so it must apply to all three axes. z
        carries no centre offset because the shoulders sit at z=0, making
        mid-shoulder z zero and the normalisation scale-invariant in z too.
        """
        out = points.astype(np.float64).copy()
        out[:, 0] = cx + out[:, 0] * scale / aspect
        out[:, 1] = cy + out[:, 1] * scale
        out[:, 2] = out[:, 2] * scale
        return out

    pose = np.zeros((lm.N_POSE, 3), dtype=np.float64)
    # A deterministic spread so non-shoulder joints are not all coincident.
    for i in range(lm.N_POSE):
        pose[i] = [0.02 * i - 0.14, 0.03 * i - 0.2, 0.01 * i]
    pose[lm.LEFT_SHOULDER_POS] = [+half / scale, 0.0, 0.0]
    pose[lm.RIGHT_SHOULDER_POS] = [-half / scale, 0.0, 0.0]
    pose = place(pose)

    def hand(x_offset: float) -> np.ndarray:
        pts = np.zeros((lm.N_HAND_LANDMARKS, 3), dtype=np.float64)
        for i in range(lm.N_HAND_LANDMARKS):
            pts[i] = [x_offset + 0.005 * i, 0.25 + 0.005 * i, 0.002 * i]
        return place(pts)

    face_points = None
    if face:
        pts = np.zeros((lm.N_FACE, 3), dtype=np.float64)
        for i in range(lm.N_FACE):
            pts[i] = [0.001 * i - 0.02, -0.20 + 0.001 * i, 0.0]
        face_points = place(pts)

    return FrameLandmarks(
        pose=pose,
        left_hand=hand(+0.18) if left_hand else None,
        right_hand=hand(-0.18) if right_hand else None,
        face=face_points,
    )


def normalise(frame: FrameLandmarks, *, width: int = 1000, height: int = 1000, **kwargs):
    """`normalise_frame` with frame dimensions supplied.

    Square by default so the aspect correction is a no-op and the other tests
    isolate what they are actually about. The aspect tests pass real dimensions.
    """
    return normalise_frame(frame, width=width, height=height, **kwargs)


def pose_block(vector: np.ndarray, use_z: bool = True) -> np.ndarray:
    coords = lm.coords_per_landmark(use_z)
    return vector[lm.block_slice("pose", use_z)].reshape(-1, coords)


def hand_block(vector: np.ndarray, side: str, use_z: bool = True) -> np.ndarray:
    coords = lm.coords_per_landmark(use_z)
    return vector[lm.block_slice(side, use_z)].reshape(-1, coords)


# --- Mid-shoulder centring -------------------------------------------------


def test_mid_shoulder_lands_on_the_origin() -> None:
    vector, ok = normalise(physical_frame())
    assert ok
    pose = pose_block(vector)
    midpoint = (pose[lm.LEFT_SHOULDER_POS] + pose[lm.RIGHT_SHOULDER_POS]) / 2.0
    assert midpoint == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_translating_the_whole_body_changes_nothing() -> None:
    """Invariance to where the signer stands in frame."""
    a, _ = normalise(physical_frame(centre=(0.5, 0.4)))
    b, _ = normalise(physical_frame(centre=(0.2, 0.7)))
    assert a == pytest.approx(b, abs=1e-6)


# --- Inter-shoulder scaling ------------------------------------------------


def test_shoulders_are_exactly_one_unit_apart_after_scaling() -> None:
    vector, ok = normalise(physical_frame())
    assert ok
    pose = pose_block(vector)
    span = np.linalg.norm(
        (pose[lm.LEFT_SHOULDER_POS] - pose[lm.RIGHT_SHOULDER_POS])[:2]
    )
    assert span == pytest.approx(1.0, abs=1e-6)


def test_distance_from_camera_changes_nothing() -> None:
    """The whole point of scaling: a signer twice as far away is the same sign."""
    near, _ = normalise(physical_frame(scale=1.0))
    far, _ = normalise(physical_frame(scale=0.5))
    assert near == pytest.approx(far, abs=1e-6)


def test_z_is_excluded_from_the_scale_divisor() -> None:
    """z is unreliable (spec S4.3). If it entered the divisor it would corrupt
    x and y too, not merely the z channel."""
    frame = physical_frame()
    pose = frame.pose.copy()
    pose[lm.LEFT_SHOULDER_POS, 2] = 5.0  # absurd depth on one shoulder only
    pose[lm.RIGHT_SHOULDER_POS, 2] = -5.0
    perturbed = FrameLandmarks(
        pose=pose,
        left_hand=frame.left_hand,
        right_hand=frame.right_hand,
        face=frame.face,
    )

    baseline, _ = normalise(frame)
    result, ok = normalise(perturbed)
    assert ok

    # x and y are untouched by the z perturbation...
    for block in ("pose", "left_hand", "right_hand", "face"):
        got = result[lm.block_slice(block)].reshape(-1, 3)
        want = baseline[lm.block_slice(block)].reshape(-1, 3)
        assert got[:, :2] == pytest.approx(want[:, :2], abs=1e-6), block
    # ...but the z channel of the shoulders did move, so the test is not vacuous.
    assert pose_block(result)[lm.LEFT_SHOULDER_POS, 2] != pytest.approx(
        pose_block(baseline)[lm.LEFT_SHOULDER_POS, 2]
    )


# --- Aspect ratio ----------------------------------------------------------


def test_same_pose_at_different_aspect_ratios_gives_identical_features() -> None:
    """MediaPipe normalises x by width and y by height, so on a 16:9 frame a
    horizontal distance and an equal vertical one arrive as different numbers.
    Uncorrected, the inter-shoulder divisor -- almost entirely horizontal --
    would be wrong relative to the vertical hand motion it normalises.

    This cannot be caught by looking at an overlay: an overlay drawn in pixel
    space is correct whether or not the features are.
    """
    square, _ = normalise(physical_frame(aspect=1.0), width=900, height=900)
    wide, _ = normalise(physical_frame(aspect=16 / 9), width=1600, height=900)
    assert square == pytest.approx(wide, abs=1e-6)


def test_apply_aspect_touches_only_x() -> None:
    points = np.array([[0.5, 0.25, 0.125]])
    out = apply_aspect(points, 2000, 1000)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.25)
    assert out[0, 2] == pytest.approx(0.125)
    assert points[0, 0] == pytest.approx(0.5), "input must not be mutated"


def test_apply_aspect_rejects_a_degenerate_frame_size() -> None:
    with pytest.raises(ValueError, match="frame size"):
        apply_aspect(np.zeros((1, 3)), 0, 1000)


def test_normalise_frame_demands_frame_dimensions() -> None:
    """No default aspect. A default of 1.0 would silently assume square and
    reintroduce the distortion on every non-square source (spec S4).
    """
    with pytest.raises(TypeError):
        normalise_frame(physical_frame())  # type: ignore[call-arg]


def test_the_same_clip_at_two_source_aspects_needs_its_own_correction() -> None:
    """Guards the inference-time failure spec S4 calls out: applying one
    corpus's aspect to another camera's frames. Normalising 16:9 landmarks as
    though they were square must NOT match the correct result -- if it did,
    the correction would be doing nothing.
    """
    frame = physical_frame(aspect=16 / 9)
    correct, _ = normalise(frame, width=1600, height=900)
    assumed_square, _ = normalise(frame, width=900, height=900)
    assert correct != pytest.approx(assumed_square, abs=1e-4)


# --- Missing-hand mask -----------------------------------------------------


def test_present_hands_set_the_visibility_flags() -> None:
    vector, _ = normalise(physical_frame())
    assert vector[lm.visibility_slice()].tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    "left, right, expected",
    [(True, False, [1.0, 0.0]), (False, True, [0.0, 1.0]), (False, False, [0.0, 0.0])],
)
def test_absent_hands_clear_the_matching_flag(
    left: bool, right: bool, expected: list[float]
) -> None:
    frame = physical_frame(left_hand=left, right_hand=right)
    vector, ok = normalise(frame)
    assert ok, "a missing hand must not invalidate the frame"
    assert vector[lm.visibility_slice()].tolist() == expected


def test_absent_hand_is_exactly_zero_not_the_translated_origin() -> None:
    """The ordering bug this guards against.

    Zeros must be written *after* normalisation. Zeroing a hand first and then
    translating would place it at -mid_shoulder/scale -- a resting position in
    front of the chest, entirely plausible and completely wrong. The mask would
    then be the only signal, and any model ignoring it would learn a phantom
    hand pose.
    """
    vector, _ = normalise(physical_frame(left_hand=False))
    left = hand_block(vector, "left_hand")
    assert np.all(left == 0.0), "absent hand must be exactly zero"

    # And prove the test is not vacuous: the present hand is nowhere near zero.
    right = hand_block(vector, "right_hand")
    assert np.abs(right).max() > 0.1


def test_a_hand_at_the_body_origin_is_distinguishable_from_an_absent_one() -> None:
    """Both produce a near-zero coordinate block; only the mask separates them.
    This is the exact ambiguity spec S4 introduces the mask to resolve."""
    at_origin = physical_frame()
    hand = np.zeros((lm.N_HAND_LANDMARKS, 3))
    # Put every joint of the left hand exactly on the mid-shoulder point.
    hand[:] = at_origin.pose[lm.LEFT_SHOULDER_POS] / 2 + (
        at_origin.pose[lm.RIGHT_SHOULDER_POS] / 2
    )
    present = FrameLandmarks(pose=at_origin.pose, left_hand=hand, right_hand=None)

    vector, _ = normalise(present)
    assert hand_block(vector, "left_hand") == pytest.approx(0.0, abs=1e-6)
    assert hand_block(vector, "right_hand") == pytest.approx(0.0, abs=1e-6)
    # Identical coordinates, opposite meanings -- resolved only by the mask.
    assert vector[lm.visibility_slice()].tolist() == [1.0, 0.0]


# --- Degenerate frames -----------------------------------------------------


def test_missing_pose_yields_a_zero_frame_marked_absent() -> None:
    vector, ok = normalise(FrameLandmarks(pose=None))
    assert not ok
    assert np.all(vector == 0.0)
    assert vector[lm.visibility_slice()].tolist() == [0.0, 0.0]


def test_coincident_shoulders_do_not_divide_by_zero() -> None:
    """A collapsed pose must not seed the cache with infinities."""
    frame = physical_frame()
    pose = frame.pose.copy()
    pose[lm.RIGHT_SHOULDER_POS] = pose[lm.LEFT_SHOULDER_POS]
    vector, ok = normalise(FrameLandmarks(pose=pose, left_hand=frame.left_hand))

    assert not ok
    assert np.all(np.isfinite(vector))
    assert np.all(vector == 0.0)


def test_shoulder_distance_just_above_threshold_still_normalises() -> None:
    """The guard must not swallow legitimately small-but-real signers."""
    tiny = physical_frame(span=0.02)
    _, ok = normalise(tiny, min_shoulder_distance=1e-3)
    assert ok


def test_nan_landmarks_are_rejected_rather_than_propagated() -> None:
    frame = physical_frame()
    pose = frame.pose.copy()
    pose[lm.LEFT_SHOULDER_POS, 0] = np.nan
    vector, ok = normalise(FrameLandmarks(pose=pose))
    assert not ok
    assert np.all(np.isfinite(vector))


# --- Shape validation ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pose": np.zeros((7, 3))}, "pose"),
        ({"pose": np.zeros((lm.N_POSE, 2))}, "pose"),
    ],
)
def test_wrong_shapes_raise_with_the_block_named(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalise(FrameLandmarks(**kwargs))


def test_wrong_hand_shape_raises() -> None:
    frame = physical_frame()
    bad = FrameLandmarks(pose=frame.pose, left_hand=np.zeros((5, 3)))
    with pytest.raises(ValueError, match="left_hand"):
        normalise(bad)


# --- use_z ablation --------------------------------------------------------


@pytest.mark.parametrize("use_z", [True, False])
def test_feature_vector_length_matches_the_config(use_z: bool) -> None:
    vector, _ = normalise(physical_frame(), use_z=use_z)
    assert vector.shape == (lm.feature_dim(use_z),)
    assert vector.dtype == np.float32


def test_dropping_z_preserves_the_x_and_y_channels() -> None:
    with_z, _ = normalise(physical_frame(), use_z=True)
    without_z, _ = normalise(physical_frame(), use_z=False)
    for block, _ in lm.FEATURE_LAYOUT:
        xyz = with_z[lm.block_slice(block, True)].reshape(-1, 3)
        xy = without_z[lm.block_slice(block, False)].reshape(-1, 2)
        assert xy == pytest.approx(xyz[:, :2], abs=1e-6), block


# --- Clip packing ----------------------------------------------------------


def test_pack_clip_reports_detection_rates() -> None:
    frames = [
        physical_frame(),
        physical_frame(left_hand=False),
        FrameLandmarks(pose=None),
        physical_frame(),
    ]
    clip = pack_clip(frames, fps=30.0, width=1280, height=720)

    assert clip.n_frames == 4
    assert clip.features.shape == (4, lm.feature_dim(True))
    assert clip.pose_present.tolist() == [True, True, False, True]

    rates = clip.detection_rate()
    assert rates["pose"] == pytest.approx(0.75)
    assert rates["left_hand"] == pytest.approx(0.5)  # absent in frames 1 and 2
    assert rates["right_hand"] == pytest.approx(0.75)


def test_pack_clip_uses_the_frame_aspect_ratio() -> None:
    """pack_clip must derive aspect from width/height, not assume square."""
    frames = [physical_frame(aspect=16 / 9)]
    wide = pack_clip(frames, fps=30.0, width=1600, height=900)
    square = pack_clip([physical_frame(aspect=1.0)], fps=30.0, width=900, height=900)
    assert wide.features[0] == pytest.approx(square.features[0], abs=1e-6)


def test_pack_clip_rejects_a_degenerate_frame_size() -> None:
    with pytest.raises(ValueError, match="frame size"):
        pack_clip([physical_frame()], fps=30.0, width=0, height=720)


def test_clip_rejects_mismatched_frame_counts() -> None:
    with pytest.raises(ValueError, match="frame count"):
        ClipLandmarks(
            features=np.zeros((3, lm.feature_dim(True)), dtype=np.float32),
            pose_present=np.zeros(2, dtype=bool),
            fps=30.0,
            width=640,
            height=480,
            use_z=True,
        )


# --- Cache -----------------------------------------------------------------


def _write(path, payload: bytes):
    path.write_bytes(payload)
    return path


def test_cache_key_tracks_file_content_not_path(tmp_path) -> None:
    a = _write(tmp_path / "a.mp4", b"first recording")
    b = _write(tmp_path / "b.mp4", b"first recording")
    c = _write(tmp_path / "c.mp4", b"a re-record of the same prompt")
    feature = FeatureConfig()

    assert cache_key(a, feature) == cache_key(b, feature)
    assert cache_key(a, feature) != cache_key(c, feature)


def test_cache_key_ignores_use_z_because_the_cache_stores_the_superset(tmp_path) -> None:
    """One cache entry per clip, not one per consumption setting.

    The cache always stores z; a use_z=False run narrows at load time via
    landmarks.drop_z. Keying on use_z would mean the default flip to False
    invalidated every existing cache file and forced hours of re-extraction to
    produce strictly less data than is already on disk.
    """
    video = _write(tmp_path / "clip.mp4", b"bytes")
    assert cache_key(video, FeatureConfig(use_z=True)) == cache_key(
        video, FeatureConfig(use_z=False)
    )


def test_cache_path_lands_under_the_configured_root(tmp_path) -> None:
    video = _write(tmp_path / "clip.mp4", b"bytes")
    data = DataConfig(data_root=str(tmp_path / "external"))
    path = cache_path(video, FeatureConfig(), data.resolved_cache_dir)
    assert path.parent == tmp_path / "external" / "landmark_cache"
    assert path.suffix == ".npy"


def test_explicit_cache_dir_overrides_the_data_root() -> None:
    data = DataConfig(data_root="data", cache_dir="D:/msl-cache")
    assert str(data.resolved_cache_dir).replace("\\", "/") == "D:/msl-cache"


def test_missing_model_asset_names_the_file_and_the_fix(tmp_path) -> None:
    """The first error a new machine hits. It must say what to download and
    where to put it, not just that something was absent."""
    from msl.data.extract import HOLISTIC_MODEL_URL, LandmarkExtractor, MissingModelAssetError

    data = DataConfig(assets_dir=str(tmp_path / "no-such-dir"))
    with pytest.raises(MissingModelAssetError) as excinfo:
        with LandmarkExtractor(data):
            pass  # pragma: no cover - construction raises first

    message = str(excinfo.value)
    assert "holistic_landmarker.task" in message
    assert HOLISTIC_MODEL_URL in message


def test_live_frame_detection_requires_an_open_extractor() -> None:
    from msl.data.extract import LandmarkExtractor

    extractor = LandmarkExtractor()
    with pytest.raises(RuntimeError, match="context manager"):
        extractor.detect_rgb_frame(np.zeros((8, 8, 3), dtype=np.uint8), 0)


def test_save_cached_is_atomic_and_leaves_no_temp_file(tmp_path) -> None:
    clip = pack_clip([physical_frame()], fps=30.0, width=640, height=480)
    destination = save_cached(clip, tmp_path / "nested" / "clip.npy")

    assert destination.exists()
    assert list(tmp_path.rglob("*.tmp")) == []
    assert np.load(destination) == pytest.approx(clip.features)


# --- Segment-aware cache keys ----------------------------------------------


def test_segments_of_one_file_get_distinct_cache_keys(tmp_path) -> None:
    """WLASL cuts several signs from one compilation video, so those clips are
    byte-identical on disk. A content-only key gives them a single cache entry:
    the first extraction wins and every other gloss silently loads it under its
    own label. 114 of 1127 WLASL100 clips share a file with another clip.
    """
    video = _write(tmp_path / "compilation.mp4", b"one long video")
    feature = FeatureConfig()
    book = cache_key(video, feature, segment=(2150, 2249))
    school = cache_key(video, feature, segment=(1068, 1179))
    whole = cache_key(video, feature, segment=None)
    assert len({book, school, whole}) == 3


def test_same_segment_still_hits_the_cache(tmp_path) -> None:
    video = _write(tmp_path / "clip.mp4", b"bytes")
    feature = FeatureConfig()
    assert cache_key(video, feature, (1, 50)) == cache_key(video, feature, (1, 50))


def test_segment_reaches_the_cache_path(tmp_path) -> None:
    video = _write(tmp_path / "clip.mp4", b"bytes")
    a = cache_path(video, FeatureConfig(), tmp_path, segment=(1, 10))
    b = cache_path(video, FeatureConfig(), tmp_path, segment=(11, 20))
    assert a != b and a.suffix == ".npy"
