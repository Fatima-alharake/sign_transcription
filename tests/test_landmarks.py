"""Landmark layout arithmetic.

Cheap tests, but the feature dimension is a number that four other modules
agree on by assumption. If it silently drifts, the symptom is a shape error
somewhere far away, or worse, a model that trains on misaligned channels.
"""

from __future__ import annotations

import pytest

from msl.data import landmarks as lm


def test_pose_subset_is_upper_body_only() -> None:
    assert lm.POSE_INDICES[0] == 0  # nose, head-position anchor
    assert set(lm.POSE_INDICES[1:]) == set(range(11, 25))
    assert len(lm.POSE_INDICES) == 15
    assert max(lm.POSE_INDICES) < 33  # MediaPipe Pose has 33 landmarks


def test_shoulder_positions_index_into_the_kept_subset() -> None:
    assert lm.POSE_INDICES[lm.LEFT_SHOULDER_POS] == 11
    assert lm.POSE_INDICES[lm.RIGHT_SHOULDER_POS] == 12


def test_face_subset_has_no_duplicates_and_is_a_small_fraction_of_the_mesh() -> None:
    assert len(set(lm.FACE_INDICES)) == len(lm.FACE_INDICES)
    # The installed model emits 478 points: the 468-point canonical mesh plus
    # 10 iris points at 468-477. Staying under 468 keeps every index on a
    # canonical mesh point and off the iris block, whose landmarks carry no
    # grammatical signal and move with gaze rather than with the sign.
    assert max(lm.FACE_INDICES) < 468
    # The point of subsetting: keep grammar-carrying points, drop the noise.
    assert len(lm.FACE_INDICES) < 50


def test_total_landmark_count_matches_the_blocks() -> None:
    assert lm.N_LANDMARKS == sum(count for _, count in lm.FEATURE_LAYOUT)
    assert lm.N_LANDMARKS == 15 + 21 + 21 + len(lm.FACE_INDICES)


@pytest.mark.parametrize("use_z, coords", [(True, 3), (False, 2)])
def test_feature_dim_includes_two_visibility_channels(use_z: bool, coords: int) -> None:
    assert lm.feature_dim(use_z) == lm.N_LANDMARKS * coords + 2


def test_blocks_tile_the_frame_vector_without_gaps_or_overlap() -> None:
    for use_z in (True, False):
        cursor = 0
        for name, _ in lm.FEATURE_LAYOUT:
            block = lm.block_slice(name, use_z)
            assert block.start == cursor, f"{name} does not start where the last block ended"
            cursor = block.stop
        visibility = lm.visibility_slice(use_z)
        assert visibility.start == cursor
        assert visibility.stop == lm.feature_dim(use_z)


def test_hand_blocks_are_the_same_size() -> None:
    for use_z in (True, False):
        left = lm.block_slice("left_hand", use_z)
        right = lm.block_slice("right_hand", use_z)
        assert left.stop - left.start == right.stop - right.start


def test_unknown_block_raises() -> None:
    with pytest.raises(KeyError):
        lm.block_slice("elbow")


def test_drop_z_matches_extraction_with_use_z_false():
    """Slicing z out of a use_z=True frame == extracting with use_z=False.

    This is what makes the z ablation free: the cache holds 275-dim tensors and
    the ablation slices them instead of re-running MediaPipe over 1127 clips.
    If normalisation ever makes x or y depend on z, this test fails and the
    ablation stops being valid.
    """
    import numpy as np

    from msl.data import landmarks as lm
    from msl.data.extract import FrameLandmarks, normalise_frame

    rng = np.random.default_rng(7)
    pose = rng.normal(0.5, 0.15, size=(lm.N_POSE, 3))
    # a real shoulder separation so the frame normalises rather than degenerating
    pose[lm.LEFT_SHOULDER_POS] = [0.40, 0.45, 0.02]
    pose[lm.RIGHT_SHOULDER_POS] = [0.60, 0.45, -0.02]
    left = rng.normal(0.4, 0.1, size=(lm.N_HAND_LANDMARKS, 3))
    right = rng.normal(0.6, 0.1, size=(lm.N_HAND_LANDMARKS, 3))
    face = rng.normal(0.5, 0.05, size=(lm.N_FACE, 3))

    frame = FrameLandmarks(pose=pose, left_hand=left, right_hand=right, face=face)
    # 16:9, so the aspect correction is actually exercised rather than being 1.0
    wide, ok_wide = normalise_frame(frame, width=1920, height=1080, use_z=True)
    narrow, ok_narrow = normalise_frame(frame, width=1920, height=1080, use_z=False)
    assert ok_wide and ok_narrow

    sliced = lm.drop_z(wide[None, :])[0]
    np.testing.assert_allclose(sliced, narrow, atol=1e-6)


def test_drop_z_preserves_absent_hands_and_flags():
    import numpy as np

    from msl.data import landmarks as lm
    from msl.data.extract import FrameLandmarks, normalise_frame

    rng = np.random.default_rng(8)
    pose = rng.normal(0.5, 0.15, size=(lm.N_POSE, 3))
    pose[lm.LEFT_SHOULDER_POS] = [0.40, 0.45, 0.0]
    pose[lm.RIGHT_SHOULDER_POS] = [0.60, 0.45, 0.0]
    face = rng.normal(0.5, 0.05, size=(lm.N_FACE, 3))
    right = rng.normal(0.6, 0.1, size=(lm.N_HAND_LANDMARKS, 3))

    frame = FrameLandmarks(pose=pose, left_hand=None, right_hand=right, face=face)
    wide, _ = normalise_frame(frame, width=1920, height=1080, use_z=True)
    sliced = lm.drop_z(wide[None, :])[0]
    narrow, _ = normalise_frame(frame, width=1920, height=1080, use_z=False)
    np.testing.assert_allclose(sliced, narrow, atol=1e-6)
    # absent left hand stays zero, and its flag stays off, after slicing
    assert np.all(sliced[lm.block_slice("left_hand", False)] == 0.0)
    assert sliced[lm.visibility_slice(False)][0] == 0.0
    assert sliced[lm.visibility_slice(False)][1] == 1.0


def test_drop_z_rejects_a_tensor_that_is_already_narrow():
    import numpy as np
    import pytest

    from msl.data import landmarks as lm

    narrow = np.zeros((4, lm.feature_dim(False)), dtype=np.float32)
    with pytest.raises(ValueError, match="use_z=True"):
        lm.drop_z(narrow)


def test_scale_z_touches_only_the_z_channel():
    import numpy as np

    from msl.data import landmarks as lm

    rng = np.random.default_rng(11)
    f = rng.normal(0.4, 0.3, size=(6, lm.feature_dim(True))).astype(np.float32)
    out = lm.scale_z(f, 0.5)
    c_in = f[:, : lm.N_LANDMARKS * 3].reshape(6, lm.N_LANDMARKS, 3)
    c_out = out[:, : lm.N_LANDMARKS * 3].reshape(6, lm.N_LANDMARKS, 3)
    np.testing.assert_allclose(c_out[:, :, :2], c_in[:, :, :2], atol=1e-6)
    np.testing.assert_allclose(c_out[:, :, 2], c_in[:, :, 2] * 0.5, atol=1e-6)
    # visibility flags sit outside the coordinate region and must be untouched
    np.testing.assert_allclose(out[:, lm.visibility_slice(True)],
                               f[:, lm.visibility_slice(True)], atol=1e-6)


def test_scale_z_keeps_zeroed_blocks_zero():
    import numpy as np

    from msl.data import landmarks as lm

    f = np.zeros((4, lm.feature_dim(True)), dtype=np.float32)
    f[:, lm.block_slice("pose", True)] = 0.3
    out = lm.scale_z(f, 0.7)
    assert np.all(out[:, lm.block_slice("left_hand", True)] == 0.0)


def test_scale_z_does_not_mutate_its_input():
    import numpy as np

    from msl.data import landmarks as lm

    f = np.ones((3, lm.feature_dim(True)), dtype=np.float32)
    before = f.copy()
    lm.scale_z(f, 0.5)
    np.testing.assert_array_equal(f, before)


def _wide(n=5, seed=3):
    import numpy as np

    from msl.data import landmarks as lm

    rng = np.random.default_rng(seed)
    f = rng.normal(0.3, 0.4, size=(n, lm.feature_dim(True))).astype(np.float32)
    f[:, lm.visibility_slice(True)] = [1.0, 0.0]
    return f


def test_feature_dim_for_matches_the_block_layout():
    from msl.data import landmarks as lm

    assert lm.feature_dim_for(()) == lm.feature_dim(False) == 184
    assert lm.feature_dim_for(("pose",)) == 199
    assert lm.feature_dim_for(("left_hand", "right_hand")) == 226
    assert lm.feature_dim_for(tuple(lm.Z_BLOCK_NAMES)) == lm.feature_dim(True) == 275
    assert lm.feature_dim_for(None) == lm.feature_dim(True)


def test_feature_dim_for_rejects_an_unknown_block():
    import pytest

    from msl.data import landmarks as lm

    with pytest.raises(KeyError, match="unknown feature block"):
        lm.feature_dim_for(("torso",))


def test_select_no_blocks_equals_drop_z():
    """The empty selection must be exactly the use_z=False layout."""
    import numpy as np

    from msl.data import landmarks as lm

    f = _wide()
    np.testing.assert_allclose(lm.select_z_blocks(f, ()), lm.drop_z(f), atol=0)


def test_select_all_blocks_is_the_identity():
    import numpy as np

    from msl.data import landmarks as lm

    f = _wide()
    np.testing.assert_allclose(
        lm.select_z_blocks(f, tuple(lm.Z_BLOCK_NAMES)), f, atol=0
    )


def test_pose_z_only_keeps_pose_depth_and_drops_hand_depth():
    import numpy as np

    from msl.data import landmarks as lm

    f = _wide()
    out = lm.select_z_blocks(f, ("pose",))
    assert out.shape[1] == lm.feature_dim_for(("pose",))
    coords = f[:, : lm.N_LANDMARKS * 3].reshape(f.shape[0], lm.N_LANDMARKS, 3)
    # pose block occupies the first N_POSE landmarks and keeps all three coords
    np.testing.assert_allclose(
        out[:, : lm.N_POSE * 3].reshape(f.shape[0], lm.N_POSE, 3),
        coords[:, : lm.N_POSE, :], atol=0,
    )
    # the next block is the left hand, now two coords per landmark
    hand = out[:, lm.N_POSE * 3: lm.N_POSE * 3 + lm.N_HAND_LANDMARKS * 2]
    np.testing.assert_allclose(
        hand.reshape(f.shape[0], lm.N_HAND_LANDMARKS, 2),
        coords[:, lm.N_POSE: lm.N_POSE + lm.N_HAND_LANDMARKS, :2], atol=0,
    )


def test_select_preserves_visibility_flags_last():
    import numpy as np

    from msl.data import landmarks as lm

    f = _wide()
    for keep in ((), ("pose",), ("left_hand", "right_hand")):
        out = lm.select_z_blocks(f, keep)
        np.testing.assert_allclose(out[:, -2:], f[:, lm.visibility_slice(True)], atol=0)


def test_select_keeps_zeroed_blocks_zero():
    import numpy as np

    from msl.data import landmarks as lm

    f = _wide()
    f[:, lm.block_slice("left_hand", True)] = 0.0
    for keep in ((), ("pose",), ("left_hand",)):
        out = lm.select_z_blocks(f, keep)
        start = lm.N_POSE * (3 if "pose" in keep else 2)
        width = lm.N_HAND_LANDMARKS * (3 if "left_hand" in keep else 2)
        assert np.all(out[:, start:start + width] == 0.0)
