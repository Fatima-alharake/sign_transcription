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
