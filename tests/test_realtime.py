from __future__ import annotations

import numpy as np
import pytest

from msl.data.extract import FrameLandmarks
from msl.data.realtime import _mirror_landmarks, _should_quit


def test_mirror_landmarks_flips_only_x_and_does_not_mutate_input() -> None:
    pose = np.array([[0.2, 0.3, 0.4]])
    mirrored = _mirror_landmarks(FrameLandmarks(pose=pose))

    assert mirrored.pose is not None
    assert mirrored.pose[0].tolist() == pytest.approx([0.8, 0.3, 0.4])
    assert pose[0].tolist() == pytest.approx([0.2, 0.3, 0.4])
    assert mirrored.left_hand is None


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_quit_key_accepts_q_in_both_cases_and_escape(key: int) -> None:
    assert _should_quit(key)


def test_quit_key_ignores_other_keys() -> None:
    assert not _should_quit(ord("a"))
