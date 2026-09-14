"""Tests for the landmark overlay renderer.

Every test here builds its landmarks by hand, so the suite needs no video, no
camera and no MediaPipe model bundle -- the same constraint the extraction core
was written under.

There is a limit to what a test can say about a renderer. That the lip ring
lands on a real mouth is not assertable without a real mouth; that is what the
CLI and a human's eyes are for. What *is* assertable is the machinery around it:
that the pixel mapping is the one intended, that nothing is mutated, that every
kept point is actually drawn, and that absent landmarks degrade quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from msl.data import landmarks as lm
from msl.data.extract import FrameLandmarks
from msl.data.overlay import (
    DrawStyle,
    COLOURS,
    draw_frame,
    draw_legend,
    face_bounds,
    face_panel,
    hand_side_report,
    radial_offsets,
    to_pixels,
)

WIDTH, HEIGHT = 320, 240


def blank(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _grid(n: int, *, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """n points on a line in normalised coords, with a z channel of zeros."""
    points = np.zeros((n, 3), dtype=np.float64)
    points[:, 0] = np.linspace(x0, x1, n)
    points[:, 1] = np.linspace(y0, y1, n)
    return points


def a_pose() -> np.ndarray:
    """A plausible upper-body pose: shoulders apart, everything in frame."""
    pose = np.full((lm.N_POSE, 3), 0.5, dtype=np.float64)
    pose[lm.LEFT_SHOULDER_POS] = (0.6, 0.4, 0.0)
    pose[lm.RIGHT_SHOULDER_POS] = (0.4, 0.4, 0.0)
    return pose


def a_face_ring() -> np.ndarray:
    """A face block whose lip positions trace an ellipse, as a real one does.

    `a_face` lays every point on a straight line, which is fine for "is it
    drawn" but degenerate for anything about ring geometry.
    """
    points = np.zeros((lm.N_FACE, 3), dtype=np.float64)
    lower = np.linspace(np.pi, 2 * np.pi, 11)
    upper = np.linspace(2 * np.pi, 3 * np.pi, 11)[1:-1]
    for i, angle in enumerate(list(lower) + list(upper)):
        points[i] = (0.5 + 0.04 * np.cos(angle), 0.6 - 0.02 * np.sin(angle), 0.0)
    points[20:] = _grid(lm.N_FACE - 20, x0=0.45, x1=0.55, y0=0.3, y1=0.34)
    return points


def a_face() -> np.ndarray:
    """A face block spread over a small region, so the crop has extent."""
    return _grid(lm.N_FACE, x0=0.45, x1=0.55, y0=0.25, y1=0.35)


def a_hand(centre_x: float) -> np.ndarray:
    return _grid(lm.N_HAND_LANDMARKS, x0=centre_x - 0.05, x1=centre_x + 0.05,
                 y0=0.55, y1=0.65)


def full_frame() -> FrameLandmarks:
    return FrameLandmarks(
        pose=a_pose(),
        left_hand=a_hand(0.7),
        right_hand=a_hand(0.3),
        face=a_face(),
    )


# --- Pixel mapping ---------------------------------------------------------


def test_to_pixels_scales_x_by_width_and_y_by_height():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.25, 9.9]])
    pixels = to_pixels(points, 640, 480)
    assert pixels.tolist() == [[0, 0], [640, 480], [320, 120]]


def test_to_pixels_is_deliberately_not_aspect_corrected():
    """Guards the design decision in the module docstring.

    An overlay that applied aspect correction would place landmarks off the
    signer, and worse, it would make a correct extraction pipeline look broken
    while making a broken one look fine. The correction belongs in the feature
    path only. If someone "fixes" to_pixels to multiply x by width/height, this
    fails and the docstring explains why.
    """
    points = np.array([[0.5, 0.5, 0.0]])
    wide = to_pixels(points, 1920, 1080)
    square = to_pixels(points, 1080, 1080)
    # x is width/2 in both cases -- it tracks the frame, not the aspect ratio.
    assert wide[0][0] == 960
    assert square[0][0] == 540
    assert wide[0][1] == square[0][1] == 540


def test_to_pixels_drops_z():
    points = np.array([[0.5, 0.5, -3.0], [0.5, 0.5, 3.0]])
    pixels = to_pixels(points, WIDTH, HEIGHT)
    assert pixels.shape == (2, 2)
    assert pixels[0].tolist() == pixels[1].tolist()


def test_to_pixels_does_not_mutate_input():
    points = np.array([[0.5, 0.5, 0.0]])
    before = points.copy()
    to_pixels(points, WIDTH, HEIGHT)
    assert np.array_equal(points, before)


# --- draw_frame ------------------------------------------------------------


def test_draw_frame_does_not_mutate_the_input_image():
    image = blank()
    before = image.copy()
    draw_frame(image, full_frame())
    assert np.array_equal(image, before)


def test_draw_frame_returns_a_new_array():
    image = blank()
    assert draw_frame(image, full_frame()) is not image


def test_draw_frame_marks_pixels():
    drawn = draw_frame(blank(), full_frame())
    assert drawn.any(), "nothing was drawn"


def test_draw_frame_on_an_empty_frame_draws_nothing():
    """No pose, no hands, no face: the frame should come back untouched.

    MediaPipe returns exactly this when it loses the subject, and a renderer
    that scribbled on those frames would make a tracking dropout hard to see in
    the output video -- which is the one thing the operator most needs to see.
    """
    image = blank()
    drawn = draw_frame(image, FrameLandmarks(), label_hands=False)
    assert np.array_equal(drawn, image)


@pytest.mark.parametrize("missing", ["pose", "left_hand", "right_hand", "face"])
def test_draw_frame_tolerates_each_block_being_absent(missing):
    frame = full_frame()
    frame = FrameLandmarks(
        **{
            field: (None if field == missing else getattr(frame, field))
            for field in ("pose", "left_hand", "right_hand", "face")
        }
    )
    assert draw_frame(blank(), frame).any()


def test_each_hand_is_drawn_in_its_own_colour():
    """A left/right swap must be visible as a colour change, not a nuance.

    The renderer is the last line of defence against HolisticLandmarker
    assigning hands wrongly, and it only works if the two are told apart at a
    glance (spec S4b treats the silent swap as a first-class hazard).
    """
    drawn = draw_frame(blank(), full_frame(), label_hands=False)
    present = {tuple(int(c) for c in px) for px in drawn.reshape(-1, 3)}
    assert COLOURS["left_hand"] in present
    assert COLOURS["right_hand"] in present
    assert COLOURS["left_hand"] != COLOURS["right_hand"]


def test_every_kept_face_point_is_drawn():
    """No point in the model's input is invisible in the overlay.

    A point that is extracted but never drawn is a point whose index can be
    wrong forever without anyone noticing -- the exact failure mode this
    milestone exists to close.
    """
    image = blank(640, 640)
    frame = FrameLandmarks(face=a_face())
    drawn = draw_frame(image, frame)
    pixels = to_pixels(frame.face, 640, 640)
    for name, order, _ in lm.FACE_GROUPS:
        for position in order:
            x, y = pixels[position]
            patch = drawn[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
            assert patch.any(), f"{name} point {lm.FACE_INDICES[position]} not drawn"


def test_hand_labels_add_ink():
    frame = full_frame()
    with_labels = draw_frame(blank(), frame, label_hands=True)
    without = draw_frame(blank(), frame, label_hands=False)
    assert with_labels.astype(int).sum() > without.astype(int).sum()


def test_landmarks_at_the_frame_edge_do_not_raise():
    """Hands leave frame constantly; clipping is OpenCV's job, not a crash."""
    edge = np.zeros((lm.N_HAND_LANDMARKS, 3))
    edge[:, 0] = np.linspace(-0.2, 1.2, lm.N_HAND_LANDMARKS)
    edge[:, 1] = np.linspace(-0.2, 1.2, lm.N_HAND_LANDMARKS)
    draw_frame(blank(), FrameLandmarks(left_hand=edge))


# --- DrawStyle -------------------------------------------------------------


@pytest.mark.parametrize("height", [240, 480, 720, 1080, 2160])
def test_draw_style_stays_positive_at_every_resolution(height):
    style = DrawStyle.for_frame(height)
    assert style.point_radius >= 1
    assert style.line_thickness >= 1
    assert style.font_scale > 0


def test_draw_style_grows_with_resolution():
    assert DrawStyle.for_frame(2160).point_radius > DrawStyle.for_frame(480).point_radius


# --- Legend ----------------------------------------------------------------


def test_legend_does_not_mutate_and_adds_ink():
    image = blank(640, 480)
    before = image.copy()
    out = draw_legend(image)
    assert np.array_equal(image, before)
    assert out.any()


# --- Face panel ------------------------------------------------------------


def test_face_panel_returns_none_without_a_face():
    assert face_panel(blank(), FrameLandmarks(pose=a_pose())) is None


def test_face_panel_is_magnified():
    """The panel exists because 34 points on a full frame are unreadable."""
    frame = FrameLandmarks(face=a_face())
    panel = face_panel(blank(1920, 1080), frame, size=640)
    assert panel is not None
    x0, y0, x1, y1 = face_bounds(frame.face, 1920, 1080)
    assert max(panel.shape[:2]) == 640
    assert max(panel.shape[:2]) > max(x1 - x0, y1 - y0)


def test_face_bounds_stay_inside_the_frame():
    """A face near the edge must not produce a negative or overrunning crop."""
    face = _grid(lm.N_FACE, x0=0.0, x1=0.04, y0=0.0, y1=0.04)
    x0, y0, x1, y1 = face_bounds(face, WIDTH, HEIGHT)
    assert 0 <= x0 < x1 <= WIDTH
    assert 0 <= y0 < y1 <= HEIGHT


def test_face_bounds_handles_a_degenerate_face():
    """All 34 points coincident: pad by at least a pixel rather than 0x0."""
    face = np.full((lm.N_FACE, 3), 0.5)
    x0, y0, x1, y1 = face_bounds(face, WIDTH, HEIGHT)
    assert x1 > x0 and y1 > y0


def test_face_panel_does_not_mutate_the_input_image():
    image = blank(640, 480)
    before = image.copy()
    face_panel(image, FrameLandmarks(face=a_face()))
    assert np.array_equal(image, before)


# --- Hand side report ------------------------------------------------------


def test_hand_side_report_flags_an_unmirrored_clip():
    """Signer's left hand on the viewer's right => larger x."""
    frames = [
        FrameLandmarks(left_hand=a_hand(0.75), right_hand=a_hand(0.25))
        for _ in range(10)
    ]
    report = hand_side_report(frames)
    assert report["fraction_as_expected"] == 1.0
    assert report["verdict"] == "consistent with an unmirrored recording"


def test_hand_side_report_flags_a_mirrored_clip():
    frames = [
        FrameLandmarks(left_hand=a_hand(0.25), right_hand=a_hand(0.75))
        for _ in range(10)
    ]
    report = hand_side_report(frames)
    assert report["fraction_as_expected"] == 0.0
    assert "MIRRORED" in report["verdict"]


def test_hand_side_report_is_honest_about_ambiguity():
    """Hands crossing the midline must not be reported as a swap.

    Signs genuinely cross the body. Reporting that as a mirrored recording would
    train the operator to ignore the check, which is worse than not having it.
    """
    frames = [
        FrameLandmarks(left_hand=a_hand(0.75), right_hand=a_hand(0.25))
        for _ in range(5)
    ] + [
        FrameLandmarks(left_hand=a_hand(0.25), right_hand=a_hand(0.75))
        for _ in range(5)
    ]
    assert "inconsistent" in hand_side_report(frames)["verdict"]


def test_hand_side_report_ignores_frames_with_one_hand():
    frames = [
        FrameLandmarks(left_hand=a_hand(0.75)),
        FrameLandmarks(right_hand=a_hand(0.25)),
        FrameLandmarks(left_hand=a_hand(0.75), right_hand=a_hand(0.25)),
    ]
    assert hand_side_report(frames)["frames_with_both_hands"] == 1


def test_hand_side_report_handles_no_usable_frames():
    assert hand_side_report([FrameLandmarks()])["frames_with_both_hands"] == 0


# --- Label placement -------------------------------------------------------


def test_radial_offsets_point_away_from_the_centroid():
    pixels = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=np.int32)
    order = (0, 1, 2, 3)
    offsets = radial_offsets(pixels, order, 5.0)
    centroid = pixels.mean(axis=0)
    for point, offset in zip(pixels, offsets):
        assert float(np.dot(offset, point - centroid)) > 0


def test_radial_offsets_have_the_requested_magnitude():
    pixels = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]], dtype=np.int32)
    offsets = radial_offsets(pixels, (0, 1, 2, 3), 7.0)
    assert np.allclose(np.linalg.norm(offsets, axis=1), 7.0)


def test_radial_offsets_survive_coincident_points():
    """A degenerate group must not produce NaN offsets.

    Every point landing on the centroid is a zero-length direction vector, and
    a NaN offset propagates into an integer cast that raises rather than
    drawing -- turning a mildly odd frame into a crash mid-render.
    """
    pixels = np.full((4, 2), 5, dtype=np.int32)
    offsets = radial_offsets(pixels, (0, 1, 2, 3), 4.0)
    assert np.isfinite(offsets).all()
    assert np.allclose(np.linalg.norm(offsets, axis=1), 4.0)


def test_radial_offsets_separate_the_two_lips():
    """The case the fix was for.

    The lip ring's upper and lower contours run close together, so a fixed
    offset put their labels on top of each other exactly where reading an index
    matters most. Radiating outward must send them in opposite directions.
    """
    pixels = to_pixels(a_face_ring(), 640, 640)
    order = lm.FACE_GROUPS[0][1]  # lips
    offsets = radial_offsets(pixels, order, 6.0)
    # Ring order is lower lip (0-10) then back along the upper lip (11-19), so
    # position 5 is the bottom centre and 15 the top centre -- the two that sit
    # closest together vertically and collided under a fixed offset. Two points
    # on the same side of the mouth both radiate sideways, so picking those
    # would prove nothing.
    bottom_centre = offsets[5]
    top_centre = offsets[15]
    assert float(np.dot(bottom_centre, top_centre)) < 0


def test_labels_are_legible_against_their_own_colour():
    """Regression: the hand's "L" was invisible on the hand.

    A label drawn in its group's colour, on top of that group's skeleton, is
    the same colour as its background. The dark halo is what makes it readable,
    and without it the hand-assignment check -- the whole reason the label
    exists -- silently stops working.
    """
    background = np.full((80, 80, 3), COLOURS["left_hand"], dtype=np.uint8)
    frame = FrameLandmarks(left_hand=np.full((lm.N_HAND_LANDMARKS, 3), 0.5))
    drawn = draw_frame(background, frame, label_hands=True)
    assert (drawn != background).any(), "the label left no readable mark"
