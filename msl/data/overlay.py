"""Landmark overlay renderer (spec S10.1, and the live preview of S4a).

Milestone 1 ends with a demand that is easy to under-read: "visualise extracted
landmarks overlaid on the video to prove extraction is correct -- this is the
only way to catch a silently broken MediaPipe setup." The failures this is aimed
at do not raise. A wrong face index still returns a float. A left/right hand
swap still returns two hands. Extraction reports a healthy detection rate either
way, and a classifier trained on the result still converges. Looking is the
check.

WHAT THIS OVERLAY CANNOT CHECK
------------------------------
Aspect correction. This renderer maps MediaPipe's normalised coordinates
straight back to pixels -- x*width, y*height -- which is the inverse of what
MediaPipe did, so the drawing lands on the signer whether or not the feature
pipeline's aspect correction (spec S4.0) is right, and whether or not it is
applied at all. That is not a limitation to work around; drawing anything else
here would be wrong, because the point is to verify the *raw* landmark stream.
Aspect correction is guarded by tests asserting that the same physical pose at
1:1 and 16:9 yields identical features. Do not try to make the overlay do it.

So: this module verifies index semantics, hand assignment and tracking quality.
It is silent on normalisation, on purpose.

WHAT IT IS AIMED AT
-------------------
1. Face index semantics. The subset in landmarks.py is pinned to MediaPipe's own
   topology tables by tests/test_topology.py, which proves our indices are what
   MediaPipe *declares* lips and eyebrows to be. It does not prove the bundled
   tflite emits its 478 points in that order. Rendering the lip ring on a real
   mouth closes that gap, which is why `face_panel` crops and magnifies: 34 face
   points on a full-body 1080p frame are a few pixels apart and prove nothing at
   full size.
2. Hand assignment. HolisticLandmarker assigns left/right from the body model.
   Left and right are drawn in different colours and labelled at the wrist, and
   `hand_side_report` measures the assignment numerically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from msl.config import DataConfig
from msl.data import landmarks as lm
from msl.data.extract import FrameLandmarks

# BGR, because OpenCV. Chosen to stay distinguishable against skin and most
# backgrounds, and to keep left/right far apart in hue rather than in lightness
# -- a left/right swap must be obvious at a glance, not a shade difference.
COLOURS: dict[str, tuple[int, int, int]] = {
    "pose": (200, 200, 200),           # neutral grey: scaffolding, not subject
    "left_hand": (80, 200, 255),       # amber
    "right_hand": (255, 160, 60),      # blue
    "lips": (120, 120, 255),           # red
    "left_eyebrow": (120, 255, 120),   # green
    "right_eyebrow": (255, 255, 120),  # cyan
    "eye_corners": (255, 120, 255),    # magenta
}

# Face groups drawn as points with no contour between them.
_POINT_ONLY = lm.FACE_POINT_ONLY_GROUPS


@dataclass(frozen=True)
class DrawStyle:
    """Sizes in pixels. Scaled from frame height by `DrawStyle.for_frame`."""

    point_radius: int = 2
    line_thickness: int = 1
    font_scale: float = 0.5
    font_thickness: int = 1

    @classmethod
    def for_frame(cls, height: int) -> "DrawStyle":
        """Pick sizes that stay legible from 480p to 4K.

        A fixed 2px dot vanishes on a 2160-tall frame and swamps the face on a
        240-tall one, and either way the overlay stops being readable, which
        defeats its only purpose.
        """
        scale = max(height / 720.0, 0.4)
        return cls(
            point_radius=max(1, round(2 * scale)),
            line_thickness=max(1, round(1.5 * scale)),
            font_scale=max(0.35, 0.5 * scale),
            font_thickness=max(1, round(1 * scale)),
        )


def to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (x, y, z) -> integer pixel (x, y).

    Deliberately NOT aspect-corrected; see the module docstring. z is dropped
    rather than used for depth shading: MediaPipe's z is unreliable enough
    (spec S4.3) that encoding it visually would invite reading meaning into it.
    """
    xy = np.asarray(points, dtype=np.float64)[:, :2].copy()
    xy[:, 0] *= width
    xy[:, 1] *= height
    return np.rint(xy).astype(np.int32)


def _draw_chain(
    image: np.ndarray,
    pixels: np.ndarray,
    order: tuple[int, ...],
    colour: tuple[int, int, int],
    style: DrawStyle,
    *,
    closed: bool,
) -> None:
    if len(order) < 2:
        return
    polyline = pixels[list(order)].reshape(-1, 1, 2)
    cv2.polylines(image, [polyline], closed, colour, style.line_thickness, cv2.LINE_AA)


def _draw_edges(
    image: np.ndarray,
    pixels: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    colour: tuple[int, int, int],
    style: DrawStyle,
) -> None:
    for a, b in edges:
        cv2.line(
            image,
            tuple(pixels[a]),
            tuple(pixels[b]),
            colour,
            style.line_thickness,
            cv2.LINE_AA,
        )


def _draw_points(
    image: np.ndarray,
    pixels: np.ndarray,
    colour: tuple[int, int, int],
    style: DrawStyle,
    order: tuple[int, ...] | None = None,
) -> None:
    for index in order if order is not None else range(len(pixels)):
        cv2.circle(
            image, tuple(pixels[index]), style.point_radius, colour, -1, cv2.LINE_AA
        )


def _put_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    style: DrawStyle,
    *,
    scale: float = 1.0,
) -> None:
    """Draw text with a dark halo so it reads against any background.

    Without the halo a label drawn in its own group's colour disappears into
    the skeleton it labels -- the hand's "L" was invisible against the hand.
    Skin, dark clothing and a bright window all defeat plain text too, and an
    overlay whose labels cannot be read is an overlay that verifies nothing.
    """
    font_scale = style.font_scale * scale
    thickness = style.font_thickness
    for halo_colour, halo_thickness in (
        ((0, 0, 0), thickness + 2),
        (colour, thickness),
    ):
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            halo_colour,
            halo_thickness,
            cv2.LINE_AA,
        )


def radial_offsets(
    pixels: np.ndarray, order: tuple[int, ...], distance: float
) -> np.ndarray:
    """Per-point label offsets pointing away from the group's centroid.

    A fixed up-and-right offset stacks labels on top of each other wherever a
    contour curves back on itself -- on the lip ring the upper and lower lips
    run within a few pixels of each other and their labels overlap into an
    unreadable smear. Pushing each label outward from the centre separates the
    two lips, which is exactly where confirming an index matters most.

    Points at the centroid get a fixed fallback direction rather than a
    division by zero.
    """
    points = pixels[list(order)].astype(np.float64)
    centroid = points.mean(axis=0)
    vectors = points - centroid
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = np.divide(
        vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-9
    )
    # Anything sitting on the centroid is pushed up-right, matching the old
    # behaviour rather than landing exactly on its own dot.
    directions[norms[:, 0] <= 1e-9] = (0.7071, -0.7071)
    return directions * distance


def draw_frame(
    image: np.ndarray,
    frame: FrameLandmarks,
    *,
    style: DrawStyle | None = None,
    label_hands: bool = True,
) -> np.ndarray:
    """Draw one frame's kept landmarks over a BGR image, returning a new array.

    Never mutates `image`: the caller usually still wants the clean frame, and a
    renderer that quietly annotates its input turns "write both" into a bug.

    Only the landmarks we actually feed the model are drawn. Drawing all 478
    face points would look more impressive and would hide precisely the defect
    this exists to find -- a full mesh renders as a plausible face even when our
    34-point subset addresses the wrong vertices.
    """
    canvas = np.array(image, dtype=np.uint8, copy=True)
    height, width = canvas.shape[:2]
    style = style or DrawStyle.for_frame(height)

    # Pose first, so hands and face land on top of the scaffolding.
    if frame.pose is not None:
        pixels = to_pixels(frame.pose, width, height)
        _draw_edges(canvas, pixels, lm.POSE_CONNECTIONS, COLOURS["pose"], style)
        _draw_points(canvas, pixels, COLOURS["pose"], style)

    if frame.face is not None:
        pixels = to_pixels(frame.face, width, height)
        for name, order, closed in lm.FACE_GROUPS:
            colour = COLOURS[name]
            if name not in _POINT_ONLY:
                _draw_chain(canvas, pixels, order, colour, style, closed=closed)
            _draw_points(canvas, pixels, colour, style, order)

    for side in ("left_hand", "right_hand"):
        points = getattr(frame, side)
        if points is None:
            continue
        pixels = to_pixels(points, width, height)
        colour = COLOURS[side]
        _draw_edges(canvas, pixels, lm.HAND_CONNECTIONS, colour, style)
        _draw_points(canvas, pixels, colour, style)
        if label_hands:
            # Below and right of the wrist (hand landmark 0). The fingers run
            # away from the wrist, so downward is the one direction reliably
            # clear of the skeleton this label belongs to.
            x, y = pixels[0]
            _put_label(
                canvas,
                "L" if side == "left_hand" else "R",
                (
                    int(x) + 5 * style.point_radius,
                    int(y) + 8 * style.point_radius,
                ),
                colour,
                style,
                scale=1.4,
            )
    return canvas


def draw_legend(image: np.ndarray, *, extra_lines: tuple[str, ...] = ()) -> np.ndarray:
    """Colour key, plus the caveat that stops the overlay being over-read.

    The "pixel space" line is not decoration. An overlay that lands perfectly on
    the signer reads as proof the whole pipeline is right, and it is not -- it
    says nothing about normalisation.
    """
    canvas = np.array(image, dtype=np.uint8, copy=True)
    height = canvas.shape[0]
    style = DrawStyle.for_frame(height)
    pad = max(6, round(height / 90))
    step = max(12, round(height / 40))

    entries = [
        ("pose (upper body)", COLOURS["pose"]),
        ("left hand (signer's)", COLOURS["left_hand"]),
        ("right hand (signer's)", COLOURS["right_hand"]),
        ("lips outer ring", COLOURS["lips"]),
        ("left eyebrow", COLOURS["left_eyebrow"]),
        ("right eyebrow", COLOURS["right_eyebrow"]),
        ("eye corners", COLOURS["eye_corners"]),
    ]
    lines = list(extra_lines) + ["pixel space - does NOT check aspect correction"]

    box_height = step * (len(entries) + len(lines)) + pad
    box_width = max(
        round(canvas.shape[1] * 0.34),
        round(style.font_scale * 22 * max(len(t) for t, _ in entries)),
    )
    panel = canvas[pad : pad + box_height, pad : pad + box_width]
    if panel.size:
        # Darken rather than fill: the frame underneath stays partly visible, so
        # the legend cannot hide a tracking failure in the corner it covers.
        panel[:] = (panel * 0.35).astype(np.uint8)

    y = pad + step
    for text, colour in entries:
        cv2.circle(
            canvas,
            (pad + step // 2, y - step // 4),
            style.point_radius + 1,
            colour,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (pad + step, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            colour,
            style.font_thickness,
            cv2.LINE_AA,
        )
        y += step
    for text in lines:
        cv2.putText(
            canvas,
            text,
            (pad + step // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale * 0.9,
            (180, 180, 180),
            style.font_thickness,
            cv2.LINE_AA,
        )
        y += step
    return canvas


def face_bounds(
    face: np.ndarray, width: int, height: int, *, margin: float = 0.6
) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) pixel box around the face subset, with margin.

    The margin is generous because the kept subset covers only lips, eyebrows
    and eye corners -- a tight box would crop the forehead and jaw, and the
    surrounding face is exactly the context needed to judge whether the lip ring
    sits on the lips.
    """
    pixels = to_pixels(face, width, height)
    x0, y0 = pixels.min(axis=0)
    x1, y1 = pixels.max(axis=0)
    pad_x = max(1, round((x1 - x0) * margin))
    pad_y = max(1, round((y1 - y0) * margin))
    return (
        int(max(0, x0 - pad_x)),
        int(max(0, y0 - pad_y)),
        int(min(width, x1 + pad_x)),
        int(min(height, y1 + pad_y)),
    )


def face_panel(
    image: np.ndarray,
    frame: FrameLandmarks,
    *,
    size: int = 640,
    label_indices: bool = True,
) -> np.ndarray | None:
    """Magnified face crop with the kept points drawn and optionally labelled.

    This is the artifact that clears the PROVISIONAL status on the face indices.
    Returns None when no face was detected, rather than an empty image, so the
    caller can say so instead of writing a black PNG that looks like a bug.

    Labels are raw mesh indices. Seeing "61" sit on a lip corner is the specific
    confirmation wanted -- the group colour alone would still look right if the
    whole subset were offset by a constant.
    """
    if frame.face is None:
        return None
    height, width = image.shape[:2]
    x0, y0, x1, y1 = face_bounds(frame.face, width, height)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0:y1, x0:x1]
    scale = size / max(crop.shape[0], crop.shape[1])
    panel = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    panel_h, panel_w = panel.shape[:2]
    style = DrawStyle(
        point_radius=max(2, round(size / 200)),
        line_thickness=max(1, round(size / 400)),
        # Large enough that a 3-digit index stays readable through its halo;
        # the panel exists to be read, not to look tidy.
        font_scale=size / 1400,
        font_thickness=1,
    )

    # Re-project into crop-local pixels: full-frame pixels, minus the crop
    # origin, times the magnification.
    pixels = to_pixels(frame.face, width, height).astype(np.float64)
    pixels[:, 0] = (pixels[:, 0] - x0) * (panel_w / crop.shape[1])
    pixels[:, 1] = (pixels[:, 1] - y0) * (panel_h / crop.shape[0])
    pixels = np.rint(pixels).astype(np.int32)

    for name, order, closed in lm.FACE_GROUPS:
        colour = COLOURS[name]
        if name not in _POINT_ONLY:
            _draw_chain(panel, pixels, order, colour, style, closed=closed)
        _draw_points(panel, pixels, colour, style, order)
        if label_indices:
            offsets = radial_offsets(pixels, order, style.point_radius * 3.5)
            for position, offset in zip(order, offsets):
                _put_label(
                    panel,
                    str(lm.FACE_INDICES[position]),
                    (
                        int(pixels[position][0] + offset[0]),
                        int(pixels[position][1] + offset[1]),
                    ),
                    colour,
                    style,
                )
    return panel


def hand_side_report(frames: list[FrameLandmarks]) -> dict[str, object]:
    """Numeric check on HolisticLandmarker's left/right hand assignment.

    Over frames where both hands are visible, compare their mean x in normalised
    image coordinates. On an unmirrored frame of a signer facing the camera, the
    signer's left hand appears on the viewer's RIGHT, so mean x of `left_hand`
    should be the LARGER value.

    Reported, never asserted. A mirrored recording inverts it legitimately, and
    many webcam stacks mirror the live preview while writing an unmirrored file
    -- so a violation means "look at the clip", not "the extractor is broken".
    Spec S4a records `mirrored` per clip for exactly this reason.
    """
    both = [f for f in frames if f.left_hand is not None and f.right_hand is not None]
    if not both:
        return {"frames_with_both_hands": 0, "verdict": "no frames with both hands"}

    left_x = float(np.mean([f.left_hand[:, 0].mean() for f in both]))
    right_x = float(np.mean([f.right_hand[:, 0].mean() for f in both]))
    agree = sum(1 for f in both if f.left_hand[:, 0].mean() > f.right_hand[:, 0].mean())
    fraction = agree / len(both)
    return {
        "frames_with_both_hands": len(both),
        "mean_x_left_hand": left_x,
        "mean_x_right_hand": right_x,
        "fraction_as_expected": fraction,
        "verdict": (
            "consistent with an unmirrored recording"
            if fraction >= 0.8
            else "consistent with a MIRRORED recording"
            if fraction <= 0.2
            else "inconsistent - hands may cross the midline, inspect the video"
        ),
    }


# --- CLI -------------------------------------------------------------------


def render_video(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    data: DataConfig | None = None,
    max_frames: int | None = None,
    panel_every: int = 0,
    write_video: bool = True,
    label_indices: bool = True,
) -> dict[str, object]:
    """Run the extractor over a video and write an annotated copy plus panels."""
    from msl.data.extract import LandmarkExtractor, VideoReadError, probe_video

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, width, height = probe_video(video_path)

    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            str(out_dir / "overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps or 30.0,
            (width, height),
        )
        if not writer.isOpened():
            raise VideoReadError(
                f"could not open a writer for {out_dir / 'overlay.mp4'}"
            )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoReadError(f"could not open {video_path}")

    collected: list[FrameLandmarks] = []
    panels: list[Path] = []
    counts = {"pose": 0, "left_hand": 0, "right_hand": 0, "face": 0}
    index = 0
    try:
        with LandmarkExtractor(data or DataConfig()) as extractor:
            for frame, bgr in _iter_annotated(extractor, capture, fps, max_frames):
                collected.append(frame)
                for key in counts:
                    if getattr(frame, key) is not None:
                        counts[key] += 1

                if writer is not None:
                    writer.write(draw_legend(draw_frame(bgr, frame)))
                if panel_every and index % panel_every == 0:
                    panel = face_panel(bgr, frame, label_indices=label_indices)
                    if panel is not None:
                        path = out_dir / f"face_{index:04d}.png"
                        cv2.imwrite(str(path), panel)
                        panels.append(path)
                index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    total = max(index, 1)
    return {
        "video": str(video_path),
        "frames": index,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "aspect": width / height,
        "detection_rate": {k: v / total for k, v in counts.items()},
        "hand_sides": hand_side_report(collected),
        "overlay_video": str(out_dir / "overlay.mp4") if write_video else None,
        "face_panels": [str(p) for p in panels],
    }


def _iter_annotated(extractor, capture, fps: float, max_frames: int | None):
    """Yield (FrameLandmarks, bgr_frame) pairs.

    Separate from `LandmarkExtractor.frames` because the overlay needs the
    decoded pixels alongside the landmarks, and `frames()` deliberately yields
    only landmarks -- extraction has no business holding whole frames in memory.
    """
    import mediapipe as mp

    index = 0
    while max_frames is None or index < max_frames:
        ok, bgr = capture.read()
        if not ok:
            return
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        result = extractor._landmarker.detect_for_video(
            image, int(index / (fps or 30.0) * 1000)
        )
        yield (
            FrameLandmarks(
                pose=extractor._select(result.pose_landmarks, lm.POSE_INDICES),
                left_hand=extractor._select(result.left_hand_landmarks, None),
                right_hand=extractor._select(result.right_hand_landmarks, None),
                face=extractor._select(result.face_landmarks, lm.FACE_INDICES),
            ),
            bgr,
        )
        index += 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render MediaPipe landmarks over a video to verify extraction "
            "(spec S10.1). Checks index semantics and hand assignment; says "
            "nothing about aspect correction -- see the module docstring."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/overlay"))
    parser.add_argument(
        "--max-frames", type=int, default=None, help="stop after N frames"
    )
    parser.add_argument(
        "--panel-every",
        type=int,
        default=15,
        help="write a magnified face crop every N frames (0 disables)",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--no-index-labels",
        action="store_true",
        help="omit raw mesh index numbers on the face panels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import json

    args = build_arg_parser().parse_args(argv)
    if not args.video.exists():
        raise SystemExit(f"no such video: {args.video}")

    report = render_video(
        args.video,
        args.out,
        max_frames=args.max_frames,
        panel_every=args.panel_every,
        write_video=not args.no_video,
        label_indices=not args.no_index_labels,
    )
    print(json.dumps(report, indent=2))
    if report["frames"] == 0:
        raise SystemExit("decoded zero frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
