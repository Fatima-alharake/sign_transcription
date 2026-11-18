"""Live webcam landmark detection and overlay preview."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from msl.data.extract import FrameLandmarks, LandmarkExtractor
from msl.data.overlay import draw_frame, draw_legend


class WebcamError(RuntimeError):
    """The webcam could not be opened or read."""


def _mirror_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    mirrored = points.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def _mirror_landmarks(frame: FrameLandmarks) -> FrameLandmarks:
    return FrameLandmarks(
        pose=_mirror_points(frame.pose),
        left_hand=_mirror_points(frame.left_hand),
        right_hand=_mirror_points(frame.right_hand),
        face=_mirror_points(frame.face),
    )


def _should_quit(key: int) -> bool:
    return (key & 0xFF) in (ord("q"), ord("Q"), 27)


def _should_clear(key: int) -> bool:
    return (key & 0xFF) in (ord(" "), ord("c"), ord("C"))


def _put_status(image: np.ndarray, text: str, y: int) -> None:
    for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(
            image,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            thickness,
            cv2.LINE_AA,
        )


def _draw_hud(
    image: np.ndarray,
    *,
    is_active: bool,
    status_text: str,
    controls_text: str = "[Space] Clear   [Q] Quit",
) -> None:
    """Draw clean, non-overlapping top HUD pills for state and controls."""
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    status_size = cv2.getTextSize(status_text, font, 0.48, 1)[0]
    pill_w = status_size[0] + 34
    pill_h = 28
    p1 = (12, 10)
    p2 = (min(width - 150, 12 + pill_w), 10 + pill_h)

    sub_left = image[p1[1] : p2[1], p1[0] : p2[0]]
    if sub_left.size > 0:
        dark_left = np.full_like(sub_left, 15, dtype=np.uint8)
        cv2.addWeighted(dark_left, 0.75, sub_left, 0.25, 0, sub_left)
        cv2.rectangle(image, p1, p2, (65, 65, 65), 1, cv2.LINE_AA)

    dot_color = (0, 220, 0) if is_active else (140, 140, 140)
    cv2.circle(image, (24, 10 + pill_h // 2), 4, dot_color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        status_text,
        (35, 10 + round(pill_h * 0.68)),
        font,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    ctrl_size = cv2.getTextSize(controls_text, font, 0.44, 1)[0]
    c_w = ctrl_size[0] + 20
    c_h = 28
    c1 = (max(p2[0] + 10, width - c_w - 12), 10)
    c2 = (width - 12, 10 + c_h)

    if c1[0] < c2[0]:
        sub_right = image[c1[1] : c2[1], c1[0] : c2[0]]
        if sub_right.size > 0:
            dark_right = np.full_like(sub_right, 15, dtype=np.uint8)
            cv2.addWeighted(dark_right, 0.75, sub_right, 0.25, 0, sub_right)
            cv2.rectangle(image, c1, c2, (65, 65, 65), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            controls_text,
            (c1[0] + 10, 10 + round(c_h * 0.68)),
            font,
            0.44,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def _draw_subtitles_banner(
    image: np.ndarray,
    subtitles_text: str,
    *,
    is_active: bool,
) -> None:
    """Draw a dedicated, collision-free closed-caption subtitle banner at the bottom."""
    height, width = image.shape[:2]
    box_h = 56
    box_y1 = height - box_h - 14
    box_y2 = height - 14
    box_x1 = 15
    box_x2 = width - 15

    if box_y1 < 0 or box_x1 >= box_x2:
        return

    sub = image[box_y1:box_y2, box_x1:box_x2]
    dark_overlay = np.full_like(sub, 18, dtype=np.uint8)
    cv2.addWeighted(dark_overlay, 0.82, sub, 0.18, 0, sub)
    cv2.rectangle(
        image,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (75, 75, 75),
        1,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    prefix = "[CC] "
    cv2.putText(
        image,
        prefix,
        (box_x1 + 14, box_y1 + round(box_h * 0.64)),
        font,
        0.58,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    prefix_size = cv2.getTextSize(prefix, font, 0.58, 2)[0]
    text_x = box_x1 + 16 + prefix_size[0]
    text_y = box_y1 + round(box_h * 0.64)

    if subtitles_text:
        cv2.putText(
            image,
            subtitles_text,
            (text_x, text_y),
            font,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        hint = (
            "(Signing detected...)"
            if is_active
            else "(Ready - perform signs to translate)"
        )
        cv2.putText(
            image,
            hint,
            (text_x, text_y),
            font,
            0.48,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )



def run_webcam(
    camera_index: int = 0,
    *,
    mirror: bool = True,
    show_legend: bool = False,
    checkpoint_path: Path | None = None,
    device: str = "auto",
    min_confidence: float = 0.40,
    motion_threshold: float = 0.008,
    cooldown_seconds: float = 1.2,
) -> None:
    """Show the live landmark overlay; press Q or Esc to close the window."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise WebcamError(
            f"Could not open webcam index {camera_index}. "
            "Check camera permissions or try --camera 1."
        )

    started_at = time.monotonic()
    last_timestamp_ms = -1
    frame_index = 0
    try:
        predictor = None
        frame_history = None
        motion_gate = None
        subtitles = None
        last_predictions = None
        if checkpoint_path is not None:
            from msl.infer.realtime import (
                MotionGate,
                SubtitleAccumulator,
                WlaslSignPredictor,
            )

            camera_fps = capture.get(cv2.CAP_PROP_FPS)
            predictor = WlaslSignPredictor(
                checkpoint_path,
                device=device,
                fps=camera_fps,
            )
            frame_history = deque(maxlen=predictor.window_frames)
            motion_gate = MotionGate(motion_threshold=motion_threshold)
            subtitles = SubtitleAccumulator(
                min_confidence=min_confidence,
                cooldown_seconds=cooldown_seconds,
            )

        with LandmarkExtractor() as extractor:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    raise WebcamError(f"Could not read from webcam index {camera_index}.")

                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int((time.monotonic() - started_at) * 1000),
                )
                last_timestamp_ms = timestamp_ms
                landmarks = extractor.detect_rgb_frame(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    timestamp_ms,
                )
                height, width = bgr.shape[:2]
                is_active = False
                if predictor is not None and frame_history is not None and motion_gate is not None:
                    frame_history.append(landmarks)
                    is_active = motion_gate.update(landmarks)
                    if is_active:
                        if (
                            len(frame_history) == predictor.window_frames
                            and frame_index % predictor.prediction_interval == 0
                        ):
                            last_predictions = predictor.predict(
                                list(frame_history),
                                width=width,
                                height=height,
                            )
                            if subtitles is not None:
                                subtitles.update(
                                    last_predictions,
                                    is_active=True,
                                    current_time=time.monotonic(),
                                )
                    else:
                        last_predictions = None
                        if subtitles is not None:
                            subtitles.update(
                                None,
                                is_active=False,
                                current_time=time.monotonic(),
                            )

                if mirror:
                    bgr = cv2.flip(bgr, 1)
                    landmarks = _mirror_landmarks(landmarks)
                preview = draw_frame(bgr, landmarks)
                if show_legend:
                    preview = draw_legend(preview)

                if predictor is not None and motion_gate is not None and subtitles is not None:
                    if is_active:
                        if len(frame_history) < predictor.window_frames:
                            status_text = (
                                f"SIGNING: Collecting {len(frame_history)}/{predictor.window_frames}"
                            )
                        elif last_predictions:
                            best = last_predictions[0]
                            status_text = f"SIGNING: {best.gloss} ({best.confidence:.0%})"
                            if len(last_predictions) > 1:
                                next_best = last_predictions[1]
                                status_text += f" | Next: {next_best.gloss} ({next_best.confidence:.0%})"
                        else:
                            status_text = "SIGNING: Gesture active"
                    else:
                        status_text = "IDLE (Resting hands)"

                    _draw_hud(
                        preview,
                        is_active=is_active,
                        status_text=status_text,
                        controls_text="[Space] Clear   [Q] Quit",
                    )
                    _draw_subtitles_banner(
                        preview,
                        subtitles.get_text(),
                        is_active=is_active,
                    )
                else:
                    _draw_hud(
                        preview,
                        is_active=False,
                        status_text="LIVE LANDMARKS",
                        controls_text="[Q] Quit",
                    )

                cv2.imshow("MSL Live Sign Recognition", preview)
                key = cv2.waitKey(1)
                if _should_quit(key):
                    break
                if _should_clear(key) and subtitles is not None:
                    subtitles.clear()
                frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show live MediaPipe pose, hand, and face landmarks from a webcam."
        )
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="webcam index (default: 0)"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="show the camera image without mirroring it",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="show the landmark colour legend",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="enable WLASL ASL predictions using this trained checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="prediction device (default: auto)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="minimum confidence threshold to commit a sign to subtitles (default: 0.40)",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.008,
        help="hand motion energy threshold for active signing (default: 0.008)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.2,
        help="seconds before the same sign can repeat in subtitles (default: 1.2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_webcam(
            camera_index=args.camera,
            mirror=not args.no_mirror,
            show_legend=args.legend,
            checkpoint_path=args.checkpoint,
            device=args.device,
            min_confidence=args.min_confidence,
            motion_threshold=args.motion_threshold,
            cooldown_seconds=args.cooldown,
        )
    except (WebcamError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
