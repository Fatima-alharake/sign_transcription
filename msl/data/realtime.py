"""Live webcam landmark detection and overlay preview."""

from __future__ import annotations

import argparse
import time

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


def run_webcam(
    camera_index: int = 0,
    *,
    mirror: bool = True,
    show_legend: bool = False,
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
    try:
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
                if mirror:
                    bgr = cv2.flip(bgr, 1)
                    landmarks = _mirror_landmarks(landmarks)
                preview = draw_frame(bgr, landmarks)
                if show_legend:
                    preview = draw_legend(preview)

                status = "LIVE LANDMARKS - Q / Esc to quit"
                cv2.putText(
                    preview,
                    status,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    status,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("MSL Live Landmark Detection", preview)
                if _should_quit(cv2.waitKey(1)):
                    break
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_webcam(
            camera_index=args.camera,
            mirror=not args.no_mirror,
            show_legend=args.legend,
        )
    except WebcamError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
