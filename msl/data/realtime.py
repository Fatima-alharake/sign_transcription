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


def run_webcam(
    camera_index: int = 0,
    *,
    mirror: bool = True,
    show_legend: bool = False,
    checkpoint_path: Path | None = None,
    device: str = "auto",
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
        last_predictions = None
        if checkpoint_path is not None:
            from msl.infer.realtime import WlaslSignPredictor

            camera_fps = capture.get(cv2.CAP_PROP_FPS)
            predictor = WlaslSignPredictor(
                checkpoint_path,
                device=device,
                fps=camera_fps,
            )
            frame_history = deque(maxlen=predictor.window_frames)

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
                if predictor is not None and frame_history is not None:
                    frame_history.append(landmarks)
                    if (
                        len(frame_history) == predictor.window_frames
                        and frame_index % predictor.prediction_interval == 0
                    ):
                        last_predictions = predictor.predict(
                            list(frame_history),
                            width=width,
                            height=height,
                        )
                if mirror:
                    bgr = cv2.flip(bgr, 1)
                    landmarks = _mirror_landmarks(landmarks)
                preview = draw_frame(bgr, landmarks)
                if show_legend:
                    preview = draw_legend(preview)

                status = "LIVE LANDMARKS - Q / Esc to quit"
                _put_status(preview, status, 30)
                if predictor is not None:
                    if len(frame_history) < predictor.window_frames:
                        prediction_text = (
                            "ASL: collecting sign window "
                            f"{len(frame_history)}/{predictor.window_frames}"
                        )
                    elif last_predictions:
                        best = last_predictions[0]
                        prediction_text = (
                            f"ASL candidate: {best.gloss} "
                            f"({best.confidence:.0%})"
                        )
                        if len(last_predictions) > 1:
                            next_best = last_predictions[1]
                            prediction_text += (
                                f" | next: {next_best.gloss} "
                                f"({next_best.confidence:.0%})"
                            )
                    else:
                        prediction_text = "ASL: signer not tracked well enough"
                    _put_status(preview, prediction_text, 60)

                cv2.imshow("MSL Live Landmark Detection", preview)
                if _should_quit(cv2.waitKey(1)):
                    break
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
        )
    except (WebcamError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
