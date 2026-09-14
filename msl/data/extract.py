"""Video -> normalised landmark tensors (spec S4).

Two layers, deliberately separated:

* A **pure core** (`normalise_frame`, `pack_clip`) that operates on plain numpy
  arrays and imports no MediaPipe and no OpenCV. All the logic worth testing
  lives here, so the test suite runs with neither a model bundle nor a video
  file present.
* A **MediaPipe layer** (`LandmarkExtractor`, `extract_video`) that turns a video
  into those arrays. Imported lazily, so `import msl.data.extract` stays cheap
  and CI without model assets still exercises the core.

MediaPipe version note (spec S2): the legacy `mediapipe.solutions.holistic` API
is not merely deprecated in MediaPipe 1.0 -- `mediapipe.solutions` does not
exist at all. This module uses the Tasks API `HolisticLandmarker`, which also
assigns left/right hand from the body model rather than leaving us to
disambiguate handedness ourselves.

Coordinate conventions, in the order applied:

1. **Aspect correction.** MediaPipe normalises x by frame width and y by frame
   height, so on any non-square video the two axes carry different units and a
   circle traced by the hand comes out an ellipse. x is multiplied by
   width/height to put both axes in units of image height. This is invisible in
   a pixel-space overlay -- an overlay renders correctly either way -- so it can
   only be caught by reasoning about the units, not by looking.
2. **Translate** so the mid-shoulder point is the origin.
3. **Scale** by inter-shoulder distance, measured in the image plane only. z is
   unreliable (spec S4.3) and letting it into the *divisor* would corrupt every
   coordinate rather than just the z channel.

Missing hands emit zeros plus a visibility mask (spec S4). The zeros are written
*after* normalisation, never before: a zeroed hand fed through the translate
step would land at -mid_shoulder/scale, a perfectly plausible resting position,
which is precisely the ambiguity the mask exists to resolve.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

from msl.config import DataConfig, FeatureConfig
from msl.data import landmarks as lm

if TYPE_CHECKING:  # pragma: no cover - typing only
    from msl.config import ExperimentConfig

# One-off download; 13.7 MB, not vendored into git.
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

# Bumped when the packing logic changes in a way that invalidates cached .npy.
# v2: the left/right eyebrow index tuples in landmarks.py were inverted relative
# to MediaPipe's own topology tables. Correcting the names swapped those two
# 5-point blocks' positions inside the face block, so any v1 tensor has ten
# points in the wrong slots.
EXTRACTOR_VERSION = 2


class MissingModelAssetError(RuntimeError):
    """The MediaPipe .task bundle is not on disk."""


class VideoReadError(RuntimeError):
    """The video could not be opened or contained no frames."""


@dataclass(frozen=True)
class FrameLandmarks:
    """One frame's raw landmarks, before normalisation.

    Each array is (n_points, 3) in MediaPipe's normalised image coordinates, or
    None when that part was not detected. Keeping this as a plain dataclass is
    what lets the tests build frames by hand.
    """

    pose: np.ndarray | None = None
    left_hand: np.ndarray | None = None
    right_hand: np.ndarray | None = None
    face: np.ndarray | None = None


@dataclass(frozen=True)
class ClipLandmarks:
    """Extraction result for one clip, at the video's native frame count.

    Resampling to a fixed `seq_len` is `dataset.py`'s job (spec S5), not this
    module's -- the cache stores what was actually seen.
    """

    features: np.ndarray  # (T, n_features) float32
    pose_present: np.ndarray  # (T,) bool -- frame normalisable at all
    fps: float
    width: int
    height: int
    use_z: bool

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"features must be 2-D, got {self.features.shape}")
        if len(self.pose_present) != len(self.features):
            raise ValueError("pose_present and features disagree on frame count")

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[0])

    @property
    def aspect(self) -> float:
        """width/height, for the manifest (spec S4).

        Recorded per clip because it is a property of the camera that produced
        the clip, not of the project. A corpus mixing 16:9 and 4:3 sources is
        fine as long as each clip's correction was derived from its own frame
        size; it is silently wrong the moment one constant is assumed for all.
        """
        return self.width / self.height

    @property
    def hand_visibility(self) -> np.ndarray:
        """(T, 2) mask: per frame, whether the left/right hand was detected."""
        return self.features[:, lm.visibility_slice(self.use_z)]

    def detection_rate(self) -> dict[str, float]:
        """Fraction of frames with pose / left hand / right hand detected.

        These are the numbers the capture-time quality gates in spec S4a act on,
        which is why extraction reports them rather than making the caller
        recompute them from the packed vector.
        """
        if self.n_frames == 0:
            return {"pose": 0.0, "left_hand": 0.0, "right_hand": 0.0}
        visibility = self.hand_visibility
        return {
            "pose": float(self.pose_present.mean()),
            "left_hand": float((visibility[:, 0] > 0).mean()),
            "right_hand": float((visibility[:, 1] > 0).mean()),
        }


def _as_points(array: np.ndarray | None, expected: int, name: str) -> np.ndarray | None:
    """Validate a landmark block, returning it as float64 (n, 3)."""
    if array is None:
        return None
    points = np.asarray(array, dtype=np.float64)
    if points.shape != (expected, 3):
        raise ValueError(f"{name} must have shape ({expected}, 3), got {points.shape}")
    return points


def apply_aspect(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rescale x into units of image height, leaving y and z alone.

    Takes the frame's actual pixel dimensions rather than a precomputed
    constant, because the correction must be re-derived from whatever camera or
    corpus produced the frame (spec S4). AUTSL is one aspect ratio and a webcam
    is likely another; carrying a training-time constant into `realtime.py`
    would reintroduce exactly the distortion this corrects, at the point where
    it is hardest to notice.

    Without this, horizontal and vertical distances are not comparable and the
    inter-shoulder divisor -- almost entirely horizontal -- is wrong relative to
    the vertical hand motion it is meant to normalise.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    out = points.copy()
    out[:, 0] *= width / height
    return out


def normalise_frame(
    frame: FrameLandmarks,
    *,
    width: int,
    height: int,
    use_z: bool = True,
    min_shoulder_distance: float = 1e-3,
) -> tuple[np.ndarray, bool]:
    """Pack one frame into its feature vector.

    Returns the vector and whether the frame was normalisable at all. An
    un-normalisable frame (no pose, or shoulders too close together to give a
    trustworthy scale) yields an all-zero vector with both hand-visibility flags
    clear, rather than dividing by ~0 and seeding the cache with infinities.
    """
    coords = lm.coords_per_landmark(use_z)
    vector = np.zeros(lm.feature_dim(use_z), dtype=np.float32)

    pose = _as_points(frame.pose, lm.N_POSE, "pose")
    if pose is None:
        return vector, False

    pose = apply_aspect(pose, width, height)
    left_shoulder = pose[lm.LEFT_SHOULDER_POS]
    right_shoulder = pose[lm.RIGHT_SHOULDER_POS]

    origin = (left_shoulder + right_shoulder) / 2.0
    # Image-plane distance only: z is excluded from the divisor on purpose.
    scale = float(np.linalg.norm((left_shoulder - right_shoulder)[:2]))
    if not np.isfinite(scale) or scale < min_shoulder_distance:
        return vector, False

    blocks: dict[str, np.ndarray | None] = {
        "pose": pose,
        "left_hand": _as_points(frame.left_hand, lm.N_HAND_LANDMARKS, "left_hand"),
        "right_hand": _as_points(frame.right_hand, lm.N_HAND_LANDMARKS, "right_hand"),
        "face": _as_points(frame.face, lm.N_FACE, "face"),
    }

    for name, points in blocks.items():
        if points is None:
            continue  # leave zeros; for hands the mask below says why
        if name != "pose":
            points = apply_aspect(points, width, height)
        centred = (points - origin) / scale
        vector[lm.block_slice(name, use_z)] = centred[:, :coords].reshape(-1)

    visibility = lm.visibility_slice(use_z)
    vector[visibility] = [
        float(blocks["left_hand"] is not None),
        float(blocks["right_hand"] is not None),
    ]
    return vector, True


def pack_clip(
    frames: list[FrameLandmarks],
    *,
    fps: float,
    width: int,
    height: int,
    use_z: bool = True,
    min_shoulder_distance: float = 1e-3,
) -> ClipLandmarks:
    """Normalise every frame of a clip into a (T, n_features) tensor."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")

    vectors = np.zeros((len(frames), lm.feature_dim(use_z)), dtype=np.float32)
    present = np.zeros(len(frames), dtype=bool)
    for i, frame in enumerate(frames):
        vectors[i], present[i] = normalise_frame(
            frame,
            width=width,
            height=height,
            use_z=use_z,
            min_shoulder_distance=min_shoulder_distance,
        )
    return ClipLandmarks(
        features=vectors,
        pose_present=present,
        fps=fps,
        width=width,
        height=height,
        use_z=use_z,
    )


# --- Caching ---------------------------------------------------------------


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, read in chunks so a 4K clip is not slurped."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(path: str | Path, feature: FeatureConfig) -> str:
    """Cache filename stem for a video under a given feature config.

    Keyed on the video's *content* (spec S4) and on the settings that change the
    output shape. Keying on the path alone would serve stale tensors after a
    re-record; keying on content alone would serve a use_z=True tensor to a
    use_z=False run, where the shape mismatch might not surface until training.
    """
    fingerprint = json.dumps(
        {"v": EXTRACTOR_VERSION, "use_z": feature.use_z},
        sort_keys=True,
    )
    suffix = hashlib.sha256(fingerprint.encode()).hexdigest()[:8]
    return f"{file_digest(path)}-{suffix}"


def cache_path(path: str | Path, feature: FeatureConfig, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{cache_key(path, feature)}.npy"


def save_cached(clip: ClipLandmarks, destination: Path) -> Path:
    """Write the feature tensor atomically -- a torn .npy is worse than none."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".npy.tmp")
    # Write through an open handle: np.save() appends ".npy" to any path that
    # does not already end in it, so np.save(temp, ...) would silently produce
    # "clip.npy.tmp.npy" and leave the rename below with nothing to move.
    with open(temp, "wb") as fh:
        np.save(fh, clip.features)
    temp.replace(destination)
    return destination


# --- MediaPipe layer -------------------------------------------------------


def _require_model(task_path: Path) -> Path:
    task_path = Path(task_path)
    if not task_path.exists():
        parent = task_path.parent
        raise MissingModelAssetError(
            f"MediaPipe model bundle not found at {task_path}.\n"
            f"Download it once with:\n"
            f"  mkdir -p {parent} && curl -L -o {task_path} {HOLISTIC_MODEL_URL}\n"
            f"(13.7 MB; deliberately not vendored into git. Set data.assets_dir "
            f"in the config to keep it elsewhere.)"
        )
    return task_path


class LandmarkExtractor:
    """MediaPipe Tasks HolisticLandmarker wrapped for video files.

    Use as a context manager: the underlying graph holds native resources that
    must be closed, and leaking them across a few hundred clips will exhaust
    them.
    """

    def __init__(self, data: DataConfig | None = None) -> None:
        self.data = data or DataConfig()
        self._landmarker = None

    def __enter__(self) -> "LandmarkExtractor":
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
            RunningMode,
        )

        task_path = _require_model(self.data.holistic_task_path)
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.data.min_pose_confidence,
            min_hand_landmarks_confidence=self.data.min_hand_confidence,
            min_face_detection_confidence=self.data.min_face_confidence,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _select(self, detected: list, indices: tuple[int, ...] | None) -> np.ndarray | None:
        """MediaPipe landmark list -> (n, 3) array, or None if nothing detected."""
        if not detected:
            return None
        points = np.array([[p.x, p.y, p.z] for p in detected], dtype=np.float64)
        if indices is None:
            return points
        if max(indices) >= len(points):
            raise ValueError(
                f"landmark index {max(indices)} out of range for {len(points)} "
                "detected points -- the installed MediaPipe topology differs "
                "from the indices in msl/data/landmarks.py"
            )
        return points[list(indices)]

    def frames(self, video_path: str | Path) -> Iterator[FrameLandmarks]:
        """Yield per-frame landmarks. Requires the context manager to be open."""
        import cv2

        if self._landmarker is None:
            raise RuntimeError("use LandmarkExtractor as a context manager")
        import mediapipe as mp

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoReadError(f"could not open {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                )
                # Timestamps must increase monotonically in VIDEO mode.
                result = self._landmarker.detect_for_video(
                    image, int(index / fps * 1000)
                )
                yield FrameLandmarks(
                    pose=self._select(result.pose_landmarks, lm.POSE_INDICES),
                    left_hand=self._select(result.left_hand_landmarks, None),
                    right_hand=self._select(result.right_hand_landmarks, None),
                    face=self._select(result.face_landmarks, lm.FACE_INDICES),
                )
                index += 1
        finally:
            capture.release()


def probe_video(video_path: str | Path) -> tuple[float, int, int]:
    """(fps, width, height) without decoding the whole file."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoReadError(f"could not open {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise VideoReadError(f"{video_path} reports frame size {width}x{height}")
    return fps, width, height


def extract_video(
    video_path: str | Path,
    feature: FeatureConfig | None = None,
    data: DataConfig | None = None,
) -> ClipLandmarks:
    """Run MediaPipe over a video and return normalised landmarks."""
    feature = feature or FeatureConfig()
    data = data or DataConfig()

    fps, width, height = probe_video(video_path)
    with LandmarkExtractor(data) as extractor:
        frames = list(extractor.frames(video_path))
    if not frames:
        raise VideoReadError(f"{video_path} decoded zero frames")

    return pack_clip(
        frames,
        fps=fps,
        width=width,
        height=height,
        use_z=feature.use_z,
        min_shoulder_distance=data.min_shoulder_distance,
    )


def extract_video_cached(
    video_path: str | Path,
    cfg: "ExperimentConfig",
) -> tuple[np.ndarray, bool]:
    """Extract, or reuse the cached .npy if one exists (spec S4).

    Returns the (T, n_features) tensor and whether it came from cache. Only the
    tensor is cached; `pose_present` and the detection rates are capture-time
    quality signals, and re-deriving them means re-running MediaPipe anyway.
    """
    destination = cache_path(video_path, cfg.feature, cfg.data.resolved_cache_dir)
    if destination.exists():
        return np.load(destination), True

    clip = extract_video(video_path, cfg.feature, cfg.data)
    save_cached(clip, destination)
    return clip.features, False
