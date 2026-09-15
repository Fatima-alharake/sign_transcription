"""Download the WLASL corpus, validating every file (spec S3a.3).

WLASL is **American Sign Language**. It exists here to pretrain a backbone, and
nothing more: low-level motion features transfer across sign languages,
vocabulary does not. No WLASL gloss is ever an MSL gloss.

WLASL ships URLs, not video, and the URLs rot. The upstream downloader catches
every exception, logs it, and moves on, so it exits successfully having silently
dropped a quarter of the corpus. Worse, one host (aslsearch.com) answers 200
with zero content length, so a naive run writes an error page to
{video_id}.mp4 -- and because the upstream resume check is os.path.exists,
every later run trusts that file forever.

This module is built against exactly those failures:

* a response with no body is a failure, not a success;
* every file must **decode as video** before it counts, not merely exist;
* resume re-validates, so a corrupt file from an earlier run is re-fetched
  rather than inherited;
* every clip's outcome is written to a CSV as it happens, so the failures are
  visible and countable instead of scrolling past in a log.

The per-class counts that matter are the ones computed from that CSV after the
fact -- never the metadata counts, which describe a corpus that no longer
fully exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from msl.config import DataConfig

WLASL_RAW_BASE = "https://raw.githubusercontent.com/dxli94/WLASL/master"

# Metadata fetched on demand. Small, and versioned upstream.
METADATA_FILES: dict[str, str] = {
    "WLASL_v0.3.json": "start_kit/WLASL_v0.3.json",
    "wlasl_class_list.txt": "code/I3D/preprocess/wlasl_class_list.txt",
    "nslt_100.json": "code/I3D/preprocess/nslt_100.json",
    "nslt_300.json": "code/I3D/preprocess/nslt_300.json",
    "nslt_1000.json": "code/I3D/preprocess/nslt_1000.json",
    "nslt_2000.json": "code/I3D/preprocess/nslt_2000.json",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# A clip shorter than this is not a sign; it is an error page that happened to
# decode, or a truncated download.
MIN_FRAMES = 4

# Cap the decode used for validation. Dictionary clips are 2-3s; a stray
# feature-length YouTube result should not cost minutes to reject.
MAX_PROBE_FRAMES = 900

# Outcome vocabulary. Plain strings so the CSV is readable without this module,
# and deliberately distinct so failures can be counted by kind.
OK = "ok"                        # downloaded and decodes
CACHED = "cached"                # already on disk and still decodes
HTTP_ERROR = "http_error"        # server said no (404/403/...)
EMPTY = "empty"                  # 200 with no body -- the aslsearch failure
UNDECODABLE = "undecodable"      # bytes arrived but are not video
NETWORK_ERROR = "network_error"  # timeout, DNS, TLS
YTDLP_FAILED = "ytdlp_failed"    # yt-dlp ran and could not fetch it
NO_YTDLP = "ytdlp_missing"       # yt-dlp not installed

USABLE = (OK, CACHED)

LOG_COLUMNS = (
    "video_id", "gloss", "gloss_id", "split", "signer_id", "source", "host",
    "status", "http_code", "bytes", "frames", "width", "height", "fps", "url", "note",
)


class MetadataMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClipRequest:
    video_id: str
    gloss: str
    gloss_id: int
    split: str
    signer_id: int
    source: str
    url: str

    @property
    def host(self) -> str:
        from urllib.parse import urlparse

        return urlparse(self.url).netloc

    @property
    def is_youtube(self) -> bool:
        return "youtube" in self.url or "youtu.be" in self.url


@dataclass(frozen=True)
class VideoProbe:
    """Result of actually decoding a file. `ok` is the only thing that counts."""

    ok: bool
    frames: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    reason: str = ""


def inspect_video(path: Path, max_frames: int = MAX_PROBE_FRAMES) -> VideoProbe:
    """Decode `path` far enough to prove it is really video.

    Existence proves nothing: an HTML error page saved as .mp4 exists, has
    nonzero size, and satisfies every check short of decoding it.
    """
    import cv2

    path = Path(path)
    if not path.exists():
        return VideoProbe(False, reason="missing")
    if path.stat().st_size == 0:
        return VideoProbe(False, reason="zero bytes")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return VideoProbe(False, reason="cannot open")
    frames = width = height = 0
    fps = 0.0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        while frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frames == 0:
                height, width = frame.shape[:2]
            frames += 1
    finally:
        capture.release()

    if frames < MIN_FRAMES:
        return VideoProbe(
            False, frames, width, height, fps, f"only {frames} decodable frames"
        )
    return VideoProbe(True, frames, width, height, fps)


def ensure_metadata(meta_dir: Path) -> Path:
    """Fetch the WLASL metadata files if absent. Small; not the corpus."""
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name, rel in METADATA_FILES.items():
        target = meta_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{WLASL_RAW_BASE}/{rel}"
        print(f"fetching metadata: {name}")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if not payload:
            raise MetadataMissingError(f"empty metadata response for {url}")
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(target)
    return meta_dir


def load_subset(meta_dir: Path, subset: int) -> list[ClipRequest]:
    """The official WLASL{subset} clip list, joined to its per-instance metadata.

    Keyed off the shipped nslt_{subset}.json rather than a top-K reconstruction:
    that file carries the canonical class ids the published baselines use.
    """
    meta_dir = Path(meta_dir)
    nslt_path = meta_dir / f"nslt_{subset}.json"
    if not nslt_path.exists():
        raise MetadataMissingError(f"{nslt_path} not found; run ensure_metadata first")

    nslt = json.loads(nslt_path.read_text(encoding="utf-8"))
    class_ids = {vid: entry["action"][0] for vid, entry in nslt.items()}

    full = json.loads((meta_dir / "WLASL_v0.3.json").read_text(encoding="utf-8"))
    requests: list[ClipRequest] = []
    for entry in full:
        for inst in entry["instances"]:
            vid = inst["video_id"]
            if vid not in class_ids:
                continue
            requests.append(
                ClipRequest(
                    video_id=vid,
                    gloss=entry["gloss"],
                    gloss_id=class_ids[vid],
                    split=inst["split"],
                    signer_id=inst["signer_id"],
                    source=inst.get("source", ""),
                    url=inst["url"],
                )
            )
    return requests


def _existing_file(video_dir: Path, video_id: str) -> Path | None:
    """Any already-downloaded file for this clip, whatever extension it got."""
    for candidate in sorted(Path(video_dir).glob(f"{video_id}.*")):
        if candidate.suffix != ".tmp":
            return candidate
    return None


def _fetch_bytes(url: str, referer: str = "", timeout: int = 45) -> tuple[bytes, int]:
    """GET `url`, returning (body, http_code). Raises on transport failure."""
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, None, headers)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read(), int(getattr(response, "status", 200) or 200)


def _resolve_ytdlp() -> str | None:
    found = shutil.which("yt-dlp")
    if found:
        return found
    # Installed into the project venv rather than onto PATH.
    for name in ("yt-dlp.exe", "yt-dlp"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    return None


def _download_youtube(
    clip: ClipRequest, video_dir: Path, ytdlp: str, timeout: int = 180
) -> tuple[int, str]:
    template = str(Path(video_dir) / f"{clip.video_id}.%(ext)s")
    result = subprocess.run(
        [ytdlp, clip.url, "-o", template, "--no-playlist", "--no-warnings",
         "--quiet", "-f", "mp4/best"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = (result.stderr or "").strip().splitlines()
    return result.returncode, (tail[-1] if tail else "")


def download_clip(clip: ClipRequest, video_dir: Path, ytdlp: str | None) -> dict:
    """Fetch one clip and validate it. Returns a log row."""
    video_dir = Path(video_dir)
    row: dict = {
        "video_id": clip.video_id, "gloss": clip.gloss, "gloss_id": clip.gloss_id,
        "split": clip.split, "signer_id": clip.signer_id, "source": clip.source,
        "host": clip.host, "url": clip.url, "http_code": "", "bytes": 0,
        "frames": 0, "width": 0, "height": 0, "fps": 0.0, "note": "",
    }

    # Resume, but verify. An earlier run may have written an error page.
    existing = _existing_file(video_dir, clip.video_id)
    if existing is not None:
        probe = inspect_video(existing)
        if probe.ok:
            row.update(
                status=CACHED, bytes=existing.stat().st_size, frames=probe.frames,
                width=probe.width, height=probe.height, fps=round(probe.fps, 2),
            )
            return row
        existing.unlink(missing_ok=True)
        row["note"] = f"refetch: cached file {probe.reason}"

    if clip.is_youtube:
        if ytdlp is None:
            row.update(status=NO_YTDLP, note="yt-dlp not installed")
            return row
        try:
            code, tail = _download_youtube(clip, video_dir, ytdlp)
        except subprocess.TimeoutExpired:
            row.update(status=NETWORK_ERROR, note="yt-dlp timeout")
            return row
        produced = _existing_file(video_dir, clip.video_id)
        if code != 0 or produced is None:
            row.update(status=YTDLP_FAILED, note=(tail or f"exit {code}")[:160])
            return row
        probe = inspect_video(produced)
        row.update(
            bytes=produced.stat().st_size, frames=probe.frames, width=probe.width,
            height=probe.height, fps=round(probe.fps, 2),
        )
        if not probe.ok:
            produced.unlink(missing_ok=True)
            row.update(status=UNDECODABLE, note=probe.reason)
            return row
        row["status"] = OK
        return row

    referer = (
        "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi" if "aslpro" in clip.url else ""
    )
    suffix = ".swf" if "aslpro" in clip.url else ".mp4"
    try:
        payload, code = _fetch_bytes(clip.url, referer=referer)
        row["http_code"] = code
    except urllib.error.HTTPError as exc:
        row.update(status=HTTP_ERROR, http_code=exc.code, note=str(exc.reason)[:160])
        return row
    except Exception as exc:
        row.update(status=NETWORK_ERROR, note=f"{type(exc).__name__}: {exc}"[:160])
        return row

    row["bytes"] = len(payload)
    if not payload:
        # 200 with no body. The upstream downloader calls this success.
        row.update(status=EMPTY, note="200 with empty body")
        return row

    destination = video_dir / f"{clip.video_id}{suffix}"
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(destination)

    probe = inspect_video(destination)
    row.update(
        frames=probe.frames, width=probe.width, height=probe.height,
        fps=round(probe.fps, 2),
    )
    if not probe.ok:
        # .swf lands here: Flash is not decodable by OpenCV. Kept on disk so a
        # later swf2mp4 pass can recover it rather than re-downloading.
        row.update(status=UNDECODABLE, note=probe.reason)
        return row
    row["status"] = OK
    return row


def _append_row(log_path: Path, row: dict, write_header: bool) -> None:
    """Append one outcome immediately, so a crash keeps everything before it."""
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
        handle.flush()


def download_subset(
    subset: int = 100,
    data: DataConfig | None = None,
    limit: int | None = None,
    pause: tuple[float, float] = (0.5, 1.5),
) -> Path:
    """Download WLASL{subset}, writing a per-clip outcome log. Returns its path."""
    data = data or DataConfig()
    root = Path(data.data_root) / "wlasl"
    meta_dir = ensure_metadata(root / "metadata")
    video_dir = root / "raw_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / f"download_log_{subset}.csv"

    clips = load_subset(meta_dir, subset)
    if limit:
        clips = clips[:limit]
    ytdlp = _resolve_ytdlp()
    print(f"WLASL{subset} (AMERICAN SIGN LANGUAGE): {len(clips)} clips in metadata")
    print(f"videos  -> {video_dir}")
    print(f"log     -> {log_path}")
    print(f"yt-dlp  : {ytdlp or 'NOT FOUND -- YouTube clips will be skipped'}")

    write_header = not log_path.exists()
    counts: dict[str, int] = {}
    for index, clip in enumerate(clips, 1):
        row = download_clip(clip, video_dir, ytdlp)
        _append_row(log_path, row, write_header)
        write_header = False
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if index % 25 == 0 or index == len(clips):
            good = counts.get(OK, 0) + counts.get(CACHED, 0)
            others = " ".join(
                f"{k}={v}" for k, v in sorted(counts.items()) if k not in USABLE
            )
            print(f"  [{index}/{len(clips)}] usable {good}  {others}", flush=True)
        if row["status"] not in (CACHED, NO_YTDLP):
            time.sleep(random.uniform(*pause))
    return log_path


def summarise(log_path: Path) -> dict:
    """Per-class counts from what actually landed on disk, not from metadata."""
    import collections

    with open(log_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [r for r in rows if r["status"] in USABLE]
    per_class = collections.Counter(r["gloss"] for r in usable)
    counts = sorted(per_class.values())
    by_host = collections.Counter(
        r["host"] for r in rows if r["status"] not in USABLE
    )
    return {
        "language": "American Sign Language (WLASL)",
        "attempted": len(rows),
        "usable": len(usable),
        "usable_fraction": round(len(usable) / len(rows), 3) if rows else 0.0,
        "by_status": dict(collections.Counter(r["status"] for r in rows)),
        "failures_by_host": dict(by_host.most_common(8)),
        "glosses_with_any_clip": len(per_class),
        "per_class_min": counts[0] if counts else 0,
        "per_class_median": counts[len(counts) // 2] if counts else 0,
        "per_class_max": counts[-1] if counts else 0,
        "per_official_split": dict(collections.Counter(r["split"] for r in usable)),
        "distinct_signers": len({r["signer_id"] for r in usable}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download WLASL (American Sign Language) with per-clip validation"
    )
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--limit", type=int, default=None, help="first N clips only")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args(argv)

    data = DataConfig(data_root=args.data_root) if args.data_root else DataConfig()
    root = Path(data.data_root) / "wlasl"
    log_path = root / f"download_log_{args.subset}.csv"

    if not args.summarise_only:
        log_path = download_subset(args.subset, data, limit=args.limit)

    if not log_path.exists():
        raise SystemExit(f"no log at {log_path}")
    print(f"\n=== WLASL{args.subset} - AMERICAN SIGN LANGUAGE - actual, not metadata ===")
    print(json.dumps(summarise(log_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
