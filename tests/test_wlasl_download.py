"""Validation logic in the WLASL downloader.

These tests exist because of one specific observed failure: aslsearch.com
answers 200 with an empty body, and the upstream downloader records that as a
success, writes it to {video_id}.mp4, and then trusts it forever because its
resume check is os.path.exists. Every test here is aimed at some version of
"the file is there, so it must be fine".

No network. The one test that would touch it monkeypatches the fetch.
"""

from __future__ import annotations

import numpy as np
import pytest

from msl.data.wlasl_download import (
    CACHED,
    EMPTY,
    HTTP_ERROR,
    LOG_COLUMNS,
    MIN_FRAMES,
    OK,
    UNDECODABLE,
    ClipRequest,
    download_clip,
    inspect_video,
    summarise,
)


def write_video(path, frames: int = 20, size: tuple[int, int] = (64, 48)) -> bool:
    """Write a real decodable clip. Returns False if no codec is available."""
    import cv2

    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
    )
    if not writer.isOpened():
        return False
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()
    return path.exists() and path.stat().st_size > 0


@pytest.fixture
def clip() -> ClipRequest:
    return ClipRequest(
        video_id="12345", gloss="book", gloss_id=0, split="train",
        signer_id=11, source="aslsearch", url="http://example.invalid/book.mp4",
    )


# --- inspect_video ---------------------------------------------------------


def test_missing_file_is_not_ok(tmp_path) -> None:
    probe = inspect_video(tmp_path / "nope.mp4")
    assert not probe.ok and probe.reason == "missing"


def test_zero_byte_file_is_not_ok(tmp_path) -> None:
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    probe = inspect_video(path)
    assert not probe.ok and probe.reason == "zero bytes"


def test_html_error_page_saved_as_mp4_is_rejected(tmp_path) -> None:
    """The exact aslsearch failure: real bytes, nonzero size, plausible name,
    and not video. Existence checks pass; decoding is what catches it."""
    path = tmp_path / "12345.mp4"
    path.write_bytes(b"<!DOCTYPE html><html><body>404 Not Found</body></html>")
    probe = inspect_video(path)
    assert not probe.ok
    assert path.stat().st_size > 0, "the point is that the file is non-empty"


def test_real_video_is_accepted(tmp_path) -> None:
    path = tmp_path / "good.mp4"
    if not write_video(path, frames=20):
        pytest.skip("no mp4v encoder available in this OpenCV build")
    probe = inspect_video(path)
    assert probe.ok
    assert probe.frames >= MIN_FRAMES
    assert probe.width == 64 and probe.height == 48


def test_truncated_video_below_the_frame_floor_is_rejected(tmp_path) -> None:
    """A two-frame file is not a sign, whatever it decodes to."""
    path = tmp_path / "short.mp4"
    if not write_video(path, frames=2):
        pytest.skip("no mp4v encoder available in this OpenCV build")
    probe = inspect_video(path)
    assert not probe.ok
    assert "decodable frames" in probe.reason


# --- download_clip ---------------------------------------------------------


def test_empty_body_is_a_failure_not_a_success(tmp_path, monkeypatch, clip) -> None:
    """200 with no body. Upstream treats this as downloaded."""
    monkeypatch.setattr(
        "msl.data.wlasl_download._fetch_bytes", lambda *a, **k: (b"", 200)
    )
    row = download_clip(clip, tmp_path, ytdlp=None)
    assert row["status"] == EMPTY
    assert row["bytes"] == 0
    assert not list(tmp_path.glob("12345.*")), "nothing should be written"


def test_non_video_body_is_recorded_undecodable(tmp_path, monkeypatch, clip) -> None:
    monkeypatch.setattr(
        "msl.data.wlasl_download._fetch_bytes",
        lambda *a, **k: (b"<html>not a video</html>", 200),
    )
    row = download_clip(clip, tmp_path, ytdlp=None)
    assert row["status"] == UNDECODABLE
    assert row["bytes"] > 0


def test_http_error_is_recorded_with_its_code(tmp_path, monkeypatch, clip) -> None:
    import urllib.error

    def raise_403(*a, **k):
        raise urllib.error.HTTPError(clip.url, 403, "Forbidden", {}, None)

    monkeypatch.setattr("msl.data.wlasl_download._fetch_bytes", raise_403)
    row = download_clip(clip, tmp_path, ytdlp=None)
    assert row["status"] == HTTP_ERROR
    assert row["http_code"] == 403


def test_valid_cached_file_is_reused_without_fetching(tmp_path, monkeypatch, clip) -> None:
    if not write_video(tmp_path / "12345.mp4", frames=20):
        pytest.skip("no mp4v encoder available in this OpenCV build")

    def explode(*a, **k):
        raise AssertionError("must not re-fetch a file that still decodes")

    monkeypatch.setattr("msl.data.wlasl_download._fetch_bytes", explode)
    row = download_clip(clip, tmp_path, ytdlp=None)
    assert row["status"] == CACHED
    assert row["frames"] >= MIN_FRAMES


def test_corrupt_cached_file_is_refetched_not_trusted(tmp_path, monkeypatch, clip) -> None:
    """The resume bug. An error page from an earlier run must not be inherited
    just because a file with the right name is on disk."""
    (tmp_path / "12345.mp4").write_bytes(b"<html>404</html>")

    calls: list[str] = []

    def fetch(url, *a, **k):
        calls.append(url)
        return (tmp_path / "_src.mp4").read_bytes(), 200

    if not write_video(tmp_path / "_src.mp4", frames=20):
        pytest.skip("no mp4v encoder available in this OpenCV build")
    monkeypatch.setattr("msl.data.wlasl_download._fetch_bytes", fetch)

    row = download_clip(clip, tmp_path, ytdlp=None)
    assert calls, "a corrupt cached file must trigger a re-fetch"
    assert row["status"] == OK
    assert "refetch" in row["note"]


def test_every_row_fills_the_log_schema(tmp_path, monkeypatch, clip) -> None:
    """A row missing a column silently shifts the CSV for every later row."""
    monkeypatch.setattr(
        "msl.data.wlasl_download._fetch_bytes", lambda *a, **k: (b"", 200)
    )
    row = download_clip(clip, tmp_path, ytdlp=None)
    assert set(LOG_COLUMNS) - set(row) == set()


def test_youtube_without_ytdlp_is_reported_not_silently_skipped(tmp_path) -> None:
    yt = ClipRequest(
        video_id="99", gloss="book", gloss_id=0, split="train", signer_id=1,
        source="youtube", url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    row = download_clip(yt, tmp_path, ytdlp=None)
    assert row["status"] == "ytdlp_missing"


# --- summarise -------------------------------------------------------------


def test_summarise_counts_only_files_that_survived(tmp_path) -> None:
    """Per-class counts must come from disk, never from the metadata."""
    import csv

    log = tmp_path / "log.csv"
    rows = [
        dict(video_id="1", gloss="book", status=OK, split="train", signer_id=1, host="a"),
        dict(video_id="2", gloss="book", status=CACHED, split="test", signer_id=2, host="a"),
        dict(video_id="3", gloss="book", status=HTTP_ERROR, split="train", signer_id=3, host="dead"),
        dict(video_id="4", gloss="drink", status=EMPTY, split="train", signer_id=4, host="dead"),
    ]
    with open(log, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})

    out = summarise(log)
    assert out["attempted"] == 4
    assert out["usable"] == 2
    # "drink" had one instance and it failed: the gloss is gone entirely.
    assert out["glosses_with_any_clip"] == 1
    assert out["failures_by_host"] == {"dead": 2}
    assert "American Sign Language" in out["language"]
