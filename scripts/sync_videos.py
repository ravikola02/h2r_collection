#!/usr/bin/env python3
"""
Multi-camera video sync tool for DJI Osmo Nano sessions.

EVERY RUN computes the audio-refined sync and writes the sync artifact:

    data/raw/<session>/sync.json

After that, a flag picks which clip(s) to cut from the synced timeline. The
clipping always uses the same frame-exact `create_clip_batch` -- only the
time window differs:

  | Mode              | Trigger              | Output                                              |
  |-------------------|----------------------|-----------------------------------------------------|
  | sync-only         | --info               | sync.json (no video)                                |
  | session overlap   | (no flag)            | data/output/session_overlap/<session>_{role}.mp4    |
  | per-episode clips | --config clips.json  | data/output/episode_clips/<name>_{role}.mp4 *       |
  | single clip       | --clip S E --name N  | data/output/episode_clips/<N>_{role}.mp4 *          |

  * the cut sheet's "output_dir" field (or --output) overrides the subfolder.

The "session_overlap" output is one big clip per camera covering the whole
audio-overlap window -- this is what the episode tagger scrubs to author a
cut sheet. The "episode_clips" output is many small clips, one per cut-sheet
entry -- this is what `convert_to_lerobot.py` consumes downstream.

WHY AUDIO SYNC
--------------
MP4 `creation_time` is whole-second only (e.g. 13:11:22.000000Z), so the true
inter-camera offset is only known to +/-0.5 s -- up to ~30 frames of error.
We seed from metadata, then refine to sub-frame precision by cross-correlating
the 48 kHz audio tracks.

WHY RE-ENCODE
-------------
`ffmpeg -c copy` cuts only at keyframe boundaries (and differently per camera),
so it is NOT frame-accurate. Clips here are re-encoded with an accurate seek so
every camera starts on the same unified-timeline frame.

USAGE
  python3 sync_videos.py                 # sync latest + emit session-overlap clip
  python3 sync_videos.py --all           # same, for every session
  python3 sync_videos.py --info          # sync only, no video output
  python3 sync_videos.py --session "video 2"
  # episode clipping from a cut sheet (typical):
  python3 sync_videos.py --config clips.json
  # ad-hoc one-shot clip (testing):
  python3 sync_videos.py --clip 5.0 9.0 --name episode_001

CONFIG FILE (clips.json)
  {
    "output_dir": "episode_clips",
    "clips": [
      {"start": 5.0, "end": 9.0, "name": "ep001",
       "episode_id": 1, "task_label": "pick cube", "dominant_hand": "right"}
    ]
  }
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import correlate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"

# role -> subdirectory under a session folder
CAMERA_LAYOUT = {
    "head": "top_cam",
    "wrist_left": "left_w_cam",
    "wrist_right": "right_w_cam",
}
REFERENCE_CAMERA = "head"

# audio refinement
AUDIO_SR = 16000
SEARCH_WINDOW_S = 1.5          # +/- around the metadata-seeded offset
# Confidence = peak-to-sidelobe ratio: (peak - mean) / std of the xcorr window.
# A sharp, unambiguous alignment peak sits well above the noise floor; ambient
# audio with no shared transient stays near ~3. Below this -> fall back.
CONFIDENCE_THRESHOLD = 6.0


@dataclass
class Camera:
    name: str
    path: Path
    creation_time: datetime
    creation_time_source: str          # 'format' | 'stream:N' | 'file_mtime'
    duration: float
    fps_num: int
    fps_den: int
    n_frames: int
    resolution: Tuple[int, int]
    rotation: int
    has_audio: bool = True
    metadata_offset: float = 0.0
    audio_offset: Optional[float] = None
    confidence: float = 1.0
    method: str = "metadata"
    offset_in_timeline: float = 0.0

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den


def _parse_iso_creation_time(value: str) -> Optional[datetime]:
    """Parse an ffprobe creation_time string ('...Z' or with tz offset)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_metadata(video_path: str) -> Dict:
    """Extract creation time, duration, exact fps, frame count, resolution, rotation.

    ``creation_time`` is looked up at three layers, in order:
      1. ``format.tags.creation_time`` — what DJI Osmo writes.
      2. ``streams[*].tags.creation_time`` — some encoders only tag streams.
      3. file mtime — last-resort fallback if no metadata timestamp exists.
         Emits a WARN; the mtime can be wrong by minutes if the file was
         copied or moved, in which case the audio xcorr step is doing the
         real alignment work and the metadata seed is just a coarse hint.

    The chosen source is returned as ``creation_time_source`` so callers
    can surface it (it lands in ``sync.json`` for traceability).
    """
    cmd = [
        "ffprobe", "-v", "error", "-of", "json",
        "-show_format", "-show_streams", video_path,
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)

    tags = data["format"].get("tags", {})
    creation_time = _parse_iso_creation_time(tags.get("creation_time", ""))
    creation_time_source = "format" if creation_time is not None else None

    if creation_time is None:
        for s in data.get("streams", []):
            stream_ct = _parse_iso_creation_time(
                s.get("tags", {}).get("creation_time", ""))
            if stream_ct is not None:
                creation_time = stream_ct
                creation_time_source = f"stream:{s.get('index')}"
                break

    if creation_time is None:
        mtime = Path(video_path).stat().st_mtime
        creation_time = datetime.fromtimestamp(mtime, tz=timezone.utc)
        creation_time_source = "file_mtime"
        print(f"WARN  {Path(video_path).name}: no creation_time in MP4 "
              f"metadata; using file mtime as fallback "
              f"({creation_time.isoformat()}). This may be off by minutes "
              f"if the file was copied or moved -- audio xcorr will still "
              f"refine to sub-frame, but a wildly wrong mtime can push the "
              f"true offset outside the +/-1.5s search window.",
              file=sys.stderr)

    duration = float(data["format"]["duration"])

    video_stream = next(
        (s for s in data["streams"]
         if s["codec_type"] == "video" and s.get("width")),
        None,
    )
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    # Whether the file carries a real audio stream. If not, the xcorr-based
    # sync refinement is unusable -- callers should fall back to the metadata
    # offset cleanly without a per-camera WARN cascade.
    has_audio = any(
        s.get("codec_type") == "audio" for s in data.get("streams", [])
    )

    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        fps_num, fps_den = (int(x) for x in fps_str.split("/"))
    else:
        fps_num, fps_den = int(round(float(fps_str))), 1

    n_frames = int(video_stream.get("nb_frames", 0)) or int(
        round(duration * fps_num / fps_den)
    )
    width = video_stream.get("width", 1920)
    height = video_stream.get("height", 1080)

    rotation = 0
    for stream in data["streams"]:
        if stream.get("codec_type") != "video":
            continue
        for side in stream.get("side_data_list", []):
            if side.get("side_data_type") == "Display Matrix":
                rotation = int(side.get("rotation", 0))
        if rotation:
            break

    return {
        "creation_time": creation_time,
        "creation_time_source": creation_time_source,
        "duration": duration,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "n_frames": n_frames,
        "resolution": (width, height),
        "rotation": rotation,
        "has_audio": has_audio,
    }


def discover_sessions(raw_root: Path) -> List[Path]:
    """Return session dirs matching 'video <N>' under raw_root, sorted by N."""
    sessions = []
    for d in raw_root.iterdir():
        if d.is_dir():
            m = re.fullmatch(r"video\s*(\d+)", d.name, re.IGNORECASE)
            if m:
                sessions.append((int(m.group(1)), d))
    return [d for _, d in sorted(sessions)]


def find_camera_files(session_dir: Path,
                       quiet_missing: bool = False) -> Dict[str, Path]:
    """Map each camera role to its single video file (glob, not a fixed name).

    Missing wrist dirs / empty wrist dirs are normal in head-only sessions, so
    callers that handle the head-only case explicitly can pass
    ``quiet_missing=True`` to suppress the per-camera WARN spam.
    """
    found = {}
    for role, subdir in CAMERA_LAYOUT.items():
        cam_dir = session_dir / subdir
        if not cam_dir.is_dir():
            if not quiet_missing:
                print(f"WARN  {role}: missing dir {cam_dir}", file=sys.stderr)
            continue
        vids = sorted(
            p for p in cam_dir.iterdir()
            if p.suffix.lower() in (".mp4", ".mov")
        )
        if not vids:
            if not quiet_missing:
                print(f"WARN  {role}: no video in {cam_dir}", file=sys.stderr)
            continue
        if len(vids) > 1:
            print(f"WARN  {role}: multiple videos, using {vids[0].name}",
                  file=sys.stderr)
        found[role] = vids[0]
    return found


def detect_session_mode(session_dir: Path) -> str:
    """Auto-detect 'head_only' vs 'head_wrist' from the session's camera dirs.

    Head-only when neither wrist subdir has any video file. The head camera
    must be present in both modes; absence is reported by load_session.
    """
    files = find_camera_files(session_dir, quiet_missing=True)
    has_wrist = any(role != REFERENCE_CAMERA for role in files)
    return "head_wrist" if has_wrist else "head_only"


def _decode_audio(path: Path, max_seconds: float) -> np.ndarray:
    """Decode mono float32 PCM of the first `max_seconds` for cross-correlation."""
    cmd = [
        "ffmpeg", "-v", "error", "-t", str(max_seconds), "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def refine_offset_audio(
    ref: Camera, other: Camera, coarse_offset: float
) -> Tuple[float, float]:
    """
    Refine `other`'s offset vs the reference camera via audio cross-correlation.

    Returns (refined_offset_seconds, confidence). `coarse_offset` is the
    metadata estimate of (other.start - ref.start) in the unified timeline.
    """
    # Both clips overlap from `coarse_offset` (ref-time) onward. Decode a window
    # that comfortably covers the overlap plus the +/- search slack.
    span = min(ref.duration - coarse_offset, other.duration)
    span = min(span, 20.0)  # 20 s of audio is plenty for a clean peak
    if span <= 1.0:
        return coarse_offset, 0.0

    ref_audio = _decode_audio(ref.path, coarse_offset + span + SEARCH_WINDOW_S)
    other_audio = _decode_audio(other.path, span + SEARCH_WINDOW_S)

    # Compare the same physical time window in both streams.
    start = int(round(coarse_offset * AUDIO_SR))
    n = int(round(span * AUDIO_SR))
    a = ref_audio[start:start + n]
    b = other_audio[:n]
    if len(a) < AUDIO_SR or len(b) < AUDIO_SR:
        return coarse_offset, 0.0

    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(round(SEARCH_WINDOW_S * AUDIO_SR))

    corr = correlate(a, b, mode="full", method="fft")
    mid = len(b) - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    window = corr[lo:hi]

    aw = np.abs(window)
    peak = int(np.argmax(aw))
    # scipy.signal.correlate 'full' displacement of the peak. Verified sign:
    # a shared event appearing Δs later in `b` than in `a` gives d = -Δ·sr,
    # and the true offset is O = coarse + d/sr.
    lag_samples = peak - max_lag

    # Peak-to-sidelobe ratio over the search window, excluding a small guard
    # band around the peak so the peak itself doesn't inflate the noise stats.
    guard = int(0.02 * AUDIO_SR)
    mask = np.ones(len(aw), dtype=bool)
    mask[max(0, peak - guard):peak + guard + 1] = False
    sidelobe = aw[mask]
    if sidelobe.size and sidelobe.std() > 0:
        confidence = float((aw[peak] - sidelobe.mean()) / sidelobe.std())
    else:
        confidence = 0.0

    refined = coarse_offset + (lag_samples / AUDIO_SR)
    return refined, confidence


def load_session(session_dir: Path, use_audio: bool = True) -> Dict[str, Camera]:
    """Load cameras for a session and compute (audio-refined) timeline offsets.

    Head-only sessions (no populated wrist dirs) skip audio refinement
    entirely -- the head camera defines the timeline on its own.
    """
    mode = detect_session_mode(session_dir)
    files = find_camera_files(session_dir, quiet_missing=(mode == "head_only"))
    if REFERENCE_CAMERA not in files:
        raise RuntimeError(
            f"Reference camera '{REFERENCE_CAMERA}' not found in {session_dir}"
        )

    cameras: Dict[str, Camera] = {}
    for role, path in files.items():
        m = extract_metadata(str(path))
        cameras[role] = Camera(
            name=role, path=path,
            creation_time=m["creation_time"],
            creation_time_source=m["creation_time_source"],
            duration=m["duration"],
            fps_num=m["fps_num"], fps_den=m["fps_den"],
            n_frames=m["n_frames"], resolution=m["resolution"],
            rotation=m["rotation"],
            has_audio=m["has_audio"],
        )

    earliest = min(c.creation_time for c in cameras.values())
    for cam in cameras.values():
        cam.metadata_offset = (cam.creation_time - earliest).total_seconds()

    ref = cameras[REFERENCE_CAMERA]
    ref.offset_in_timeline = ref.metadata_offset
    ref.method = "reference"
    ref.confidence = 1.0
    ref.audio_offset = ref.metadata_offset

    # If any camera lacks an audio stream, xcorr cannot run -- short-circuit
    # cleanly with one informative message instead of a per-camera WARN
    # cascade.
    missing_audio = [r for r, c in cameras.items() if not c.has_audio]
    audio_possible = use_audio and not missing_audio
    if use_audio and missing_audio:
        print(f"NOTE  audio refinement disabled: no audio stream in "
              f"{missing_audio}. Using metadata offsets directly "
              f"(method='metadata-no-audio').", file=sys.stderr)

    for role, cam in cameras.items():
        if role == REFERENCE_CAMERA:
            continue
        coarse = cam.metadata_offset - ref.metadata_offset
        if audio_possible:
            try:
                refined, conf = refine_offset_audio(ref, cam, coarse)
            except subprocess.CalledProcessError as e:
                print(f"WARN  {role}: audio decode failed ({e}); using metadata",
                      file=sys.stderr)
                refined, conf = coarse, 0.0
            cam.audio_offset = refined
            cam.confidence = conf
            if conf >= CONFIDENCE_THRESHOLD:
                cam.offset_in_timeline = refined
                cam.method = "audio"
            else:
                print(f"WARN  {role}: low xcorr confidence {conf:.3f} "
                      f"(< {CONFIDENCE_THRESHOLD}); using metadata offset",
                      file=sys.stderr)
                cam.offset_in_timeline = coarse
                cam.method = "metadata-fallback"
        else:
            cam.offset_in_timeline = coarse
            cam.method = ("metadata-no-audio" if missing_audio
                          else "metadata")

    return cameras


def overlap_window(cameras: Dict[str, Camera]) -> Tuple[float, float]:
    """All-cameras overlap in the unified timeline (seconds)."""
    start = max(c.offset_in_timeline for c in cameras.values())
    end = min(c.offset_in_timeline + c.duration for c in cameras.values())
    return start, end


def write_sync_json(session_dir: Path, cameras: Dict[str, Camera]) -> Path:
    out = session_dir / "sync.json"
    win = overlap_window(cameras)
    payload = {
        "session": session_dir.name,
        "reference_camera": REFERENCE_CAMERA,
        "overlap_window_s": [round(win[0], 4), round(win[1], 4)],
        "cameras": {
            name: {
                "file": str(c.path),
                "creation_time": c.creation_time.isoformat(),
                "creation_time_source": c.creation_time_source,
                "fps": f"{c.fps_num}/{c.fps_den}",
                "n_frames": c.n_frames,
                "duration_s": round(c.duration, 4),
                "metadata_offset_s": round(c.metadata_offset
                                           - cameras[REFERENCE_CAMERA].metadata_offset, 4),
                "audio_offset_s": (round(c.audio_offset, 4)
                                   if c.audio_offset is not None else None),
                "offset_used_s": round(c.offset_in_timeline, 4),
                "confidence": round(c.confidence, 4),
                "method": c.method,
            }
            for name, c in cameras.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    return out


def print_sync_info(session_dir: Path, cameras: Dict[str, Camera]):
    mode = "head_only" if len(cameras) == 1 else "head_wrist"
    print("\n" + "=" * 78)
    print(f"SESSION: {session_dir.name}   (reference = {REFERENCE_CAMERA}, "
          f"mode = {mode})")
    print("=" * 78)
    if mode == "head_only":
        print(f"\n{REFERENCE_CAMERA:12} {cameras[REFERENCE_CAMERA].path.name}")
        c = cameras[REFERENCE_CAMERA]
        print(f"  start={c.creation_time.isoformat()}  "
              f"fps={c.fps_num}/{c.fps_den} ({c.fps:.3f})  frames={c.n_frames}")
        print(f"  duration={c.duration:.3f}s  (sync skipped: no wrist cameras)")
        print("=" * 78 + "\n")
        return
    for name, c in sorted(cameras.items(),
                          key=lambda kv: kv[1].offset_in_timeline):
        print(f"\n{name:12} {c.path.name}")
        print(f"  start={c.creation_time.isoformat()}  "
              f"fps={c.fps_num}/{c.fps_den} ({c.fps:.3f})  frames={c.n_frames}")
        print(f"  metadata_offset=+{c.metadata_offset - cameras[REFERENCE_CAMERA].metadata_offset:.3f}s"
              f"  audio_offset="
              f"{'%+.4fs' % c.audio_offset if c.audio_offset is not None else 'n/a'}")
        print(f"  -> offset_used=+{c.offset_in_timeline:.4f}s  "
              f"[{c.method}, conf={c.confidence:.3f}]")
    lo, hi = overlap_window(cameras)
    print(f"\nAll-camera overlap window: [{lo:.3f}s, {hi:.3f}s]  "
          f"({hi - lo:.3f}s)")
    print("=" * 78 + "\n")


def clip_camera(
    cam: Camera,
    ref: Camera,
    start_unified: float,
    end_unified: float,
    output_path: Path,
) -> Tuple[int, int]:
    """
    Frame-exact clip via accurate seek + re-encode.

    Snaps the start to the nearest frame using exact fps so all cameras begin
    on the same unified-timeline frame. Returns (start_frame, n_frames).
    """
    fps = cam.fps
    # Snap unified start to a whole reference-frame grid, then map per camera.
    ref_frame = round((start_unified - ref.offset_in_timeline) * ref.fps)
    snapped_unified = ref.offset_in_timeline + ref_frame / ref.fps

    local_start = snapped_unified - cam.offset_in_timeline
    local_end = end_unified - cam.offset_in_timeline
    local_start = max(0.0, local_start)
    local_end = min(cam.duration, local_end)
    duration = local_end - local_start
    if duration <= 0:
        raise ValueError(f"{cam.name}: empty clip after clamping")

    start_frame = round(local_start * fps)
    n_frames = round(duration * fps)

    # Fast pre-seek (input, keyframe) + accurate fine-seek (output) + re-encode.
    pre = max(0.0, local_start - 2.0)
    fine = local_start - pre
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{pre:.6f}", "-i", str(cam.path),
        "-ss", f"{fine:.6f}", "-t", f"{duration:.6f}",
        # DJI MP4s carry an MJPEG attached-pic thumbnail + djmd/dbgi data
        # streams; map only the real video + audio so libx264/aac don't
        # choke on them.
        "-map", "0:v:0", "-map", "0:a:0?", "-dn", "-sn",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map_metadata", "0", "-movflags", "+faststart",
        "-y", str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        tail = "\n".join(err.splitlines()[-15:]) or "(no ffmpeg stderr)"
        raise RuntimeError(
            f"ffmpeg failed for {cam.name} -> {output_path.name} "
            f"(exit {proc.returncode}):\n{tail}"
        )
    return start_frame, n_frames


def create_clip_batch(
    cameras: Dict[str, Camera],
    start: float,
    end: float,
    output_dir: Path,
    name: str,
    verbose: bool = True,
) -> Dict[str, str]:
    """Cut one clip per camera between unified-timeline [start, end].

    If the requested window drifts slightly past the overlap edges (e.g.
    because sync was recomputed after a tag was authored), the range is
    clamped into the overlap window rather than rejected. Only ranges that
    miss the overlap entirely are skipped.
    """
    lo, hi = overlap_window(cameras)
    if end <= lo or start >= hi:
        print(f"WARN  '{name}' [{start:.3f},{end:.3f}] entirely outside "
              f"overlap [{lo:.3f},{hi:.3f}] -- skipped", file=sys.stderr)
        return {}

    clamped_start = max(start, lo)
    clamped_end = min(end, hi)
    if (clamped_start != start or clamped_end != end) and verbose:
        print(f"  '{name}' clamped to overlap: "
              f"[{start:.3f},{end:.3f}] -> "
              f"[{clamped_start:.3f},{clamped_end:.3f}]")
    start, end = clamped_start, clamped_end

    output_dir.mkdir(parents=True, exist_ok=True)
    ref = cameras[REFERENCE_CAMERA]
    results = {}
    for role, cam in cameras.items():
        out = output_dir / f"{name}_{role}.mp4"
        sf, nf = clip_camera(cam, ref, start, end, out)
        if verbose:
            print(f"  {role:12} frames {sf}..{sf + nf} ({nf}) -> {out.name}")
        results[role] = str(out)
    return results


def has_reference_camera(session_dir: Path) -> bool:
    """True if the session has a usable head (reference) video file."""
    cam_dir = session_dir / CAMERA_LAYOUT[REFERENCE_CAMERA]
    if not cam_dir.is_dir():
        return False
    return any(p.suffix.lower() in (".mp4", ".mov") for p in cam_dir.iterdir())


def resolve_session(args) -> Path:
    raw_root = Path(args.raw_root)
    if args.session:
        # Explicit name -> literal folder under raw_root; no pattern matching.
        candidate = raw_root / args.session
        if not candidate.is_dir():
            print(f"Session folder not found: {candidate}", file=sys.stderr)
            sys.exit(1)
        return candidate
    sessions = discover_sessions(raw_root)
    if not sessions:
        print(f"No 'video N' sessions found under {raw_root} "
              f"(pass --session NAME to use a non-standard folder)",
              file=sys.stderr)
        sys.exit(1)
    # Newest session that is actually populated; skip incomplete folders
    # (e.g. a freshly-created 'video N' whose footage isn't copied in yet).
    for s in reversed(sessions):
        if has_reference_camera(s):
            return s
        print(f"WARN  skipping incomplete session '{s.name}' "
              f"(no {REFERENCE_CAMERA} video yet)", file=sys.stderr)
    print("No session with a head camera found", file=sys.stderr)
    sys.exit(1)


def process_session(session_dir: Path, args):
    cameras = load_session(session_dir, use_audio=not args.no_audio)
    print_sync_info(session_dir, cameras)
    # Head-only sessions have nothing to sync -- no sync.json artifact.
    if len(cameras) > 1:
        sync_path = write_sync_json(session_dir, cameras)
        print(f"Wrote {sync_path}")
    else:
        # Delete a stale sync.json from a prior head_wrist run, if any.
        stale = session_dir / "sync.json"
        if stale.is_file():
            stale.unlink()

    if args.info:
        return

    # session_dir = data/raw/video N  ->  parents[1] = data  ->  data/output/...
    out_base = session_dir.parents[1] / "output"
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = out_base / output_dir

    # Default single run (no --clip/--config): emit one clip per camera covering
    # the whole overlap window -- the "session-overlap" output the tagger scrubs.
    if not args.clip and not args.config:
        lo, hi = overlap_window(cameras)
        overlap_dir = out_base / "session_overlap"
        name = session_dir.name.replace(" ", "_")
        print(f"\nSession-overlap clip '{name}' "
              f"[{lo:.3f},{hi:.3f}] ({hi - lo:.3f}s)")
        create_clip_batch(cameras, lo, hi, overlap_dir, name,
                          verbose=not args.quiet)
        return

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        out = output_dir
        if cfg.get("output_dir"):
            out = out_base / cfg["output_dir"]
        for clip in cfg.get("clips", []):
            label = clip.get("task_label", "")
            print(f"\nClip '{clip['name']}' "
                  f"[{clip['start']:.3f},{clip['end']:.3f}] {label}")
            create_clip_batch(cameras, clip["start"], clip["end"],
                               out, clip["name"], verbose=not args.quiet)
    elif args.clip:
        s, e = args.clip
        print(f"\nClip '{args.name}' [{s:.3f},{e:.3f}]")
        create_clip_batch(cameras, s, e, output_dir, args.name,
                           verbose=not args.quiet)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT),
                   help="Directory containing session folders "
                        "(default: %(default)s)")
    p.add_argument("--session", help="Specific session, e.g. 'video 2' "
                                     "(default: latest)")
    p.add_argument("--all", action="store_true",
                   help="Process every discovered session")
    p.add_argument("--output", default="episode_clips",
                   help="Per-episode clip output dir (relative to data/output/). "
                        "The default session-overlap output always goes to "
                        "data/output/session_overlap/ and ignores this flag.")
    p.add_argument("--info", action="store_true",
                   help="Sync only: print info + write sync.json, no video")
    p.add_argument("--clip", type=float, nargs=2, metavar=("START", "END"),
                   help="Single clip in unified-timeline seconds")
    p.add_argument("--name", default="clip_001", help="Clip output name")
    p.add_argument("--config", help="JSON cut-sheet with multiple clips")
    p.add_argument("--no-audio", action="store_true",
                   help="Skip audio refinement (metadata offsets only)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    if args.all:
        for s in discover_sessions(raw_root):
            if not has_reference_camera(s):
                print(f"WARN  skipping incomplete session '{s.name}' "
                      f"(no {REFERENCE_CAMERA} video yet)", file=sys.stderr)
                continue
            process_session(s, args)
    else:
        process_session(resolve_session(args), args)


if __name__ == "__main__":
    main()
