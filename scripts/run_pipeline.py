#!/usr/bin/env python3
"""Run the whole H2R pipeline end-to-end with one command.

Defaults aim at the "common case": process the latest session under
``data/raw/`` and emit a LeRobot v2.1 dataset under
``data/output/lerobot_v2/``. No arguments are required::

    python3 scripts/run_pipeline.py

The bare invocation:

  1. Resolves the latest session (auto-detects head_only vs head_wrist).
  2. Launches the episode tagger (system python3 -- needs Tk + cv2). The
     operator scrubs, marks episodes, and saves on Ctrl+S; that writes the
     cut sheet AND cuts the per-episode clips in one step.
  3. Once the tagger exits, launches ``convert_to_lerobot.py`` under the
     ``hawor`` conda env to run HaWoR + write the LeRobot dataset.

Re-running with no changes is cheap because the converter's HaWoR cache
hits on unchanged clip content.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# scripts/ is on sys.path automatically (Python adds the launched script's
# directory). Reuse sync_videos's session discovery instead of re-globbing.
import sync_videos as sv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "output"
DEFAULT_HAWOR_PY = Path.home() / "anaconda3" / "envs" / "hawor" / "bin" / "python"


def session_clips_dir(session: Path,
                      output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Per-session clips dir: data/output/<session>/episode_clips/."""
    return output_root / session.name / "episode_clips"


def session_dataset_dir(session: Path,
                        output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Per-session LeRobot dataset dir: data/output/<session>/lerobot_v2/."""
    return output_root / session.name / "lerobot_v2"


def resolve_session(raw_root: Path, name: str | None) -> Path:
    """Pick the session dir under raw_root.

    When ``name`` is given it is treated as the literal folder name under
    ``raw_root`` -- no pattern matching, so arbitrary capture names work.
    Otherwise fall back to the latest ``video N`` session that has a head
    camera populated.
    """
    if name:
        candidate = raw_root / name
        if not candidate.is_dir():
            sys.exit(f"Session folder not found: {candidate}")
        return candidate
    sessions = sv.discover_sessions(raw_root)
    if not sessions:
        sys.exit(f"No 'video N' sessions under {raw_root} (pass --session "
                 f"NAME to use a non-standard folder)")
    for s in reversed(sessions):
        if sv.has_reference_camera(s):
            return s
    sys.exit("No session has a head camera populated yet")


def run_tagger(session: Path, output_dir_rel: str) -> None:
    """Launch the tagger, forcing the per-session output_dir.

    ``output_dir_rel`` is the path under ``data/output/`` that the tagger
    writes into clips.json and cuts triplets to (e.g. ``"Sample/episode_clips"``).
    We pass it explicitly so a stale ``output_dir`` in an existing clips.json
    cannot redirect the cuts to a shared/wrong location.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "episode_tagger.py"),
        "--session", session.name,
        "--output-dir", output_dir_rel,
    ]
    print(f"[stage 1] tagger -> {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        sys.exit(f"Tagger exited with status {rc}")


def run_converter(cut_sheet: Path, clips_dir: Path, out: Path,
                  hawor_py: Path) -> None:
    if not hawor_py.is_file():
        sys.exit(f"hawor python not found at {hawor_py}. "
                 f"Set --hawor-python or HAWOR_PYTHON env var.")
    cmd = [
        str(hawor_py),
        str(REPO_ROOT / "scripts" / "convert_to_lerobot.py"),
        "--cut-sheet", str(cut_sheet),
        "--clips-dir", str(clips_dir),
        "--out", str(out),
    ]
    print(f"[stage 2] converter -> {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        sys.exit(f"Converter exited with status {rc}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session", help="session name (default: latest under "
                                      "data/raw/)")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT),
                   help="raw data root (default: %(default)s)")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="output root (default: %(default)s). Per-session "
                        "sub-dirs are <output-root>/<session>/episode_clips "
                        "and <output-root>/<session>/lerobot_v2.")
    p.add_argument("--clips-dir",
                   help="override the per-session clips dir "
                        "(default: <output-root>/<session>/episode_clips)")
    p.add_argument("--out",
                   help="override the LeRobot dataset dir "
                        "(default: <output-root>/<session>/lerobot_v2)")
    p.add_argument("--hawor-python",
                   default=os.environ.get("HAWOR_PYTHON",
                                          str(DEFAULT_HAWOR_PY)),
                   help="python interpreter for the converter "
                        "(default: %(default)s; override with HAWOR_PYTHON)")
    p.add_argument("--skip-tag", action="store_true",
                   help="skip the tagger; reuse the existing clips.json + "
                        "clipped MP4s (re-convert only)")
    args = p.parse_args(argv)

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    session = resolve_session(raw_root, args.session)
    mode = sv.detect_session_mode(session)
    cut_sheet = session / "clips.json"
    # Per-session output paths. Overridable but default = session-scoped so
    # different captures cannot clobber each other.
    clips_dir = (Path(args.clips_dir) if args.clips_dir
                 else session_clips_dir(session, output_root))
    out = (Path(args.out) if args.out
           else session_dataset_dir(session, output_root))
    # Path passed to the tagger's --output-dir, relative to data/output/.
    tagger_output_dir = str(clips_dir.relative_to(output_root))
    hawor_py = Path(args.hawor_python)

    print("=" * 70)
    print(f"  session:    {session.name}   ({mode})")
    print(f"  cut sheet:  {cut_sheet}")
    print(f"  clips dir:  {clips_dir}")
    print(f"  dataset:    {out}")
    print(f"  hawor py:   {hawor_py}")
    print("=" * 70)

    if args.skip_tag:
        if not cut_sheet.is_file():
            sys.exit(f"--skip-tag set but {cut_sheet} does not exist")
        try:
            n = len(json.loads(cut_sheet.read_text()).get("clips", []))
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"Could not parse {cut_sheet}: {exc}")
        print(f"[stage 1] tagger SKIPPED  ({n} episodes in cut sheet)")
    else:
        run_tagger(session, tagger_output_dir)

    if not cut_sheet.is_file():
        sys.exit(f"No cut sheet at {cut_sheet} -- did the tagger save? "
                 f"(Ctrl+S inside the GUI)")
    if not clips_dir.is_dir() or not any(clips_dir.glob("*_head.mp4")):
        sys.exit(f"No head clips found in {clips_dir} -- tagger save likely "
                 f"failed or wrote to a different dir")

    run_converter(cut_sheet, clips_dir, out, hawor_py)

    print("\nDone.  Dataset at:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
