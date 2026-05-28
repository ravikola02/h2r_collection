"""Episode discovery — pair cut-sheet entries with clipped MP4 triplets.

The cut sheet (``clips.json``) is the same file the operator fed
``scripts/sync_videos.py`` and is the source of truth for which episodes
exist and their labels. ``sync_videos.create_clip_batch`` writes each clip as
``{name}_{role}.mp4`` into the clips dir, so discovery is: for every cut-sheet
entry, resolve the three role files by that naming convention.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROLES = ("head", "wrist_left", "wrist_right")


@dataclass
class EpisodeSpec:
    episode_id: int
    name: str
    task_label: str
    dominant_hand: str
    clips: Dict[str, Path] = field(default_factory=dict)  # role -> mp4 path

    @property
    def head_clip(self) -> Path:
        return self.clips["head"]


def video_hash(path: Path) -> str:
    """Fast content fingerprint for cache keys.

    sha1 over file size + first/last 1 MiB. Cheap on multi-hundred-MB clips
    while still changing whenever the clip is re-cut.
    """
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode())
    chunk = 1 << 20
    with path.open("rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def load_episodes(cut_sheet: Path, clips_dir: Path) -> List[EpisodeSpec]:
    """Parse the cut sheet and resolve each episode's clip triplet.

    Cut-sheet schema (see README.md / sync_videos.py):
    ``{"clips": [{"name", "start", "end", "episode_id"?, "task_label"?,
    "dominant_hand"?}, ...]}``. ``name``/``episode_id`` fall back to the list
    index. A missing ``head`` clip is fatal for that episode; missing wrist
    clips are warned and skipped per camera (e.g. empty ``right_w_cam``).
    Head-only sessions (no wrist clip exists for any episode) suppress the
    per-camera WARN noise.
    """
    cut_sheet = Path(cut_sheet)
    clips_dir = Path(clips_dir)
    cfg = json.loads(cut_sheet.read_text())
    raw_clips = cfg.get("clips", [])
    if not raw_clips:
        raise ValueError(f"No 'clips' entries in cut sheet {cut_sheet}")

    # First pass: head-only if no wrist clip exists for any entry.
    any_wrist_clip = any(
        (clips_dir / f"{c.get('name', f'episode_{i + 1:03d}')}_{role}.mp4").is_file()
        for i, c in enumerate(raw_clips)
        for role in ("wrist_left", "wrist_right")
    )

    episodes: List[EpisodeSpec] = []
    for idx, clip in enumerate(raw_clips):
        name = clip.get("name", f"episode_{idx + 1:03d}")
        episode_id = int(clip.get("episode_id", idx + 1))
        clips: Dict[str, Path] = {}
        for role in ROLES:
            p = clips_dir / f"{name}_{role}.mp4"
            if p.is_file():
                clips[role] = p
            elif role != "head" and any_wrist_clip:
                print(f"WARN  episode '{name}': missing {role} clip "
                      f"({p.name}) -- skipping that camera", file=sys.stderr)
        if "head" not in clips:
            print(f"WARN  episode '{name}': no head clip -- episode skipped",
                  file=sys.stderr)
            continue
        episodes.append(EpisodeSpec(
            episode_id=episode_id,
            name=name,
            task_label=clip.get("task_label", ""),
            dominant_hand=clip.get("dominant_hand", "right"),
            clips=clips,
        ))

    if not episodes:
        raise ValueError(
            f"No episodes resolved: cut sheet {cut_sheet} lists "
            f"{len(raw_clips)} clip(s) but none had a head MP4 in {clips_dir}")
    return episodes
