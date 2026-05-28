#!/usr/bin/env python3
"""Convert per-episode clipped triplets -> LeRobot 2.1 dataset (dual-hand 14-d).

Single-file orchestrator that runs **inside the hawor conda env** because it
imports HaWoR directly. Invoke with the hawor python explicitly::

    $HOME/anaconda3/envs/hawor/bin/python \\
        scripts/convert_to_lerobot.py \\
        --cut-sheet "data/raw/<session>/clips.json" \\
        --clips-dir "data/output/<session>/episode_clips" \\
        --out       "data/output/<session>/lerobot_v2"

Per-episode flow:

1. Run HaWoR end-to-end on the head clip (detect+track, motion estimation,
   masked DROID-SLAM, infiller). Cache the result (``hawor_result.npz``)
   under ``<out>/cache/<vhash16>_<bridge_version>/``. Subsequent runs
   skip stages 1-4 on cache hit. The whole ``cache/`` dir is removed once
   the dataset finishes successfully (see end of ``main``).
2. Build a dual-hand 14-d positional state from the MANO joints:

       [L pos 3, L rpy 3, R pos 3, R rpy 3, L grip 1, R grip 1]

   wrist position = joint 0 in HaWoR-world frame; orientation = intrinsic
   XYZ Euler from ``aa_to_rotmat(pred_rot[h])``.
3. Encode one MP4 per camera at 1280x720, ``--traj-fps`` (default 30).
4. Write the row's slice of the LeRobot parquet (one parquet per episode,
   under ``data/chunk-000/``).

After all episodes finish, emit ``meta/info.json``, ``meta/episodes.jsonl``,
``meta/tasks.jsonl``, ``meta/stats.json``.

The gripper signal in state dims 12/13 is a **linear normalization of
||thumb_tip - pinky_tip||** in ``[0, 1]``: open at ``GRIPPER_OPEN_M``,
closed at ``GRIPPER_CLOSED_M``, clipped. Recalibrate after the Week-3
study.
"""

# --------------------------------------------------------------------------
# Self-bootstrap LD_LIBRARY_PATH so the conda env's libstdc++ (which carries
# the newer CXXABI required by lietorch_backends + pytorch3d) loads before
# the system one. Without this, running the env's bin/python directly skips
# conda's activate hooks and DROID-SLAM's C extensions fail to import.
# Has to happen BEFORE we import torch / pytorch3d / lietorch.
import os, sys  # noqa: E401
from pathlib import Path  # noqa: E402

_env_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
if Path(_env_lib).is_dir() and _env_lib not in os.environ.get(
        "LD_LIBRARY_PATH", "").split(":"):
    _new_env = os.environ.copy()
    _new_env["LD_LIBRARY_PATH"] = (
        f"{_env_lib}:{os.environ['LD_LIBRARY_PATH']}"
        if "LD_LIBRARY_PATH" in os.environ else _env_lib)
    os.execve(sys.executable, [sys.executable] + sys.argv, _new_env)

# Force /usr/bin/ffmpeg ahead of the snap-bundled /snap/bin/ffmpeg (which is
# sandboxed and cannot read/write under /tmp or non-home paths). Both our
# encode_video and HaWoR's internal detect_track_video.extract_frames call
# `subprocess.run(["ffmpeg", ...])`, so the PATH shadowing flows through
# both.
if Path("/usr/bin/ffmpeg").is_file():
    os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
# --------------------------------------------------------------------------

import argparse
import contextlib
import gc
import json
import shutil
import subprocess
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# repo + HaWoR plumbing (runs in the hawor conda env, in-process imports)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
HAWOR_REPO = Path(os.environ.get(
    "HAWOR_REPO", REPO_ROOT / "HaWoR")).resolve()

# Preserve the user's invocation cwd so CLI path args can still be relative
# to where they ran the script from.
USER_CWD = Path.cwd()

# HaWoR's masked_droid_slam.py and _DATA/data references are CWD-relative
# (`sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')` etc.), so we must
# chdir(HAWOR_REPO) BEFORE importing the pipeline modules.
os.chdir(str(HAWOR_REPO))

# Our own helpers (cut-sheet parser + video_hash). scripts/ is already on
# sys.path (Python auto-adds the launched script's directory), and these
# modules have no torch deps so the import is safe under the hawor env.
from episodes import EpisodeSpec, load_episodes, video_hash  # noqa: E402

# HaWoR's own modules
sys.path.insert(0, str(HAWOR_REPO))
import torch  # noqa: E402
from scripts.scripts_test_video.detect_track_video import detect_track_video  # noqa: E402
from scripts.scripts_test_video.hawor_video import (  # noqa: E402
    hawor_motion_estimation, hawor_infiller)
from scripts.scripts_test_video.hawor_slam import hawor_slam  # noqa: E402
from hawor.utils.process import run_mano, run_mano_left  # noqa: E402
from hawor.utils.geometry import aa_to_rotmat  # noqa: E402
from lib.eval_utils.custom_utils import load_slam_cam  # noqa: E402


# ---------------------------------------------------------------------------
# constants — change `BRIDGE_VERSION` to invalidate the cache cleanly
# ---------------------------------------------------------------------------

BRIDGE_VERSION = "hawor-v0.1"
CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000

# smplx MANO joint indices (verified against HaWoR/example/head_test/world_space_res.pth):
#   joint  0  = wrist
#   joint 16 = thumb tip
#   joint 20 = little-finger (pinky) tip
THUMB_TIP_IDX = 16
PINKY_TIP_IDX = 20

# Gripper signal = linear normalization of ||thumb_tip - pinky_tip||,
# clipped to [0, 1]. Eyeballed against ep000-ep002 observed range
# (~12-44 mm); revisit once the Week-3 calibration study lands.
GRIPPER_OPEN_M = 0.04   # distance at which gripper reads 0.0 (open)
GRIPPER_CLOSED_M = 0.01  # distance at which gripper reads 1.0 (closed)

STATE_DIM = 14
STATE_LAYOUT = "[L pos 3, L rpy 3, R pos 3, R rpy 3, L grip 1, R grip 1]"
STATE_NAMES = [
    "left.wrist.x", "left.wrist.y", "left.wrist.z",
    "left.wrist.roll", "left.wrist.pitch", "left.wrist.yaw",
    "right.wrist.x", "right.wrist.y", "right.wrist.z",
    "right.wrist.roll", "right.wrist.pitch", "right.wrist.yaw",
    "left.gripper", "right.gripper",
]
CAMERA_KEYS = {
    "head":        "observation.images.head",
    "wrist_left":  "observation.images.wrist_left",
    "wrist_right": "observation.images.wrist_right",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _chdir(path: Path):
    """HaWoR's MANO/cfg paths are relative (`_DATA/data/mano`); we must run
    from HAWOR_REPO so it can find the model weights + MANO pickles."""
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _user_path(s: str) -> Path:
    """Resolve a user-supplied path against USER_CWD (their invocation dir),
    not the HAWOR_REPO we chdir'd into at import time."""
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    return (USER_CWD / p).resolve()


def _chunk_id(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def _ep_parquet_path(out_root: Path, episode_index: int) -> Path:
    return (out_root / "data" / f"chunk-{_chunk_id(episode_index):03d}"
            / f"episode_{episode_index:06d}.parquet")


def _ep_video_path(out_root: Path, video_key: str, episode_index: int) -> Path:
    return (out_root / "videos" / f"chunk-{_chunk_id(episode_index):03d}"
            / video_key / f"episode_{episode_index:06d}.mp4")


class HaworArgs:
    """Duck-type for HaWoR's `args.video_path` / `.checkpoint` / etc. access."""
    def __init__(self, video_path: Path, checkpoint: Path,
                 infiller_weight: Path, img_focal: Optional[float] = None):
        self.video_path = str(video_path)
        self.checkpoint = str(checkpoint)
        self.infiller_weight = str(infiller_weight)
        self.img_focal = img_focal
        self.input_type = "file"


# ---------------------------------------------------------------------------
# Stage D — run HaWoR (or load cached result)
# ---------------------------------------------------------------------------

def _stage_input_clip(head_clip: Path, workdir: Path) -> Path:
    """Symlink the head clip into the per-episode workdir so HaWoR's
    extracted_images/, tracks_*, SLAM/ caches live there, not next to the
    source episode_clips/<name>_head.mp4 (which would pollute multiple GB
    per episode into the clip dir)."""
    staged = workdir / "head.mp4"
    if staged.exists() or staged.is_symlink():
        return staged
    try:
        staged.symlink_to(head_clip.resolve())
    except OSError:
        shutil.copyfile(head_clip, staged)
    return staged


def run_hawor_or_cache(head_clip: Path, cache_root: Path,
                        hawor_repo: Path,
                        img_focal: Optional[float],
                        verbose: bool = True) -> Dict[str, np.ndarray]:
    """Run the four HaWoR stages on `head_clip`, or load the cached output.

    Returns a dict of plain numpy arrays — no torch tensors leak past here.
    """
    vhash16 = video_hash(head_clip)[:16]
    workdir = cache_root / f"{vhash16}_{BRIDGE_VERSION}"
    workdir.mkdir(parents=True, exist_ok=True)
    result_path = workdir / "hawor_result.npz"

    if result_path.is_file():
        if verbose:
            print(f"  [HAWOR] cache hit ({result_path.relative_to(cache_root)})")
        z = np.load(result_path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    staged = _stage_input_clip(head_clip, workdir)
    args = HaworArgs(
        video_path=staged,
        checkpoint=hawor_repo / "weights/hawor/checkpoints/hawor.ckpt",
        infiller_weight=hawor_repo / "weights/hawor/checkpoints/infiller.pt",
        img_focal=img_focal,
    )

    with _chdir(hawor_repo):
        if verbose:
            print(f"  [HAWOR] stage 1: detect_track on {staged.name}")
        s, e, seq_folder, _ = detect_track_video(args)

        if verbose:
            print(f"  [HAWOR] stage 2: motion_estimation [{s},{e})")
        frame_chunks_all, focal_used = hawor_motion_estimation(
            args, s, e, seq_folder)

        slam_path = Path(seq_folder) / f"SLAM/hawor_slam_w_scale_{s}_{e}.npz"
        if not slam_path.exists():
            if verbose:
                print(f"  [HAWOR] stage 3: masked DROID-SLAM (slow) ...")
            hawor_slam(args, s, e)
        elif verbose:
            print(f"  [HAWOR] stage 3: SLAM cache hit ({slam_path.name})")

        if verbose:
            print(f"  [HAWOR] stage 4: infiller (cam->world + missing frames)")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
            hawor_infiller(args, s, e, frame_chunks_all)

        R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(str(slam_path))

        # MANO forward -> world-frame joints for both hands.
        if verbose:
            print(f"  [HAWOR] MANO forward (joints in world frame)")
        out_l = run_mano_left(pred_trans[0:1], pred_rot[0:1],
                              pred_hand_pose[0:1], betas=pred_betas[0:1])
        out_r = run_mano(pred_trans[1:2], pred_rot[1:2],
                         pred_hand_pose[1:2], betas=pred_betas[1:2])
        joints_left = out_l["joints"][0].detach().cpu().numpy()    # (T, 21, 3)
        joints_right = out_r["joints"][0].detach().cpu().numpy()

        # Wrist world-frame rotation = aa_to_rotmat of root orientation.
        wrist_rot_left = aa_to_rotmat(
            pred_rot[0].reshape(-1, 3)).detach().cpu().numpy()  # (T,3,3)
        wrist_rot_right = aa_to_rotmat(
            pred_rot[1].reshape(-1, 3)).detach().cpu().numpy()

    T = int(pred_trans.shape[1])
    # T_world->cam composed from SLAM:
    cam_pose_w2c = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    cam_pose_w2c[:, :3, :3] = R_w2c.detach().cpu().numpy()
    cam_pose_w2c[:, :3, 3] = t_w2c.detach().cpu().numpy()

    timestamps = np.arange(T, dtype=np.float64) / 30.0  # HaWoR runs at 30 fps

    result = {
        "timestamps": timestamps,
        "joints_left": joints_left.astype(np.float64),
        "joints_right": joints_right.astype(np.float64),
        "wrist_rot_left": wrist_rot_left.astype(np.float64),
        "wrist_rot_right": wrist_rot_right.astype(np.float64),
        "valid_left": np.asarray(pred_valid[0], dtype=bool),
        "valid_right": np.asarray(pred_valid[1], dtype=bool),
        "cam_pose_w2c": cam_pose_w2c,
        "bridge_version": np.array(BRIDGE_VERSION),
        "img_focal": np.float64(focal_used),
    }
    np.savez_compressed(result_path, **result)
    if verbose:
        print(f"  [HAWOR] cached -> {result_path.relative_to(cache_root)}")

    # Reclaim disk: HaWoR's intermediate scratch (extracted_images/ +
    # tracks_*/ + SLAM/) lives under `<workdir>/head/` and runs 3-5 GB per
    # episode. The npz now contains everything downstream needs; if you bump
    # BRIDGE_VERSION the scratch will simply re-extract.
    scratch = Path(seq_folder)
    if scratch.is_dir() and scratch.resolve().is_relative_to(workdir.resolve()):
        before = sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
        shutil.rmtree(scratch, ignore_errors=True)
        if verbose:
            print(f"  [HAWOR] reclaimed {before / 1e9:.2f} GB of scratch "
                  f"({scratch.name}/)")
    return result


# ---------------------------------------------------------------------------
# Stage F — 14-d dual-hand state + placeholder gripper
# ---------------------------------------------------------------------------

def _normalize_grip(d: np.ndarray) -> np.ndarray:
    """Linear remap of thumb-pinky distance: OPEN_M -> 0, CLOSED_M -> 1,
    clipped to [0, 1]."""
    span = GRIPPER_OPEN_M - GRIPPER_CLOSED_M
    return np.clip((GRIPPER_OPEN_M - d) / span, 0.0, 1.0)


def build_state_14d(hawor: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack pos+RPY+grip per hand into the (T, 14) state matrix.

    Gripper dims (12, 13) are ``_normalize_grip(||thumb - pinky||)``, a
    continuous signal in [0, 1] where 1.0 = fully closed.
    """
    jl, jr = hawor["joints_left"], hawor["joints_right"]
    rl, rr = hawor["wrist_rot_left"], hawor["wrist_rot_right"]
    T = jl.shape[0]

    pos_l = jl[:, 0, :]                 # wrist position, world
    pos_r = jr[:, 0, :]
    rpy_l = Rotation.from_matrix(rl).as_euler("XYZ")   # (T,3) intrinsic XYZ
    rpy_r = Rotation.from_matrix(rr).as_euler("XYZ")

    raw_l = np.linalg.norm(
        jl[:, THUMB_TIP_IDX] - jl[:, PINKY_TIP_IDX], axis=1)
    raw_r = np.linalg.norm(
        jr[:, THUMB_TIP_IDX] - jr[:, PINKY_TIP_IDX], axis=1)
    # grip_l = _normalize_grip(raw_l)
    # grip_r = _normalize_grip(raw_r)
    grip_l = raw_l  
    grip_r = raw_r
    state = np.concatenate([pos_l, rpy_l, pos_r, rpy_r,
                            grip_l[:, None], grip_r[:, None]], axis=1)
    assert state.shape == (T, STATE_DIM), state.shape
    return state


def derive_action_from_state(state: np.ndarray) -> np.ndarray:
    """action[t] = state[t+1]; last frame repeats (terminal no-op).

    Standard IL convention for demonstrations: the action label at time t
    is the target the policy should arrive at by t+1.
    """
    return np.concatenate([state[1:], state[-1:]], axis=0)


# ---------------------------------------------------------------------------
# Stage G — video encode + parquet (LeRobot v2.1 layout)
# ---------------------------------------------------------------------------

def encode_video(clip: Path, out_path: Path, fps: float,
                  n_frames: int, size: Tuple[int, int]) -> int:
    """ffmpeg clip -> single H.264 MP4 at `size` and `fps`. Returns the
    written frame count (probed back so we can sanity-check vs parquet)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    w, h = size
    cmd = [
        "ffmpeg", "-v", "error", "-noautorotate",
        "-i", str(clip),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "23",
        "-an",  # no audio
        "-y", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg encode failed for {clip.name}:\n"
                           + "\n".join(tail[-10:]))
    # Probe back the actual frame count.
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", str(out_path),
    ], capture_output=True, text=True)
    try:
        return int(probe.stdout.strip())
    except ValueError:
        return -1


def write_episode_parquet(out_root: Path, episode_index: int,
                           task_index: int,
                           state: np.ndarray, action: np.ndarray,
                           fps: float, global_index_offset: int) -> int:
    """Write `data/chunk-XXX/episode_NNNNNN.parquet`. Returns row count."""
    n = int(state.shape[0])
    frame_index = np.arange(n, dtype=np.int64)
    timestamp = (frame_index / fps).astype(np.float32)
    episode_index_col = np.full(n, episode_index, dtype=np.int64)
    task_index_col = np.full(n, task_index, dtype=np.int64)
    index = (global_index_offset + frame_index).astype(np.int64)

    # state/action stored as list<float32>[14] per LeRobot convention.
    state_f32 = state.astype(np.float32)
    action_f32 = action.astype(np.float32)
    state_list = [row.tolist() for row in state_f32]
    action_list = [row.tolist() for row in action_f32]

    table = pa.table({
        "observation.state": pa.array(
            state_list, type=pa.list_(pa.float32(), STATE_DIM)),
        "action": pa.array(
            action_list, type=pa.list_(pa.float32(), STATE_DIM)),
        "timestamp":     timestamp,
        "frame_index":   frame_index,
        "episode_index": episode_index_col,
        "index":         index,
        "task_index":    task_index_col,
    })

    out_path = _ep_parquet_path(out_root, episode_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return n


def write_episode_videos(out_root: Path, episode_index: int,
                          clips: Dict[str, Path], fps: float,
                          n_frames: int, size: Tuple[int, int],
                          verbose: bool = True) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for role, clip in clips.items():
        video_key = CAMERA_KEYS[role]
        out_path = _ep_video_path(out_root, video_key, episode_index)
        cnt = encode_video(clip, out_path, fps, n_frames, size)
        counts[role] = cnt
        if verbose:
            flag = "" if cnt == n_frames else f"  (!= {n_frames} parquet rows)"
            print(f"  [G] {role:11} -> {out_path.name}  {cnt} frames{flag}")
    return counts


# ---------------------------------------------------------------------------
# meta/ writers — emitted once after all episodes are processed
# ---------------------------------------------------------------------------

def _aggregate_stats(out_root: Path, episode_count: int,
                      roles: List[str],
                      image_sample_per_episode: int = 8,
                      size: Tuple[int, int] = (1280, 720)) -> Dict:
    """Compute per-feature mean/std/min/max/count across the dataset.

    State/action/pinch stats are exact (we read every row of every parquet).
    Image stats are channel-wise (shape [3,1,1]), sampled from a handful of
    frames per camera per episode to keep this fast.
    """
    # Numeric features.
    all_state, all_action = [], []
    for i in range(episode_count):
        df = pd.read_parquet(_ep_parquet_path(out_root, i))
        all_state.append(np.stack(df["observation.state"].to_numpy()))
        all_action.append(np.stack(df["action"].to_numpy()))
    state = np.concatenate(all_state, axis=0)
    action = np.concatenate(all_action, axis=0)

    def _vec_stats(arr: np.ndarray) -> Dict:
        return {
            "mean":  arr.mean(axis=0).astype(np.float32).tolist(),
            "std":   arr.std(axis=0).astype(np.float32).tolist(),
            "min":   arr.min(axis=0).astype(np.float32).tolist(),
            "max":   arr.max(axis=0).astype(np.float32).tolist(),
            "count": [int(arr.shape[0])],
        }

    stats = {
        "observation.state": _vec_stats(state),
        "action":            _vec_stats(action),
    }

    # Image stats: per-channel pixel mean/std sampled from MP4s. Shape
    # [3,1,1] is LeRobot's standard for image normalization tensors.
    for role in roles:
        video_key = CAMERA_KEYS[role]
        ch_mean, ch_sq, ch_min, ch_max, n_pixels = (
            np.zeros(3, np.float64), np.zeros(3, np.float64),
            np.full(3, np.inf), np.full(3, -np.inf), 0)
        for i in range(episode_count):
            vp = _ep_video_path(out_root, video_key, i)
            samples = _sample_video_frames(vp, image_sample_per_episode, size)
            if samples is None:
                continue
            arr = samples.astype(np.float64) / 255.0  # (S, H, W, 3)
            ch_mean += arr.sum(axis=(0, 1, 2))
            ch_sq   += (arr ** 2).sum(axis=(0, 1, 2))
            ch_min   = np.minimum(ch_min, arr.min(axis=(0, 1, 2)))
            ch_max   = np.maximum(ch_max, arr.max(axis=(0, 1, 2)))
            n_pixels += arr.shape[0] * arr.shape[1] * arr.shape[2]
        if n_pixels == 0:
            continue
        mean = ch_mean / n_pixels
        var = ch_sq / n_pixels - mean ** 2
        std = np.sqrt(np.clip(var, 0.0, None))
        stats[video_key] = {
            "mean":  mean.reshape(3, 1, 1).astype(np.float32).tolist(),
            "std":   std.reshape(3, 1, 1).astype(np.float32).tolist(),
            "min":   ch_min.reshape(3, 1, 1).astype(np.float32).tolist(),
            "max":   ch_max.reshape(3, 1, 1).astype(np.float32).tolist(),
            "count": [int(n_pixels)],
        }
    return stats


def _sample_video_frames(video_path: Path, n_samples: int,
                          size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Decode `n_samples` evenly-spaced JPEG frames via ffmpeg and return
    (S, H, W, 3) uint8. None on failure."""
    if not video_path.is_file():
        return None
    w, h = size
    # `select` keeps every k-th frame where k = nb_frames / n_samples;
    # easiest: scale + thumbnail filter is fine for a few samples.
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", f"thumbnail=10,scale={w}:{h}",
        "-frames:v", str(n_samples),
        "-pix_fmt", "rgb24",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
        return None
    frame_bytes = w * h * 3
    got = len(proc.stdout) // frame_bytes
    if got == 0:
        return None
    arr = np.frombuffer(proc.stdout[:got * frame_bytes], dtype=np.uint8)
    return arr.reshape(got, h, w, 3).copy()


def _build_features(fps: float, size: Tuple[int, int],
                     roles: List[str]) -> Dict:
    w, h = size
    video_info = {
        "video.fps": float(fps),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    features: Dict[str, Dict] = {
        "observation.state": {
            "dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES,
        },
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
    }
    for role in roles:
        features[CAMERA_KEYS[role]] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": dict(video_info),
        }
    return features


def write_dataset_meta(out_root: Path, episodes: List[Dict],
                        tasks: List[str], fps: float,
                        size: Tuple[int, int],
                        session_id: str,
                        img_focal: float,
                        roles: List[str]) -> None:
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = sum(int(e["length"]) for e in episodes)
    total_episodes = len(episodes)
    total_videos = total_episodes * len(roles)
    total_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE or 1
    mode = "head_only" if roles == ["head"] else "head_wrist"

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": None,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": ("data/chunk-{episode_chunk:03d}/"
                       "episode_{episode_index:06d}.parquet"),
        "video_path": ("videos/chunk-{episode_chunk:03d}/"
                        "{video_key}/episode_{episode_index:06d}.mp4"),
        "features": _build_features(fps, size, roles),
        # h2r-specific extras (LeRobot ignores unknown top-level keys).
        "h2r": {
            "session_id": session_id,
            "mode": mode,
            "cameras": roles,
            "hawor_version": BRIDGE_VERSION,
            "img_focal": float(img_focal),
            "state_layout": STATE_LAYOUT,
            "thumb_tip_idx": THUMB_TIP_IDX,
            "pinky_tip_idx": PINKY_TIP_IDX,
            "gripper_open_m": float(GRIPPER_OPEN_M),
            "gripper_closed_m": float(GRIPPER_CLOSED_M),
            "gripper_signal": (
                f"Linear normalization of ||thumb_tip - pinky_tip||: "
                f"{GRIPPER_OPEN_M} m -> 0.0 (open), "
                f"{GRIPPER_CLOSED_M} m -> 1.0 (closed), clipped to [0, 1]. "
                f"State dims 12/13."),
            "gripper_calibrated": False,
            "notes": ("Per-episode HaWoR-world frame (no cross-episode "
                       "homing). Real HaWoR via in-process import."),
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    with (meta_dir / "episodes.jsonl").open("w") as f:
        for e in episodes:
            f.write(json.dumps({
                "episode_index": int(e["episode_index"]),
                "tasks": [e["task"]],
                "length": int(e["length"]),
            }) + "\n")

    with (meta_dir / "tasks.jsonl").open("w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    stats = _aggregate_stats(out_root, total_episodes, roles=roles, size=size)
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cut-sheet", required=True,
                   help="clips.json (same file fed to sync_videos.py)")
    p.add_argument("--clips-dir", required=True,
                   help="dir with {name}_{role}.mp4 clips")
    p.add_argument("--out",
                   default=str(REPO_ROOT / "data" / "output" / "lerobot_v2"),
                   help="LeRobot dataset root (meta/, data/, videos/). "
                        "Default lives under the project's data/ folder so "
                        "nothing writes to /tmp.")
    p.add_argument("--hawor-repo", default=str(HAWOR_REPO),
                   help="path to HaWoR clone (default: %(default)s)")
    p.add_argument("--traj-fps", type=float, default=30.0,
                   help="trajectory + video fps (default 30 = HaWoR's fps)")
    p.add_argument("--image-size", default="1280x720",
                   help="WxH for observation frames (default %(default)s)")
    p.add_argument("--img-focal", type=float,
                   help="HaWoR focal in px (default: HaWoR's 600 fallback)")
    p.add_argument("--cache-dir",
                   help="default: <out>/cache (removed on successful finish)")
    p.add_argument("--keep-cache", action="store_true",
                   help="keep the HaWoR cache dir after a successful run "
                        "(default: delete it once the dataset is complete)")
    p.add_argument("--session-id",
                   help="session id for info.json (default: cut-sheet parent)")
    p.add_argument("--limit", type=int,
                   help="process at most N episodes (handy for first run)")
    args = p.parse_args(argv)

    cut_sheet = _user_path(args.cut_sheet)
    clips_dir = _user_path(args.clips_dir)
    out_root = _user_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = (_user_path(args.cache_dir) if args.cache_dir
                 else out_root / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    size = parse_size(args.image_size)
    hawor_repo = _user_path(args.hawor_repo)
    session_id = args.session_id or cut_sheet.parent.name

    episodes = load_episodes(cut_sheet, clips_dir)
    if args.limit:
        episodes = episodes[: args.limit]
    # Auto-detect head-only vs head+wrist from what load_episodes resolved.
    # Head-only when no episode produced any wrist clip; the camera set is
    # uniform across the dataset so the schema decision is dataset-wide.
    any_wrist = any(
        role in ep.clips
        for ep in episodes
        for role in ("wrist_left", "wrist_right")
    )
    roles = ["head", "wrist_left", "wrist_right"] if any_wrist else ["head"]
    mode = "head_wrist" if any_wrist else "head_only"

    print(f"Resolved {len(episodes)} episode(s) from {args.cut_sheet}")
    print(f"  mode:       {mode}  (cameras: {roles})")
    print(f"  out:        {out_root}")
    print(f"  cache:      {cache_dir}")
    print(f"  HaWoR repo: {hawor_repo}")
    print(f"  size:       {size[0]}x{size[1]}  traj_fps: {args.traj_fps}")
    print(f"  gripper:    ||thumb-pinky|| normalized {GRIPPER_OPEN_M} m -> 0, "
          f"{GRIPPER_CLOSED_M} m -> 1 (clipped)")
    print(f"  cuda:       {torch.cuda.is_available()}")

    # Build task table (dedup of task_label across episodes; "" stays valid).
    tasks: List[str] = []
    task_to_idx: Dict[str, int] = {}
    for ep in episodes:
        label = ep.task_label or ""
        if label not in task_to_idx:
            task_to_idx[label] = len(tasks)
            tasks.append(label)

    summaries: List[Dict] = []
    failures: List[Tuple[int, str]] = []
    global_index = 0  # running frame counter across the dataset
    last_img_focal: Optional[float] = None

    for ep_pos, ep in enumerate(episodes):
        episode_index = ep_pos  # 0-based; LeRobot convention.
        print(f"\n=== episode {episode_index:06d} (id={ep.episode_id:04d}) "
              f"'{ep.name}' [{ep.task_label or 'no-label'}] ===")
        try:
            hawor = run_hawor_or_cache(
                ep.head_clip, cache_dir, hawor_repo, args.img_focal)
            state = build_state_14d(hawor)
            action = derive_action_from_state(state)

            n_rows = write_episode_parquet(
                out_root, episode_index,
                task_index=task_to_idx[ep.task_label or ""],
                state=state, action=action,
                fps=args.traj_fps,
                global_index_offset=global_index)
            frame_counts = write_episode_videos(
                out_root, episode_index, ep.clips,
                fps=args.traj_fps, n_frames=n_rows, size=size)

            summaries.append({
                "episode_index": episode_index,
                "episode_id": ep.episode_id,
                "name": ep.name,
                "task": ep.task_label or "",
                "length": n_rows,
                "frames": frame_counts,
            })
            global_index += n_rows
            last_img_focal = float(hawor["img_focal"])

        except Exception as exc:  # noqa: BLE001 - isolate per-episode failures
            print(f"ERROR ep{ep.episode_id:04d} ({episode_index:06d}): {exc}",
                  file=sys.stderr)
            traceback.print_exc()
            failures.append((ep.episode_id, str(exc)))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if summaries:
        print("\n[meta] writing info.json / episodes.jsonl / tasks.jsonl / stats.json ...")
        write_dataset_meta(
            out_root,
            episodes=[{
                "episode_index": s["episode_index"],
                "task": s["task"],
                "length": s["length"],
            } for s in summaries],
            tasks=tasks,
            fps=args.traj_fps, size=size,
            session_id=session_id,
            img_focal=last_img_focal if last_img_focal is not None else 600.0,
            roles=roles)

    # Cache cleanup. The cache only exists to skip HaWoR on re-runs; once the
    # dataset is written it is disposable. Delete it on a fully-successful run
    # unless --keep-cache. On partial failure keep it so a re-run hits cache
    # for the episodes that already succeeded.
    if summaries and not failures and not args.keep_cache:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[cache] removed {cache_dir}")
    elif failures and cache_dir.is_dir():
        print(f"[cache] kept {cache_dir} ({len(failures)} episode(s) failed; "
              f"re-run will hit cache for the rest)")

    print("\n===== summary =====")
    for s in summaries:
        print(f"  ep{s['episode_index']:06d}  rows={s['length']:<5} "
              f"frames={s['frames']}  task='{s['task']}'")
    for eid, err in failures:
        print(f"  ep{eid:04d}  FAILED: {err}", file=sys.stderr)
    print(f"{len(summaries)} ok, {len(failures)} failed")
    return 1 if failures and not summaries else 0


if __name__ == "__main__":
    sys.exit(main())
