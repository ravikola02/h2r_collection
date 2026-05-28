# h2r_collection — human-demonstration-video → LeRobot datasets

Turn multi-camera **human demonstration** captures into
[LeRobot](https://github.com/huggingface/lerobot) v2.1 trajectory
datasets. A capture session (one head camera, optionally two wrist
cameras) is synced, cut into per-episode clips, run through
[HaWoR](https://github.com/ThunderVVV/HaWoR) for hand + camera pose, and
written out as a standard LeRobot dataset with a 14-d dual-hand state.

HaWoR is vendored as a git submodule. See [INSTALL.md](INSTALL.md) for
setup (conda env, weights, submodules).

---

## Quick start

```bash
# One command: tag episodes in a GUI, then convert to a LeRobot dataset.
python3 scripts/run_pipeline.py --session <session>
```

`<session>` is a folder name under `data/raw/`. Omit `--session` to use
the latest `video N` folder. The run is two stages:

1. **Tagger** — opens the head video, you scrub and mark episode
   in/out points, `Ctrl+S` saves the cut sheet and cuts the clips.
2. **Converter** — runs HaWoR on each head clip and writes the dataset.

Already have a cut sheet and clips? Skip the GUI and the sync step:

```bash
python3 scripts/run_pipeline.py --session <session> --skip-tag
```

---

## Input layout

Drop captures under `data/raw/<session>/`, one subdir per camera:

```
data/raw/<session>/
├── top_cam/      <one>.MP4   # head (required)
├── left_w_cam/   <one>.MP4   # wrist_left  (optional)
└── right_w_cam/  <one>.MP4   # wrist_right (optional)
```

Cameras are found by subdirectory, not filename — any single `.MP4`/`.MOV`
inside each is used.

## Output layout

Every output is scoped by session so captures never clobber each other:

```
data/output/<session>/
├── episode_clips/                # frame-exact per-episode triplets
│   └── {ep_name}_{head,wrist_left,wrist_right}.mp4
└── lerobot_v2/                   # LeRobot v2.1 dataset
    ├── meta/{info,episodes,tasks,stats}.json[l]
    ├── data/chunk-000/episode_NNNNNN.parquet
    └── videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4
```

The **cut sheet** `data/raw/<session>/clips.json` is the source of truth.
`episode_clips/` and `lerobot_v2/` are derived — delete and rebuild from
the cut sheet any time.

---

## The pipeline

### Stage 1 — tag + cut (`scripts/episode_tagger.py`)

Opens the raw head MP4, computes sync once on startup, and lets you mark
episodes. On `Ctrl+S` it writes `clips.json` **and** cuts the per-episode
clips with a frame-exact re-encode (stream-copy can't cut on arbitrary
frames). Re-cutting is cheap — edit and `Ctrl+S` again.

Hotkeys: `Space` play/pause · `←/→` ±1f · `Shift+←/→` ±10f ·
`Ctrl+←/→` ±1s · `Home/End` jump to overlap edges · `I/O` mark IN/OUT ·
`Enter` add episode · `Del` remove · `Ctrl+S` save + cut. The scrub bar
is clamped to the camera-overlap window so episodes can only be tagged
where all cameras have footage.

### Stage 2 — convert (`scripts/convert_to_lerobot.py`)

Runs **in the `hawor` conda env** (imports HaWoR directly). For each
episode it runs HaWoR end-to-end on the head clip, derives the 14-d
state, encodes one MP4 per camera, and writes one parquet per episode.
A HaWoR result cache under `lerobot_v2/cache/` speeds re-runs and is
removed once the dataset finishes successfully (`--keep-cache` to retain).

```bash
HAWOR_PY="${HAWOR_PYTHON:-$HOME/anaconda3/envs/hawor/bin/python}"
$HAWOR_PY scripts/convert_to_lerobot.py \
    --cut-sheet "data/raw/<session>/clips.json" \
    --clips-dir "data/output/<session>/episode_clips" \
    --out       "data/output/<session>/lerobot_v2"
```

`run_pipeline.py` does this for you with session-scoped defaults.

### Dataset schema

Per-row parquet: `observation.state` and `action` as
`fixed_size_list<float>[14]`, plus `timestamp`, `frame_index`,
`episode_index`, `index` (global), `task_index`. `action[t] = state[t+1]`
(last frame repeated). The 14-d state:

```
[ L wrist xyz (3), L wrist rpy (3),
  R wrist xyz (3), R wrist rpy (3),
  L gripper (1), R gripper (1) ]
```

Wrist pose is MANO joint 0 in the HaWoR-world frame; orientation is
intrinsic XYZ Euler. Gripper (dims 12/13) is a linear, clipped
normalization of `||thumb_tip − pinky_tip||` against hard-coded
open/closed distances (`GRIPPER_OPEN_M` / `GRIPPER_CLOSED_M` in
`scripts/convert_to_lerobot.py`). `meta/info.json` records the camera set
and mode under `h2r.cameras` / `h2r.mode`.

### Head-only vs head+wrist (auto-detected)

The mode is picked from which camera subdirs are populated:

- **head+wrist** — all three cameras present. Audio sync runs, three
  `observation.images.*` features in the dataset.
- **head-only** — only `top_cam/`. Sync is skipped, each episode is one
  `*_head.mp4`, the dataset has a single image feature.

The 14-d state is identical either way — HaWoR infers both hands from the
head video regardless of how many camera streams the dataset publishes.

---

## A note on video sync

When wrist cameras are present, the clips must share a timeline to
sub-frame precision (a one-frame error moves the hand by centimeters).
MP4 `creation_time` is only whole-second accurate (±0.5 s ≈ 15–30
frames), so it's used only as a coarse seed; the real alignment comes
from **audio cross-correlation**:

1. **Seed** from metadata `creation_time` deltas. If a file has no
   `creation_time`, fall back to stream tags, then file mtime (with a
   warning).
2. **Decode** mono 16 kHz audio for the head and each wrist.
3. **Cross-correlate** within ±1.5 s of the seed and take the peak lag —
   the sub-frame correction. The narrow window prevents spurious matches.
4. **Trust check** via a peak-to-sidelobe confidence. ≥ 6 → use the
   audio offset; below → fall back to metadata and warn. Never silently
   trust a weak correlation.

Exact rational fps (`60000/1001`, `30000/1001`) is used everywhere so
long sessions don't accumulate rounding drift. The result is written to
`data/raw/<session>/sync.json`.

Sync is **skipped** in two cases: head-only sessions (nothing to sync),
and sessions whose videos have no audio stream (xcorr can't run — falls
back to metadata offsets, method `metadata-no-audio`). When a cut sheet's
range drifts a few ms past the overlap window after a re-sync, it is
clamped into the window rather than rejected.

---

## Status / placeholders

The pipeline runs end-to-end today. Deliberate deferrals:

| Item | Today | When |
|:-----|:------|:-----|
| Gripper calibration | eyeballed open/closed distances, `gripper_calibrated: false` | after a grasp-validation study |
| Camera focal length | HaWoR default 600 px | once intrinsics are calibrated |
| Cross-episode homing | per-episode HaWoR-world frame | when multi-episode policy work needs it |
| Wrist-frame masking | raw downscaled clips | TBD |

---

## Acknowledgements

Hand and camera pose come entirely from
[**HaWoR**](https://github.com/ThunderVVV/HaWoR) (Zhang et al., *HaWoR:
World-Space Hand Motion Reconstruction from Egocentric Videos*, CVPR
2025). This project is a data-collection layer around it and claims no
credit for the underlying reconstruction. HaWoR in turn builds on
[HaMeR](https://github.com/geopavlakos/hamer),
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), and the
[MANO](https://mano.is.tue.mpg.de) hand model.

Most of this repository's tooling (the sync/tagging/conversion scripts
and docs) was built with [Claude Code](https://claude.com/claude-code),
Anthropic's agentic coding tool.

## License

This project's own code is released under the [MIT License](LICENSE).

**Important:** the pipeline depends on third-party models with their own,
more restrictive terms — HaWoR is
[CC-BY-NC-ND 4.0](HaWoR/license.txt) (non-commercial, no-derivatives) and
MANO is under the [MANO license](https://mano.is.tue.mpg.de/license.html)
(non-commercial). Running the full pipeline therefore binds you to those
non-commercial terms regardless of this repo's MIT license. The MIT
license covers the scripts in this repository only.
