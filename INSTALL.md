# Installation

This pipeline turns multi-camera capture sessions into LeRobot v2.1
datasets. It is a thin layer of scripts around **HaWoR** (hand + camera
pose), which is vendored as a git submodule.

## 1. Prerequisites

- Linux with an NVIDIA GPU (HaWoR + DROID-SLAM need CUDA; tested with
  CUDA 11.7 / PyTorch 1.13).
- `git`
- A conda/mamba install (Miniconda or Anaconda).
- `ffmpeg` and `ffprobe` on `PATH` (system packages, e.g.
  `sudo apt install ffmpeg`). The converter prefers `/usr/bin/ffmpeg`
  over a snap-sandboxed one automatically.

## 2. Get the code

HaWoR lives at `HaWoR/` as a submodule of this repo, pointing at the
public upstream `https://github.com/ThunderVVV/HaWoR.git`. HaWoR has its
own nested submodules, so clone recursively:

```bash
git clone --recursive <this-repo-url> h2r_collection
cd h2r_collection
```

Already cloned without `--recursive`? Pull the submodules:

```bash
git submodule update --init --recursive
```

## 3. Create the HaWoR conda environment

The environment is named `hawor` and is defined by HaWoR's own
`requirements.txt`. The whole pipeline (sync, tagger, and the LeRobot
converter) runs in this single environment.

```bash
conda create --name hawor python=3.10 -y
conda activate hawor

# PyTorch 1.13 + CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# HaWoR's Python deps
pip install -r HaWoR/requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0
```

### Extra deps for this pipeline

Beyond HaWoR's requirements, this repo needs:

```bash
# Parquet writer (LeRobot dataset) + tagger GUI image handling
pip install pyarrow pillow
```

The episode tagger is a Tkinter GUI, so the Python interpreter needs the
`tk` bindings. If `python -c "import tkinter"` fails, install the system
package (e.g. `sudo apt install python3-tk`) or
`conda install tk`.

## 4. Build masked DROID-SLAM

```bash
cd HaWoR/thirdparty/DROID-SLAM
python setup.py install
cd -
```

## 5. Download model weights

HaWoR needs several checkpoints. Run from the `HaWoR/` directory:

```bash
cd HaWoR

# DROID-SLAM weights -> ./weights/external/droid.pth
#   (Google Drive link in HaWoR/README.md "Install masked DROID-SLAM")

# Metric3D weights -> thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
#   (Google Drive link in HaWoR/README.md "Install Metric3D")

# Detector + HaWoR checkpoints (public, direct download):
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
    -P ./weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt \
    -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt \
    -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml \
    -P ./weights/hawor/

cd -
```

### MANO hand model (manual, license-gated)

Create an account at <https://mano.is.tue.mpg.de>, download the models
(`mano_v*_*.zip`), and place the hand model files:

- `HaWoR/_DATA/data/mano/MANO_RIGHT.pkl`
- `HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl`

MANO is covered by its own [license](https://mano.is.tue.mpg.de/license.html).

The Google-Drive-hosted weights (DROID-SLAM, Metric3D) can't be `wget`'d
directly; the exact links live in `HaWoR/README.md`. HaWoR also provides
`HaWoR/scripts/setup_hawor_env.sh` if you'd rather run their bundled
setup.

## 6. Verify

```bash
conda activate hawor

# Pipeline deps import cleanly
python -c "import torch, cv2, numpy, scipy, pandas, pyarrow, PIL, tkinter; \
print('pipeline deps OK; cuda:', torch.cuda.is_available())"

# HaWoR imports resolve (run from the HaWoR dir for its relative paths)
cd HaWoR && python -c "from hawor.utils.process import run_mano; print('HaWoR OK')"; cd -

# ffmpeg present
ffmpeg -version | head -1
```

A quick end-to-end smoke test of the converter (needs a session with cut
clips already present):

```bash
python scripts/run_pipeline.py --session <session> --skip-tag
```

## 7. Data layout

Drop captures under `data/raw/<session>/` with one subdir per camera:

```
data/raw/<session>/
├── top_cam/      <one>.MP4   # head (required)
├── left_w_cam/   <one>.MP4   # wrist_left  (optional)
└── right_w_cam/  <one>.MP4   # wrist_right (optional)
```

Head-only sessions (just `top_cam/`) are supported and auto-detected.
See [README.md](README.md) for the run instructions.
