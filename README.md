# Automating Blood Pressure Readings with Ultrasound

Software prototype for a blood pressure device that reads cuff pressure alongside an
ultrasound video of an artery and vein. A vessel collapses when the cuff pressure exceeds
the pressure inside it, so the pressure at which each vessel disappears and reappears
gives the venous, diastolic and systolic pressures.

## Repository layout

| Path | Contents |
|---|---|
| `*.py` (top level) | Acquisition and early experiments: DICOM / video readers, Vernier cuff readers, `record_mirroring_labquest.py` (records ultrasound + pressure on one clock), `pressure_reader.py` (aligns older recordings by the flex spike). |
| `data/` | Recordings: ultrasound videos, pressure CSVs, manual event marks. |
| `vessel_ml/` | Vessel segmentation pipeline (labelling, training, inference). See below. |
| `data/vessel_dataset/masks/` | Artery / vein masks produced by SAM 2 from the clicks in `vessel_ml/prompts.json`. |
| `vessel_ml/checkpoints/unet_*.pt` | Trained U-Net weights. |

## Installation (vessel_ml pipeline)

Tested on Linux (WSL2) with Python 3.12 and an NVIDIA GTX 1660 SUPER.

```bash
python3 -m venv .venv-linux
source .venv-linux/bin/activate
pip install --upgrade pip

# 1. PyTorch. Pick the build that matches your NVIDIA driver's CUDA version
#    (check with `nvidia-smi`). The default wheel targeted a newer CUDA than this
#    machine's driver supported, so the CUDA 12.8 build was used here:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#    No GPU: `pip install torch torchvision` works, but SAM 2 labelling will be very slow.

# 2. Everything else
pip install -r vessel_ml/requirements.txt

# 3. Check the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`ffmpeg` must be on the PATH for `visualize.py` (`sudo apt install ffmpeg`).

Only needed if you re-run SAM 2 labelling: download the SAM 2.1 "small" weights (176 MB,
not stored in git).

```bash
mkdir -p vessel_ml/checkpoints
curl -L -o vessel_ml/checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

The top-level `requirements.txt` is an older `pip freeze` from the macOS machine used for
acquisition. It was not used or checked for the `vessel_ml` pipeline.

## Using the vessel_ml pipeline

Run from the repository root with the environment activated.

```bash
# 1. Crop every video in data/ to the ultrasound image and dump frames
#    (writes data/vessel_dataset/frames/, not stored in git)
python vessel_ml/prep_frames.py

# 2. Review / correct labels in the browser at http://localhost:8765
#    (clicks are saved to vessel_ml/prompts.json; the tool can re-run SAM 2 per video)
python vessel_ml/label_tool.py

#    ...or regenerate masks for specific videos from the saved clicks
python vessel_ml/sam2_label.py 051426_Lee3

# 3. Train, holding out whole videos for testing
python vessel_ml/train_unet.py --val 051426_Lee3 --tag holdout_lee3

# 4. Run a trained model: area curves, detected pressures, plot against cuff pressure
cd vessel_ml
python infer.py --ckpt checkpoints/unet_holdout_lee3.pt 051426_Lee3

# 5. Side-by-side review video (raw | U-Net | label)
python visualize.py --ckpt checkpoints/unet_holdout_lee3.pt phillips_no_color
```

`python vessel_ml/classical_baseline.py` runs a non-ML tracker for comparison.

Step 1 is required after a fresh clone: the cropped frames are generated files, and the
masks in git refer to them by video name and frame number.
