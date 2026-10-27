# Video Action Recognition on UCF50

Tian Liu

Action recognition on [UCF50](https://www.crcv.ucf.edu/data/UCF50.php) with LRCN models (a ResNet
frame encoder followed by an LSTM), evaluated on a strict group-held-out split. The final model
is a two-stream LRCN: one stream sees RGB frames, the other sees stacked differences between
consecutive frames (a cheap motion signal), and their predicted class probabilities are
averaged.

**Result: 93.88% test accuracy** (two-stream fusion), compared with 89.43% for the best
RGB-only LRCN.

---

## Contents

- [Video Action Recognition on UCF50](#video-action-recognition-on-ucf50)
  - [Contents](#contents)
  - [Results](#results)
  - [Experiment tracking](#experiment-tracking)
  - [Changes from the starter code](#changes-from-the-starter-code)
    - [Evaluation protocol and data](#evaluation-protocol-and-data)
    - [Models](#models)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Code quality](#code-quality)
  - [Method](#method)
    - [Evaluation protocol](#evaluation-protocol)
    - [Model](#model)
    - [RGB-difference (motion) stream](#rgb-difference-motion-stream)
    - [Fusion](#fusion)
  - [Setup](#setup)
  - [Data preparation](#data-preparation)
  - [Training](#training-1)
  - [Evaluation and fusion](#evaluation-and-fusion)
  - [Command-line options](#command-line-options)
  - [Tests](#tests)
  - [Project structure](#project-structure)
  - [Limitations](#limitations)

---

## Results

Test accuracy on 1,079 videos from held-out groups (single clip per video unless noted).

| Model | Input | Test accuracy |
|---|---|---|
| LRCN, ResNet34 | RGB, 16 frames | 86.65% |
| LRCN, ResNet50 | RGB, 16 frames | 89.25% (89.71% with 3-clip testing) |
| LRCN, ResNet50, LSTM 512 | RGB, 32 frames | 89.43% |
| LRCN, ResNet50, LSTM 512 | RGB difference, 32 positions x 1 | 89.34% |
| LRCN, ResNet50, LSTM 512 | RGB difference, 16 positions x 3 stacked | 91.38% |
| **Two-stream fusion** | **RGB (32 frames) + stacked differences** | **93.88%** |

The choice of motion stream and the fusion weight were selected on the validation split (equal
weights scored best); the test set was evaluated once with that choice.

Key observations:

- **RGB-only LRCN plateaued around 89.5%.** Changing the backbone, frame count, and LSTM size
  moved test accuracy by less than a point; the remaining errors were in fast, repetitive
  actions (JumpingJack, Nunchucks, Lunges, JugglingBalls) that look alike in single frames.
- **The motion stream fails on different classes.** It is strong on movement-defined classes
  (e.g. JumpingJack 33% -> 62%, Nunchucks 52% -> 74% versus RGB) and weaker on scene-defined
  ones (Biking, Drumming, MilitaryParade), so the two streams' errors barely overlap.
- **Fusion gives the largest gain** (+4.5 points over RGB alone): for each video, the stream
  suited to that class tends to be the more confident one.

---

## Experiment tracking

All training, evaluation, and fusion runs are logged to Weights & Biases:
[Video_Action_Recognition](https://wandb.ai/ttl074-johns-hopkins-university/Video_Action_Recognition)

- **Training runs** log per-epoch `train/loss`, `train/acc`, `val/loss`, `val/acc`, and the
  learning rate; each run's summary records `best_epoch` and `best_val_acc`, and its
  **Config** tab lists the exact arguments it was launched with.
- **Evaluation runs** log `test/acc` and `test/loss` (single clip), `test/acc_multiclip` and
  `test/loss_multiclip` (multi-clip), or `val_eval/...` when evaluating the validation split.
- **The fusion run** (`fusion_rgb_diff3`) logs the fused test and validation accuracy, the
  fusion weight and the probability files used, a per-class precision/recall/F1 table, the full
  50 x 50 confusion matrix, and the four `.npz` probability files as an artifact
  (`fusion_probs`).

---

## Changes from the starter code

The project started from an LRCN template (ResNet34 + LSTM, a fixed number of frames per video,
a stratified random split, Adam). The main changes, grouped by area:

### Evaluation protocol and data

- **Group-held-out split instead of a random split.** Clips from the same UCF50 group share the
  actor, background, and camera, so a random split leaks near-duplicate clips into the test set.
  Whole groups are now assigned to train, validation, or test (`dataset_split`), and a test
  checks that no group appears in more than one split.
- **All frames extracted, sampled per segment.** Every frame of each video is stored, and each
  clip takes one frame per equal-length temporal segment: at a random position in training
  (temporal jitter, so each epoch sees different frames) and at a fixed position in evaluation.
- **Clip-level augmentation.** Random crop, flip, and color jitter are drawn once per clip and
  applied to every frame; previously each frame was transformed independently.
- **Collate fix.** Empty clips are filtered together with their labels, so clips and labels
  can no longer fall out of alignment.

### Models

- **Stronger backbone and larger LSTM:** ResNet50 with ImageNet V2 weights and a 512-unit LSTM
  (from ResNet34 and 100 units), plus optional freezing of backbone stages, with frozen
  BatchNorm layers kept in eval mode.
- **RGB-difference motion stream:** a dataset variant that returns stacked differences of
  consecutive frames, and a widened first convolution initialized from the RGB weights.
- **Two-stream late fusion:** averaging the RGB and motion streams' probabilities, with the
  weight chosen on validation (+4.5 points over RGB alone).
- **R3D-18 option:** a 3D CNN initialized by inflating ImageNet ResNet18 weights along time.

### Training

- **AdamW** with decoupled weight decay, a `ReduceLROnPlateau` scheduler, early stopping on
  validation loss, checkpoint selection by validation accuracy, and reloading of the best
  weights when the learning rate drops.
- **Mixed precision** (`--amp`; bf16 or fp16 with loss scaling), about 2x faster training.
- **Resumable runs:** a `last.pt` checkpoint saved atomically every epoch (model, optimizer,
  scheduler, scaler, progress), with the original arguments and W&B run restored on resume.
- **Weights & Biases logging** of all runs.

### Evaluation

- **Multi-clip testing** (`--test_offsets`), validation-split evaluation (`--split val`), and
  saved probabilities (`--save_probs`) for fusion.
- Per-class classification reports and confusion matrices.

### Code quality

- A pytest suite covering the dataset, frame sampling, augmentation consistency, split leakage,
  and model behavior.
- Docstrings throughout, checked with pylint.

---

## Method

### Evaluation protocol

UCF50 clips are grouped (25 groups per class); clips in a group are cut from the same source
video, so they share the actor, background, and camera. Splitting clips at random leaks this
information between train and test and inflates accuracy. Here, **whole groups** are assigned to
one split, with the same group numbers held out for every class (seed 2026):

| Split | Groups |
|---|---|
| Train | 1 2 3 4 7 9 10 12 13 14 15 16 17 18 19 21 22 23 24 |
| Validation | 6 20 |
| Test | 5 8 11 25 |

Six classes have extra groups 26-30 (63 clips); these are excluded so every class has the same
25 groups.

### Model

- **LRCN:** each sampled frame goes through an ImageNet-pretrained ResNet (fc layer removed);
  the per-frame features are fed to an LSTM, and the last LSTM output passes through dropout
  and a linear classifier.
- **Temporal sampling:** each video is divided into equal segments and one frame is taken per
  segment: at a random position during training (temporal jitter), and at a fixed position
  (the center by default) for validation and testing.
- **Augmentation:** clip-level `RandomResizedCrop`, horizontal flip, and color jitter, with the
  same random parameters applied to every frame of a clip.
- **Training:** AdamW, `ReduceLROnPlateau` on validation loss, early stopping on validation
  loss, checkpoint chosen by validation accuracy, optional mixed precision (`--amp`), and
  resumable runs (`--resume`).

### RGB-difference (motion) stream

Subtracting consecutive frames cancels static content (background, still objects) and leaves
bright edges where something moved, an inexpensive stand-in for optical flow. At each sampled
position *i*, the stream loads frames *i* ... *i+K* and stacks the *K* consecutive differences
along the channel axis (`--diff_stack K`), so each time step covers a short motion window
(about 0.12 s for K = 3). The ResNet's first convolution is widened to 3K input channels and
initialized from its RGB weights, repeated K times and divided by K, so the pretrained layers
still receive inputs on the scale they expect.

### Fusion

Each stream is evaluated separately and its softmax probabilities are saved (`--save_probs`).
The fused prediction is the argmax of `w * p_rgb + (1 - w) * p_motion`, with `w` chosen on
validation probabilities.

---

## Setup

The project was developed on Google Colab (GPU runtime, High-RAM for more DataLoader workers).

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch, torchvision, OpenCV, scikit-learn, NumPy, Pillow, tqdm, Weights &
Biases, pytest. Training logs to the W&B project `Video_Action_Recognition`; pass `--no-wandb`
to disable logging.

The scripts import from `src/`, so set `PYTHONPATH=src` when calling `scripts/run.py` directly
(`train.sh` and `test.sh` do this for you).

---

## Data preparation

1. **Download and extract UCF50** (the script caches the `.rar` in Google Drive and reuses it):

   ```bash
   python scripts/prep_dataset.py --drive_path /content/drive/MyDrive/datasets/ucf50 \
       --local_path /content
   ```

2. **Extract frames** from every `.avi` into one folder of JPEGs per video. All frames are
   kept; videos shorter than `--n_frames` are padded by duplicating frames.

   ```bash
   python scripts/extract_frames.py --raw_dir /content/datasets/UCF50 \
       --n_frames 16 --frame_dir /content/datasets/UCF50_frames
   ```

The frames live on the runtime's local disk, so on Colab they must be re-extracted after every
runtime reset.

---

## Training

`scripts/train.sh` sets the dataset path, split sizes, number of classes, and batch size; any
extra arguments are passed through to `scripts/run.py` and override the defaults. Training
saves the splits to `splits.npy` and checkpoints to `models/<run_name>/` (`best_model_wts.pt`
and the resumable `last.pt`).

**RGB stream:**

```bash
bash scripts/train.sh --cnn_backbone resnet50 --dropout 0.3 --num_workers 7 --amp \
    --fr_per_vid 32 --rnn_hidden_size 512 --run_name r50v2_do03_fpv32_h512
```

**Stacked-difference stream:**

```bash
bash scripts/train.sh --cnn_backbone resnet50 --dropout 0.3 --num_workers 7 --amp \
    --fr_per_vid 16 --diff_stack 3 --rnn_hidden_size 512 --input_mode diff \
    --run_name r50v2_diff3_fpv16_h512
```

**Resuming** an interrupted run restores the original arguments, model, optimizer, scheduler,
and W&B run from `last.pt`:

```bash
bash scripts/train.sh --resume models/<run_name>/last.pt --num_workers 7
```

---

## Evaluation and fusion

`scripts/test.sh <checkpoint>` evaluates a checkpoint; extra arguments are passed through. The
model-defining flags (`--cnn_backbone`, `--rnn_hidden_size`, `--fr_per_vid`, `--input_mode`,
`--diff_stack`) must match the ones used in training. `--fr_per_vid` and `--input_mode` do not
raise an error when wrong; they silently produce a different (lower) accuracy.

Save validation and test probabilities for each stream:

```bash
# RGB stream
bash scripts/test.sh models/r50v2_do03_fpv32_h512/best_model_wts.pt --cnn_backbone resnet50 \
    --rnn_hidden_size 512 --fr_per_vid 32 --split val --save_probs probs/val_run6_rgb.npz
bash scripts/test.sh models/r50v2_do03_fpv32_h512/best_model_wts.pt --cnn_backbone resnet50 \
    --rnn_hidden_size 512 --fr_per_vid 32 --save_probs probs/run6_rgb.npz

# Stacked-difference stream
bash scripts/test.sh models/r50v2_diff3_fpv16_h512/best_model_wts.pt --cnn_backbone resnet50 \
    --rnn_hidden_size 512 --fr_per_vid 16 --input_mode diff --diff_stack 3 \
    --split val --save_probs probs/val_diff3.npz
bash scripts/test.sh models/r50v2_diff3_fpv16_h512/best_model_wts.pt --cnn_backbone resnet50 \
    --rnn_hidden_size 512 --fr_per_vid 16 --input_mode diff --diff_stack 3 \
    --save_probs probs/diff3.npz
```

The **Stream Fusion** section of `run.ipynb` loads these files, chooses the fusion weight on
validation, and reports the fused test accuracy and per-class report.

Each evaluation prints overall accuracy and loss, a per-class classification report, and
per-class (one-vs-rest) confusion matrices. `--test_offsets 0.25 0.5 0.75` enables multi-clip
testing (probabilities averaged over clips sampled at those positions within each segment).

---

## Command-line options

The most relevant options of `scripts/run.py` (see `python scripts/run.py --help` for all):

| Option | Purpose | Default |
|---|---|---|
| `--cnn_backbone` | ResNet backbone (`resnet18` ... `resnet152`) | `resnet34` |
| `--rnn_hidden_size`, `--rnn_n_layers` | LSTM size | 100, 1 |
| `--fr_per_vid` | Frames (time steps) sampled per video | 16 |
| `--input_mode` | `rgb` frames or `diff` frame differences | `rgb` |
| `--diff_stack` | Consecutive differences stacked per time step (`diff` only) | 1 |
| `--dropout` | Dropout before the classifier | 0.1 |
| `--learning_rate`, `--weight_decay` | AdamW settings | 3e-5, 0 |
| `--freeze_layers` | ResNet stages to freeze (weights and BatchNorm statistics) | none |
| `--n_epochs`, `--early_stop_patience` | Epoch limit and early stopping on val loss | 30, 10 |
| `--amp` | Mixed precision (bf16 on A100-class GPUs, fp16 with loss scaling otherwise) | off |
| `--resume` | Continue training from a `last.pt` checkpoint | none |
| `--test_offsets` | Clip positions within each segment at evaluation; more than one averages clips | 0.5 |
| `--split` | Split to evaluate: `test` or `val` | `test` |
| `--save_probs` | Save evaluated probabilities and labels to a `.npz` file | none |
| `--run_name`, `--wandb/--no-wandb` | W&B run name and logging | auto, on |

---

## Tests

```bash
python -m pytest -v -rs tests/
```

Tests that need the extracted UCF50 frames (frame counts, readable frames, no group leakage
between splits) are skipped when the data is not present; the others (frame sampling,
clip-level augmentation consistency, model batch independence, R3D weight inflation) run
anywhere.

---

## Project structure

```
├── run.ipynb               Colab notebook: setup, data, tests, training, evaluation, fusion
├── requirements.txt
├── pytest.ini
├── scripts/
│   ├── prep_dataset.py     Download/extract UCF50 (cached in Google Drive)
│   ├── extract_frames.py   Convert .avi videos to folders of JPEG frames
│   ├── run.py              Entry point for training and evaluation
│   ├── train.sh            Training wrapper (fixed dataset/split settings + extra args)
│   └── test.sh             Evaluation wrapper (checkpoint path + extra args)
├── src/
│   ├── video_datasets.py   Frame sampling, RGB and RGB-difference datasets, group split
│   ├── models.py           LRCN, conv1 widening for stacked differences, R3D-18
│   ├── train.py            Training loop, AMP, checkpointing and resume
│   ├── evaluate.py         Test loop, probabilities, reports, confusion matrices
│   └── utils.py            Frame extraction helpers, transforms, DataLoaders
└── tests/                  pytest suite
```

---

## Limitations

- Each configuration was trained once with a single seed, and the validation split has only 2
  groups, so differences under about one point are within noise.
- The stacked-difference run changed two settings at once (positions 32 -> 16, stack 1 -> 3),
  so its gain cannot be attributed to stacking alone.
- The two-stream model needs two forward passes per video.
- RGB differences are sensitive to camera motion and lighting changes; optical flow or
  video-pretrained (Kinetics) weights are natural next steps.