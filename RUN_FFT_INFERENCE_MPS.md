# Run LaMa FFT inference on a new PNG folder — fastest method (Apple GPU / MPS)

Instructions for a future Claude session to run LaMa inpainting on an arbitrary
folder of PNGs using the fastest FFT-based path found in this repo. Full
benchmarking rationale is in `perf.md`; this file is the operational recipe.

## TL;DR

```bash
cd /Users/azhukov/git/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
./venv-latest/bin/python predict_mps.py \
  model.path=$(pwd)/lama-fourier \
  model.checkpoint=best.ckpt \
  device=mps \
  indir=/ABSOLUTE/PATH/TO/your_folder \
  outdir=$(pwd)/output-your-run
```

That's it if `venv-latest` and `lama-fourier` already exist (they do in this
repo). Everything below is why, prerequisites, and how to verify.

## What "fastest FFT-based method" means here

- **FFT-based** = the FFC / Fourier-convolution models (`ffc_resnet` generator):
  `lama-fourier` and `big-lama`. (The `-regular` models are plain ResNet, *not*
  FFT-based — don't use them if you specifically want the FFT architecture.)
- **Fastest device** = **MPS (Apple GPU) on torch ≥ 2.13**, fp32. torch 2.13
  added the MPS FFT kernel (`aten::_fft_r2c`); older torch (e.g. 2.2.2) has no
  MPS FFT and can only run these models on CPU (~12–13x slower).
- **Fastest model** = `lama-fourier` (9 FFC blocks): ~0.32 s/image on MPS vs
  `big-lama`'s (18 blocks) ~0.47 s/image. **If you want best quality instead of
  max speed, use `model.path=$(pwd)/big-lama`** — same command otherwise.
- **Do NOT use fp16.** Tested: no speedup on this workload and severe quality
  loss for FFC models (the fp32 FFT is the bottleneck-neutral part; fp16 wrecks
  the spectral magnitudes). fp32 on MPS is the sweet spot. See `perf.md`.

## Prerequisites (already satisfied in this repo)

1. **venv-latest** — Python 3.11 + torch 2.13.0 (has MPS FFT). Verify:
   ```bash
   ./venv-latest/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
   # expect: 2.13.0 True
   ```
   If missing, recreate with uv (Python 3.11 required; torch 2.13 has no py3.9 wheel):
   ```bash
   uv venv --python 3.11 venv-latest
   uv pip install --python venv-latest/bin/python torch torchvision
   uv pip install --python venv-latest/bin/python pyyaml tqdm easydict scikit-image \
     scikit-learn opencv-python joblib matplotlib pandas albumentations \
     "hydra-core>=1.3,<1.4" pytorch-lightning tabulate kornia webdataset packaging \
     wldhx.yadisk-direct
   ```

2. **Model weights** — `./lama-fourier/` (and/or `./big-lama/`), each containing
   `config.yaml` and `models/best.ckpt`. If absent, extract from the zips:
   ```bash
   unzip -o -q models/LaMa_models.zip "LaMa_models/lama-places/lama-fourier/*" -d /tmp/lamafx
   mv /tmp/lamafx/LaMa_models/lama-places/lama-fourier ./lama-fourier
   # big-lama: unzip -o models/big-lama.zip -d .
   ```

3. **`predict_mps.py`** at repo root — a copy of `bin/predict.py` that honors
   `device=mps` (bin/ is read-only here). Verify it exists; it differs from
   `bin/predict.py` only in the device line and hydra `config_path`.

4. **Source patches** for the torch-2.13 / py3.11 / numpy-2 stack (already applied
   in this working copy; re-apply if starting from a fresh clone):
   - `saicinpainting/training/data/aug.py`: guard the `DualIAATransform`/`imgaug`
     imports (removed in albumentations≥1.0; training-only, unused at inference).
   - `saicinpainting/training/trainers/__init__.py`: `torch.load(..., weights_only=False)`
     (torch≥2.6 default flipped to True; the checkpoint pickles PL objects).
   - `saicinpainting/training/modules/ffc.py`: fp32 upcast around the FFT — a
     no-op for fp32 runs, harmless.

## Input folder requirements — IMPORTANT

LaMa is *inpainting*: it needs a **mask per image** (the region to fill). A folder
of images alone is not enough. The default evaluation dataset pairs files like so:

- **Masks**: every file matching `**/*mask*.png` (recursive) is treated as a mask.
- **Image for a mask**: the mask filename with everything from `_mask` onward
  stripped, plus `.png`. So the folder must contain both:
  ```
  your_folder/
    photo1.png            # the image
    photo1_mask.png       # its mask (white = region to inpaint, on black)
    photo2.png
    photo2_mask.png
  ```
  `photo1_mask.png` → image `photo1.png`. Masks are `.png` regardless of image
  suffix. **Images without a matching mask are silently ignored.** The run
  processes N = number of masks.

- Masks are read as grayscale and thresholded `>0`, so white(255)=fill region,
  black(0)=keep. Mask must be the same H×W as its image.

If your folder has **only images (no masks)**, you must supply masks first. For
synthetic/benchmark masks you can generate them (note: this also resizes/crops the
images):
```bash
./venv-latest/bin/python bin/gen_mask_dataset.py \
  $(pwd)/configs/data_gen/random_medium_512.yaml \
  /path/to/images_only /path/to/your_folder --ext png
```
Otherwise create/paint masks yourself following the `_mask.png` naming above.

If images are not `.png`, add `dataset.img_suffix=.jpg` (masks stay `*mask*.png`).

## Run

```bash
cd /Users/azhukov/git/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
./venv-latest/bin/python predict_mps.py \
  model.path=$(pwd)/lama-fourier \
  model.checkpoint=best.ckpt \
  device=mps \
  indir=/ABSOLUTE/PATH/TO/your_folder \
  outdir=$(pwd)/output-your-run
```
Outputs land in `outdir` as `<image_name>.png` (inpainted result), preserving
subfolder structure relative to `indir`.

### Hydra gotcha
`device` is a real config key, so `device=mps` works. Any key NOT already in
`configs/prediction/default.yaml` must be *appended* with `+`, e.g. `+dtype=fp16`
(don't — see above). `model.path`, `model.checkpoint`, `indir`, `outdir`,
`dataset.img_suffix` are all existing keys → no `+`.

## Alternative: running the non-FFT (ResNet) models

The `-regular` models (`big-lama-regular`, `lama-regular`) use a plain
`pix2pixhd_global` ResNet generator — **no FFT**. Use these when you don't need
the FFC architecture, or when you're on a torch that lacks the MPS FFT kernel.

Why you might use them:
- **They run on MPS on *any* torch**, including `venv222` (torch 2.2.2) — there's
  no `aten::_fft_r2c` dependency. So they're the MPS option if you can't/don't
  want the torch-2.13 `venv-latest`.
- Marginally faster than the same-depth FFC model on CPU; comparable on MPS.

Trade-off: these are the paper's ablation baseline and produce **softer fills on
large/wide masks** than the FFC models. For best quality prefer
`lama-fourier`/`big-lama`.

Which `-regular` to pick:
- `big-lama-regular` — 18 blocks, 501 MB, best of the ResNet models.
- `lama-regular` — 9 blocks, 388 MB, ~20–25% faster, a bit softer.

Input-folder requirements, output layout, and GPU-verification below are all
identical to the FFC recipe (same `predict_mps.py`, same `_mask.png` pairing).

### fp32 (recommended — works on any torch)

```bash
cd /Users/azhukov/git/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
./venv-latest/bin/python predict_mps.py \
  model.path=$(pwd)/big-lama-regular \
  model.checkpoint=best.ckpt \
  device=mps \
  indir=/ABSOLUTE/PATH/TO/your_folder \
  outdir=$(pwd)/output-your-run
```
Swap `venv-latest` → `venv222` if that's all you have — these models don't need
MPS FFT, so torch 2.2.2 runs them on MPS fine. Everything else identical.

### fp16 — DO use autocast, do NOT use `+dtype=fp16`

Unlike the FFC models (where fp16 is useless), the ResNet models get a real
**~1.35x speedup from autocast fp16 with no quality loss** — but *only* via the
autocast path, and *only* on **torch ≥ 2.4** (so `venv-latest`, not `venv222`;
2.2.2 raises `unsupported autocast device_type 'mps'`).

- ✅ **autocast** — use `predict_mps_amp.py` (weights stay fp32, conv/matmul run
  fp16). Quality-neutral: inside-mask MAD ~0.2/255 vs fp32.
  ```bash
  ./venv-latest/bin/python predict_mps_amp.py \
    model.path=$(pwd)/lama-regular model.checkpoint=best.ckpt device=mps \
    indir=/ABSOLUTE/PATH/TO/your_folder outdir=$(pwd)/output-your-run-fp16
  ```
- ❌ **blunt `+dtype=fp16`** on `predict_mps.py` — runs, but the fill **collapses
  to gray garbage inside the mask** (fp16 BatchNorm catastrophic cancellation).
  Do not use. Full analysis in `perf.md`.

### Model weights

If a `-regular` folder is missing, extract from the zip:
```bash
unzip -o -q models/LaMa_models.zip "LaMa_models/lama-places/big-lama-regular/*" -d /tmp/lamareg
mv /tmp/lamareg/LaMa_models/lama-places/big-lama-regular ./big-lama-regular
# lama-regular: same command with .../lama-places/lama-regular/
```

## Verify it actually used the GPU (no CPU fallback)

Run **without** the fallback env var; if it completes, every op (incl. FFT) ran
natively on MPS:
```bash
unset PYTORCH_ENABLE_MPS_FALLBACK   # ensure it's not set
# ...run the command above...
echo "outputs: $(ls output-your-run | wc -l)"   # should equal number of masks
```
If it errors with `aten::... not implemented for MPS`, your torch is too old —
check step 1 (need torch ≥ 2.13), or set `PYTORCH_ENABLE_MPS_FALLBACK=1` to limp
along on CPU for the unsupported op (slow; defeats the purpose).

Optional correctness check vs CPU (should be ~identical, max pixel diff ~1/255):
```bash
# re-run with device=cpu to outdir=output-your-run-cpu, then compare a few files
```

## Expected performance (Apple M5 Max, from perf.md)

| Model | MPS (torch 2.13, fp32) | CPU | Speedup |
|-------|------------------------|-----|---------|
| lama-fourier (9-block FFC) | ~0.32 s/image (~3.1 it/s) | ~4.2 s/image | ~13x |
| big-lama (18-block FFC)    | ~0.47 s/image (~2.1 it/s) | ~5.6 s/image | ~12x |

First run is a few seconds slower due to one-time Metal shader compilation; warm
runs hit the numbers above. Per-image time scales with image resolution.

## Do not

- Use `venv190` (torch 1.9.0 — no working FFT at all) or `venv222` (torch 2.2.2 —
  no MPS FFT) for MPS FFC inference. Both are CPU-only for these models.
- Use `+dtype=fp16` (blunt `model.half()`): useless for FFC models and *broken*
  for ResNet models (gray-fill collapse). For ResNet fp16, use the autocast path
  (`predict_mps_amp.py`) instead — see "Alternative: running the non-FFT models".
- Point `model.path` at a `-regular` model if you specifically want FFT — those
  are plain ResNet (but they're the right choice if you *don't* need FFT; see
  that section).
