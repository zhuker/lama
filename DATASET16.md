# dataset16 — reprojection inpainting dataset (access + evaluation guide)

Guide for a future Claude session to **access, run inpainting on, and evaluate**
the `dataset16` reprojection dataset. Companion to `RUN_FFT_INFERENCE_MPS.md`
(how to run LaMa on MPS) and `predict_batch_warped_mps.py` (the batch runner).

## Location

```
/Users/azhukov/projects/hybrid-reproj/dataset16/
```

## What it is

A **synthetic forward-reprojection test set for inpainting**, meant to be judged
**by eye — there is no quantitative ground truth.** Each sample starts from a
single RGB frame (`_src`) and its depth (`_depth`) and **forward-reprojects** that
frame into a different camera view, producing `_warped` (the reprojected image,
with disocclusion holes) and `_mask` (the holes to fill).

The frames are a real sequence — **id N is frame N**, and `<id>_meta.json` records
`src_frame=N`, `tgt_frame=N+16`, `K=16` — but the **src→tgt reprojection pose is
synthetic** (constructed to generate a disocclusion test, not a calibrated
capture). Don't trust the poses as calibration.

`_ref` is the real frame **N+16** (verified pixel-identical to `_src` of id N+16),
*not* a render of the synthetic target pose. So even though it exists, **there is
no true per-pixel target for the holes**: the fill should be a plausible
completion of frame N's content at the warped pose, whereas frame N+16 is a
different moment with its own scene/camera motion. Evaluation is qualitative
(visual) — see below.

## File layout

Files are flat in the folder, named `<id>_<type>.png` (+ `<id>_meta.json`).
`<id>` is a zero-padded 5-digit string (e.g. `00138`). Each id has 6 files:

| suffix | mode | meaning |
|---|---|---|
| `_warped.png` | RGB 1280×720 | **model INPUT** — source frame forward-reprojected into a new view, with holes |
| `_mask.png`   | L 1280×720   | **model INPUT** — the holes to inpaint. `>0` = fill (white), `0` = keep (black) |
| `_ref.png`    | RGB 1280×720 | ⚠️ **Frame N+16 — NOT ground truth.** Verified pixel-identical to `_src` of id **N+16** (`K=16`): it's the real later frame in the sequence, *not* a target render of the reprojection. It roughly matches the warped view outside the holes, but is **not** a valid per-pixel inpainting target — the warp pose is synthetic and 16 frames of scene/camera motion mean the disoccluded content won't match any plausible fill. Use only as a loose visual reference; do **not** score against it. |
| `_src.png`    | RGB 1280×720 | the original source RGB (pre-reprojection); input to the warp, not to the model |
| `_depth.png`  | L 1280×720   | source depth used to drive the forward reprojection |
| `_meta.json`  | —            | **synthetic** frame indices + camera poses (made up; not real captures) |

**For LaMa inpainting you only need `_warped.png` (image) + `_mask.png` (mask).**
`_src`/`_depth` are the reprojection inputs; `_meta.json` is synthetic; and
`_ref.png` is misleading (see below). There is **no** file to evaluate against —
results are judged by eye.

## Key facts

- **4307** `_warped`/`_mask` pairs; every id has exactly one of each.
- All images are **1280×720**. Both dims are divisible by 8, so LaMa's
  `pad_out_to_modulo=8` is a **no-op** — output comes back exactly 1280×720.
- Mask semantics: grayscale, thresholded `>0` → white = region to inpaint.

### ⚠️ Pairing quirk (why the stock LaMa dataset fails)

LaMa's default `InpaintingDataset` derives the image from a mask by stripping
`_mask` and appending the image suffix → it looks for `<id>.png`, which does
**not exist here** (the image is `<id>_warped.png`). So a plain
`predict.py`/`predict_mps.py` run finds **zero** images. Use
`predict_batch_warped_mps.py` (repo root), which pairs `<id>_mask.png` ↔
`<id>_warped.png` explicitly.

### ⚠️ Mask ranges

- **Valid masks start at id `00138`.** Ids below `00138` have **degenerate**
  masks (all-zero → output == input passthrough, or full-frame → whole image
  regenerated). Filter to `id >= "00138"` for meaningful evaluation.
- Even in the valid range, ~20% of masks are **empty** (all-zero) → those are
  trivial passthroughs; the runner handles them fine but they carry no signal.
- Real masks are **large and scattered**: median ~36% of pixels filled, and the
  mask bounding box spans ~the full frame in ~80% of images. Consequence:
  **crop-to-mask optimizations don't help** (bbox ≈ whole frame), and quality
  differences between models show up most on these large holes.

## How to run inpainting

Use the batch runner (see its header + `RUN_FFT_INFERENCE_MPS.md` for details).
It scans `*_mask.png`, pairs each with `*_warped.png`, and ignores everything
else. Big-lama (FFT/FFC) on MPS, fp32:

```bash
cd /Users/azhukov/git/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
./venv-latest/bin/python predict_batch_warped_mps.py \
  --indir  /Users/azhukov/projects/hybrid-reproj/dataset16 \
  --outdir $(pwd)/output-dataset16-biglama \
  --model  $(pwd)/big-lama --device mps
```

Outputs land in `--outdir` as `<id>.png` (inpainted, 1280×720). Swap
`--model $(pwd)/lama-fourier` for the faster/smaller FFC model, or a
`-regular` model for the non-FFT ResNet variant (softer on these large masks).
`--limit N` processes only the first N pairs (smoke test).

## How to evaluate

**By eye. There is no per-pixel ground truth.** The disoccluded regions the model
fills were never observed, so no file in the dataset is a valid target for them —
`_ref.png` looks like one but is **not** (see the file table). **Do not compute
PSNR/SSIM/LPIPS against `_ref`**; those numbers are meaningless here.

Judge quality qualitatively: open the inpainted `<id>.png` next to `<id>_warped.png`
(and its `_mask`) and look at the filled holes — do they blend seamlessly, respect
edges/structure and continue textures, and avoid blur/ghosting/repetition? Focus on
the larger, harder masks (`id >= 00138`, higher fill %), since below `00138` masks
are degenerate.

If you need an *aggregate* signal to compare models, use a **no-reference /
distributional** measure — e.g. **FID** over the set of outputs, or a no-reference
image-quality model — never a per-pixel metric against `_ref`. Even then, a visual
spot-check on the large-mask samples is the primary judgment.

## Environment notes

- `venv-latest` (torch 2.13, MPS FFT) is required for the FFT models on MPS; see
  `RUN_FFT_INFERENCE_MPS.md`.
- Big-lama on MPS is ~180 ms/image at 720p (fp32); cost scales ~linearly with
  pixel count (~190 ms/megapixel), and is dominated by convolution, not the FFT.
