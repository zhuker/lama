# LaMa inference performance

## Summary

All figures on the same 47 `LaMa_test_images` (mixed resolutions up to ~2k),
Apple M5 Max / 128 GB / macOS 26.6. FFC models need torch ≥2.13 for MPS (FFT
kernel); ResNet models run on MPS with any torch. fp16 = MPS autocast (fp32
weights), quality-neutral for ResNet only. See sections below for details.

| Model | Arch | blocks | ckpt | CPU total | CPU s/img | MPS total | MPS s/img | Speedup | fp16 MPS |
|-------|------|:------:|------|-----------|:---------:|-----------|:---------:|:-------:|----------|
| big-lama | FFC | 18 | 410 MB | ~261–270 s | ~5.75 | ~22 s (warm) | ~0.47 | ~12x | ~21 s (no gain) |
| big-lama-regular | ResNet | 18 | 501 MB | 238 s | ~5.07 | ~15.7 s | ~0.33 | ~15x | ~11.5 s |
| lama-fourier | FFC | 9 | 314 MB | ~194–199 s | ~4.25 | ~15 s (warm) | ~0.32 | ~13x | — |
| lama-regular | ResNet | 9 | 388 MB | ~204 s | ~4.35 | ~12.6 s | ~0.27 | ~16x | ~8.8 s (~23x) |

Key takeaways:
- **MPS gives ~12–16x over CPU** across all four models (fp32).
- **FFC fp16 is pointless** (no speedup, precision loss); **ResNet autocast fp16
  is quality-neutral and ~1.35x faster** — lama-regular fp16 hits ~23x vs CPU.
- MPS output matches CPU to within 1/255 for the FFC models.

## Run command

```bash
cd /Users/azhukov/git/lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
./venv222/bin/python bin/predict.py \
  model.path=$(pwd)/big-lama \
  indir=$(pwd)/LaMa_test_images \
  outdir=$(pwd)/output
```

Runs on CPU: `bin/predict.py` hardcodes `device = torch.device("cpu")`. For the
default **big-lama** (FFC) model the Apple GPU (MPS) cannot be used **on torch
2.2.2** — its Fourier-convolution layers call `torch.fft.rfftn`, and torch
2.2.2's MPS backend has no FFT kernel (`aten::_fft_r2c is not currently
implemented for the MPS device`).

> **Update (torch 2.13.0):** this MPS FFT gap is fixed in newer torch. On
> `venv-latest` (Python 3.11 + torch 2.13.0) the FFC models run fully on MPS with
> a ~13x speedup. See the "torch 2.13.0 — MPS FFT works" section below. torch
> 2.13.0 requires Python ≥3.10; the max torch for Python 3.9 is 2.8.0.

The **big-lama-regular** (ResNet) model has no FFT and *does* run on MPS — see
that section below for a ~15x speedup. To enable it, `bin/predict.py`'s hardcoded
device must be made configurable. Since `bin/` is read-only in this environment,
a copy lives at `predict_mps.py` (repo root) with two changes vs `bin/predict.py`:
`device = torch.device(predict_config.get('device', 'cpu'))` and hydra
`config_path='configs/prediction'`. Run it with `device=mps`.

## Environment

- venv: `venv222` (separate from the torch-1.9.0 `venv190`, which cannot run
  LaMa at all — its arm64 build has no working FFT: `fft: ATen not compiled
  with MKL support`)
- Python 3.9.6
- torch 2.2.2, torchvision 0.17.2
- numpy 1.26.4 (pinned <2 for torch 2.2.2)
- pytorch-lightning 2.2.2 (1.2.9 breaks on torch 2.x: imports removed `torch._six`)
- Device: CPU

## Hardware

- Machine: MacBook Pro (Mac17,6)
- Chip: Apple M5 Max (18 cores: 6 efficiency + 12 performance)
- RAM: 128 GB
- OS: macOS 26.6 (build 25G72), arm64

## Numbers

Model: big-lama (`big-lama/models/best.ckpt`).
Dataset: `LaMa_test_images` — 47 image/mask pairs, mixed resolutions up to ~2k.

| Metric | Value |
|--------|-------|
| Total wall time | ~4:30 (270 s) for 47 images |
| Average | ~5.75 s/image |
| Per-image range | ~3.1 – 7.8 s/image |

### big-lama-regular (ResNet, no FFT)

Model: `big-lama-regular` (`pix2pixhd_global` generator, extracted from
`models/LaMa_models.zip` → `LaMa_models/lama-places/big-lama-regular/`).
Dataset: same `LaMa_test_images` — 47 image/mask pairs.

Run command (differs only in `model.path` / `model.checkpoint` / `outdir`):

```bash
./venv222/bin/python bin/predict.py \
  model.path=$(pwd)/big-lama-regular \
  model.checkpoint=best.ckpt \
  indir=$(pwd)/LaMa_test_images \
  outdir=$(pwd)/output-big-lama-regular
```

| Metric | CPU | MPS |
|--------|-----|-----|
| Total wall time | 238 s (3:58) | ~15.7 s |
| Average | ~5.07 s/image | ~0.33 s/image (~3.0 it/s) |
| Per-image range | ~2.8 – 6.5 s/image | — |
| Speedup vs CPU | 1x | **~15x** |

MPS command (via `predict_mps.py`, output to `output-big-lama-regular-mps/`):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
./venv222/bin/python predict_mps.py \
  model.path=$(pwd)/big-lama-regular \
  model.checkpoint=best.ckpt \
  device=mps \
  indir=$(pwd)/LaMa_test_images \
  outdir=$(pwd)/output-big-lama-regular-mps
```

Notes:
- Ran cleanly on **CPU** with no FFT/MKL error — this is why the ResNet variant
  works where big-lama's Fourier layers matter for the `venv190` failure.
- Also ran cleanly on **MPS**: no ops fell back to CPU (the
  `PYTORCH_ENABLE_MPS_FALLBACK=1` guard was set but not needed), ~15x faster than
  CPU. All layers (Conv2d, ConvTranspose2d, BatchNorm, ReflectionPad, ReLU,
  Sigmoid) have native MPS kernels.
- On CPU, slightly faster than big-lama (~5.07 vs ~5.75 s/image), but this is the
  ablation baseline and produces softer fills on large/wide masks.

### lama-regular (small ResNet, 9 blocks)

Model: `lama-regular` (`pix2pixhd_global`, `n_blocks: 9` — half of
big-lama-regular's 18; `n_downsampling: 3`, sigmoid out), extracted from
`models/LaMa_models.zip` → `LaMa_models/lama-places/lama-regular/`. Checkpoint
~388 MB vs big-lama-regular's 501 MB. It is the ResNet analog of lama-fourier:
same 9-block depth, plain convs instead of FFC. Dataset: same 47
`LaMa_test_images`.

| Metric | CPU (venv222) | MPS fp32 | MPS autocast fp16 |
|--------|---------------|----------|-------------------|
| Total wall time | ~204 s (3:24) | ~12.6 s | ~8.8 s |
| Average | ~4.35 s/image | ~0.27 s/img (3.74 it/s) | ~0.19 s/img (5.31 it/s) |
| Speedup vs CPU | 1x | **~16x** | **~23x** |

MPS runs on `venv-latest` (torch 2.13.0); no FFT, so any torch works.

Notes:
- autocast fp16 is quality-safe here (see the ResNet fp16 subsection below):
  inside-mask MAD 0.15/255 vs fp32.
- The CPU figure briefly overlapped the two MPS runs at launch, so ~204 s is a
  soft upper bound.
- ~20–25% faster than big-lama-regular across all devices, consistent with half
  the residual blocks. Same ResNet family, so expect similarly soft fills on
  large masks vs the FFC models — from a shallower generator.

### ResNet models: fp16 on MPS — autocast works, blunt `.half()` collapses

For the ResNet (`pix2pixhd_global`) models, half precision behaves very
differently from the FFC fp16 result documented later. Two paths:

- **Blunt `model.half()`** (the `+dtype=fp16` path in `predict_mps.py`): runs
  without error, but the generator output **collapses to a near-constant gray
  *inside* the mask** — the synthesized fill is garbage, while the composited-back
  region *outside* the mask stays correct (that region is just copied original
  pixels via `inpainted = mask·pred + (1−mask)·image`). big-lama-regular:
  inside-mask MAD ~48/255 (worst image 104), overall MAD ~10. **Unusable.**
- **autocast fp16** (`torch.autocast(device_type='mps', dtype=torch.float16)`,
  weights kept fp32; `predict_mps_amp.py`): **quality-neutral** — inside-mask MAD
  0.23/255 (big-lama-regular) and 0.15/255 (lama-regular) vs fp32 — and ~1.35x
  faster than fp32 MPS. Requires **torch ≥2.4** for MPS autocast; on torch 2.2.2
  it raises `unsupported autocast device_type 'mps'`.

Why blunt `.half()` collapses (traced layer-by-layer on big-lama-regular): it is
**not** overflow — no inf/NaN, and activations stay well under fp16's 65504 max.
Relative error is <0.2% through the downsampling convs, then explodes to ~90%
across the residual blocks. The cause is **catastrophic cancellation in
BatchNorm**: these BN layers carry huge running stats (running_var median ~5200,
max ~4.8M; max|running_mean| ~4082). At magnitude ~4000, fp16's step size (ULP)
is ~2–4, so BN's `x − running_mean` subtracts two coarsely-quantized large
numbers and loses the normalized signal; ReLU gating then amplifies the noise
through the residual stack until the deep features decorrelate from the fp32
trajectory, and the final conv+sigmoid emit a flat ~0.42 gray. autocast keeps
BN/reductions in fp32, so the cancellation never happens.

MPS fp32 vs autocast fp16 (47 images; fp16 on venv-latest/torch 2.13.0, fp32
carried from each model's section — env-independent):

| Model | blocks | MPS fp32 | MPS autocast fp16 | fp16 inside-mask MAD |
|-------|:------:|----------|-------------------|:--------------------:|
| big-lama-regular | 18 | ~15.7 s | ~11.5 s | 0.23/255 |
| lama-regular | 9 | ~12.6 s | ~8.8 s | 0.15/255 |

### lama-fourier (smaller FFC, 9 blocks)

Model: `lama-fourier` (`ffc_resnet` generator, `n_blocks: 9` — half of
big-lama's 18), extracted from `models/LaMa_models.zip` →
`LaMa_models/lama-places/lama-fourier/`. Same Places (general-scene) domain and
same FFC/Fourier architecture as big-lama — it is the base "lama-fourier" from
the paper, which big-lama deepens. Checkpoint ~314 MB vs big-lama's 410 MB.
Dataset: same `LaMa_test_images` — 47 image/mask pairs.

Run command (CPU):

```bash
./venv222/bin/python bin/predict.py \
  model.path=$(pwd)/lama-fourier \
  model.checkpoint=best.ckpt \
  indir=$(pwd)/LaMa_test_images \
  outdir=$(pwd)/output-lama-fourier
```

| Metric | CPU |
|--------|-----|
| Total wall time | 199 s (3:19) for 47 images |
| Average | ~4.25 s/image |
| Per-image range | ~2.3 – 5.6 s/image |

Notes:
- CPU only, like big-lama: it is FFC-based (`torch.fft.rfftn`), so MPS is blocked
  by the missing `aten::_fft_r2c` kernel. MPS was not attempted.
- Fastest of the three on CPU (~4.25 vs big-lama ~5.75 and big-lama-regular
  ~5.07 s/image), consistent with having half the residual blocks — while keeping
  the FFT architecture and the general-scene domain that big-lama-regular trades
  away.

### CPU comparison summary

| Model | Arch | blocks | ckpt | Total (47 imgs) | Avg s/image |
|-------|------|:------:|------|-----------------|-------------|
| big-lama | FFC | 18 | 410 MB | ~270 s | ~5.75 |
| big-lama-regular | ResNet (no FFT) | 18 | 501 MB | 238 s | ~5.07 |
| lama-fourier | FFC | 9 | 314 MB | 199 s | ~4.25 |
| lama-regular | ResNet (no FFT) | 9 | 388 MB | ~204 s* | ~4.35 |

\* lama-regular's CPU run briefly overlapped two MPS runs at launch; soft upper bound.

### torch 2.13.0 (py3.11) — MPS FFT works for FFC models

Follow-up test of whether a newer torch closes the MPS FFT gap that blocked the
FFC models on torch 2.2.2. **It does.**

Environment: separate venv created with **uv**, **Python 3.11.15**,
**torch 2.13.0 + torchvision 0.28.0**, numpy 2.4.6.

```bash
uv venv --python 3.11 venv-latest
uv pip install --python venv-latest/bin/python torch torchvision   # -> 2.13.0
uv pip install --python venv-latest/bin/python pyyaml tqdm easydict scikit-image \
  scikit-learn opencv-python joblib matplotlib pandas albumentations \
  "hydra-core>=1.3,<1.4" pytorch-lightning tabulate kornia webdataset packaging \
  wldhx.yadisk-direct
```

`torch.fft.rfftn` on an `mps` tensor now returns a result on `mps:0` (it raised
`aten::_fft_r2c not implemented` on torch 2.2.2).

Both FFC models tested on the same 47 `LaMa_test_images`, MPS-only (no fallback):

```bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
# swap model.path=lama-fourier <-> big-lama; device=cpu for the CPU baseline
./venv-latest/bin/python predict_mps.py \
  model.path=$(pwd)/big-lama model.checkpoint=best.ckpt device=mps \
  indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output-big-lama-mps
```

| Model (FFC) | blocks | CPU total | MPS total (warm) | MPS avg | Speedup |
|-------------|:------:|-----------|------------------|---------|:-------:|
| lama-fourier | 9 | 194 s (3:14) | ~15 s | ~0.32 s/img (~3.1 it/s) | **~13x** |
| big-lama | 18 | 261 s (4:21) | ~22 s | ~0.47 s/img (~2.13 it/s) | **~12x** |

big-lama's MPS time was identical across two consecutive runs (~22 s), so it is a
stable warm figure, not cold-shader-compilation noise. As expected, the deeper
18-block model is slower than 9-block lama-fourier on both devices.

Notes:
- **Native MPS, no fallback.** The MPS run completes even with
  `PYTORCH_ENABLE_MPS_FALLBACK` unset — every op, including the FFT, runs on the
  GPU (no silent CPU fallback). The first run is slower (~22 s) due to one-time
  Metal shader compilation; warm runs are ~15 s.
- **Output parity with CPU:** max abs pixel diff = 1/255, mean ≈ 0 — numerically
  identical bar rounding.
- **This is the GPU option the conservative torch 2.2.2 pick gave up.** The only
  reason 2.2.2 was chosen was Python-3.9 compatibility + minimizing breakage of
  the 2021-era LaMa stack; the actual FFT/MPS conclusions were 2.2.2-specific.
- **Source patches required for the torch-2.13 / py3.11 / numpy-2 stack** (all
  minimal, git-tracked, backward-compatible with the older venvs):
  - `saicinpainting/training/data/aug.py`: guard the `DualIAATransform` / `imgaug`
    imports (removed in albumentations≥1.0 / not numpy-2 compatible; training-only,
    unused at inference).
  - `saicinpainting/training/trainers/__init__.py`: `torch.load(..., weights_only=False)`
    (torch≥2.6 flipped the default to True; the checkpoint pickles PL objects).
  - `hydra-core` bumped 1.1.0 → 1.3.4 (1.1.0 crashes on Python 3.11's dataclasses).

#### fp16 (half precision) on MPS — tried, not worthwhile

Attempted half precision to speed up MPS further. Enabled via `+dtype=fp16` on
`predict_mps.py` (`model.half()` + half inputs). Because `torch.fft` rejects fp16
(`Unsupported dtype Half`, even under autocast — the fft ops aren't in the fp32
promotion list), `FourierUnit.forward` in `saicinpainting/training/modules/ffc.py`
was patched to run the FFT/complex ops in fp32 and cast back (no-op for fp32 runs);
the spectral 1x1 conv still runs in fp16.

```bash
./venv-latest/bin/python predict_mps.py \
  model.path=$(pwd)/big-lama model.checkpoint=best.ckpt device=mps +dtype=fp16 \
  indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output-big-lama-mps-fp16
```

Result (big-lama, MPS):

| | fp32 MPS | fp16 MPS |
|---|---|---|
| Total wall time | ~22 s | ~21 s (**no speedup**) |
| Fidelity vs fp32 | — | mean pixel diff ~18–103/255, worse on large images |

Conclusion: **not worth it.** fp16 gives ~no speedup on this workload (the fp32
FFT is unchanged, and MPS conv isn't meaningfully faster in fp16 here), while the
spectral conv in fp16 loses precision badly — up to ~100/255 mean error on the
~2k images (fp16 can't represent the large spectral-domain magnitudes). Stick
with fp32 on MPS.

#### Could an fp16 DFT-as-matmul replace the FFT? (micro-benchmark)

A DFT is a matmul against a cos/sin basis (this is how CoreMLaMa runs on Core ML,
which has no FFT op). matmul *does* have an fp16 MPS kernel, so this routes around
the missing fp16 FFT. Micro-benchmark on MPS (batch of C=384 channels, 256×256,
2D separable DFT `X = W @ x @ W`):

**Speed** (per transform call):

| | time | vs FFT |
|---|------|:------:|
| `torch.fft.rfftn` fp32 | 0.95 ms | 1.0x |
| DFT-matmul fp32 | 7.95 ms | 8.4x slower |
| DFT-matmul fp16 | 3.64 ms | **3.8x slower** |

**Accuracy** (fp16 DFT-matmul vs fp32 FFT, realistic map, |coeff| up to ~26k):
max abs err 1.95, mean 0.07 (~1e-4 relative), no inf/NaN. Surprisingly clean
because **MPS fp16 matmul accumulates in fp32** — verified: `fp16_matmul(ones[4096])
= 4096.0`, whereas fp16 *accumulation* would stall at 2048 (fp16 integer step
becomes 2 past 2048). So the DFT sum-of-products is done in fp32; only the input
rounding (~1e-4) remains. This is far better than a native fp16 butterfly FFT
would be.

Verdict: **feasible and accurate, but pointless here.**
- It's O(N²) vs the FFT's O(N log N) — ~8x more FLOPs. fp16 halves the matmul
  (7.95 → 3.64 ms) but it is still **~3.8x slower than the fp32 FFT that already
  exists** on torch 2.13 MPS. Replacing a fast transform with a slow one to change
  dtype is backwards.
- It does not fix the real fp16 problem, which is BatchNorm catastrophic
  cancellation, not the transform (see the ResNet fp16 subsection).
- Caveat for LaMa: real spectra reach ~186k > fp16 max 65504, so the DFT *output*
  must stay fp32 (matmul takes fp16 in, fp32 out) or be scaled, else those
  coefficients overflow to inf.
- Where it *would* pay off: platforms with no FFT kernel (Core ML / some NPUs), or
  NVIDIA tensor cores where fp16 matmul can rival cuFFT for small N. Not Apple MPS.

### Caveats

- Single **interactive run**, not an isolated benchmark: no warm-up, no control
  for background load.
- Per-image time varies widely because the test images differ in resolution;
  the average is unweighted over mixed sizes. Treat ~5.75 s/image as a rough
  figure, not a controlled measurement.
- Run date: 2026-08-03.
