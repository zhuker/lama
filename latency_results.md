# LaMa (big-lama) Frame-Latency Results — Resolution Sweep

Date: 2026-09-08

Question: **does per-frame inference latency depend on resolution?**
Tested big-lama (FFC/FFT generator) at two resolutions on two GPUs, each in its
fastest mode. Copy and one-time `torch.compile` cost are excluded from the
compute-only numbers.

## Setup

- Model: `big-lama` (FFC ResNet generator; FFT via `torch.fft.rfftn/irfftn` in every FFC block)
- Framework: torch 2.10.0+cu130, CUDA 13.0
- TF32 enabled (`allow_tf32=True` for matmul + cuDNN), `cudnn.benchmark=True`
- Dataset: 80 warped/mask frame pairs (00138–00217)
- Bench: `bench_latency_cuda.py --frames 80 --warmup 20 --iters 300`
- Both resolutions are divisible by 8 → **zero modulo-8 padding** at either size
  (1280/8=160, 720/8=90; 832/8=104, 480/8=60)

| GPU | Host | Fast mode |
|-----|------|-----------|
| NVIDIA B200 | `b200` | fp32 + `torch.compile` (TF32 on) |
| NVIDIA RTX PRO 6000 Blackwell | `rtx43` | fp16-amp + `torch.compile` |

Resolutions: **1280×720** (921,600 px) and **832×480** (399,360 px, −56.7%).

## Compute-only latency (inputs already on GPU; forward pass + sync)

| GPU / mode | 1280×720 | 832×480 | Δ latency | pixels cut |
|---|---|---|---|---|
| **B200** — fp32+compile (TF32) | 11.64 ms (85.9 fps) | 10.89 ms (91.8 fps) | **−6.4%** | −56.7% |
| **RTX 6000** — fp16-amp+compile | 13.82 ms (72.4 fps) | 11.43 ms (87.5 fps) | **−17.3%** | −56.7% |

### Full percentiles

**B200, fp32+compile**

| Res | mean | median | p90 | p99 | min | max | fps |
|-----|------|--------|-----|-----|-----|-----|-----|
| 1280×720 | 11.64 | 11.63 | 11.68 | 11.72 | 11.58 | 11.73 | 85.9 |
| 832×480  | 10.89 | 10.88 | 11.02 | 11.26 | 10.58 | 11.58 | 91.8 |

**RTX 6000, fp16-amp+compile**

| Res | mean | median | p90 | p99 | min | max | fps |
|-----|------|--------|-----|-----|-----|-----|-----|
| 1280×720 | 13.82 | 13.80 | 13.92 | 14.04 | 13.73 | 14.17 | 72.4 |
| 832×480  | 11.43 | 11.43 | 11.53 | 11.60 | 11.22 | 11.66 | 87.5 |

## End-to-end latency (pinned H2D + forward + D2H readback)

| GPU / mode | 1280×720 | 832×480 |
|---|---|---|
| B200 — fp32+compile | 12.03 ms (83.1 fps) | 11.04 ms (90.6 fps) |
| RTX 6000 — fp16-amp+compile | 14.15 ms (70.6 fps) | 11.55 ms (86.6 fps) |

Copy cost (end_to_end − compute) was **0.1–0.4 ms** in every case — negligible,
because the bench stages inputs in pinned memory and copies with `non_blocking=True`.

## CUDA graphs: `torch.compile(mode="reduce-overhead")`

Re-ran the same sweep with `mode="reduce-overhead"` (CUDA graph capture) to remove
per-kernel launch overhead. Compute-only latency, vs the default-compile baseline:

| GPU / mode | Res | default compile | reduce-overhead | speedup |
|---|---|---|---|---|
| B200 fp32 (TF32) | 1280×720 | 11.64 ms (85.9 fps) | **9.84 ms (101.6 fps)** | −15.5% |
| B200 fp32 (TF32) | 832×480 | 10.89 ms (91.8 fps) | **6.06 ms (165.0 fps)** | −44.4% |
| RTX 6000 fp16-amp | 1280×720 | 13.82 ms (72.4 fps) | **12.35 ms (81.0 fps)** | −10.6% |
| RTX 6000 fp16-amp | 832×480 | 11.43 ms (87.5 fps) | **6.99 ms (143.1 fps)** | −38.8% |

End-to-end (pinned H2D + forward + D2H): B200 10.16 / 6.27 ms (720p/480p),
RTX 6000 12.61 / 7.19 ms. Copy cost still ~0.2–0.3 ms.

### Resolution scaling before vs after removing overhead

| | 480p vs 720p (default) | 480p vs 720p (reduce-overhead) |
|---|---|---|
| B200 | −6.4% | **−38.4%** |
| RTX 6000 | −17.3% | **−43.4%** |

CUDA graphs confirm the overhead-bound diagnosis: with launch overhead gone, latency
finally scales with pixel count (≈−57%), so 480p becomes ~40% faster than 720p instead
of ~6–17%. The win is modest at 720p (kernels do real work) and large at 480p (kernels
were tiny and launch-bound).

### Caveats for production use of reduce-overhead
- CUDA graphs require **static input shape + stable input memory addresses** — fine for
  fixed-resolution streaming; a new resolution triggers recapture.
- Graph outputs live in **static memory and are overwritten by the next call** — consume
  or copy each frame's output before the next forward (a streaming readback already does this).
- Costs extra GPU memory for the graph pool and a longer one-time capture.

## MPS (Apple Silicon) — does reduce-overhead help here too?

Ran the same fp32 sweep on a local Mac (MPS, torch 2.13.0), comparing eager vs
`torch.compile(default)` vs `torch.compile(reduce-overhead)`. 16 frames, 100 iters.
fp32 only — MPS has no fp16 FFT path. Compute-only latency:

| Res | eager | compile-default | compile-reduce-overhead |
|---|---|---|---|
| 1280×720 | 182.5 ms (5.5 fps) | 163.4 ms (6.1 fps) | 166.9 ms (6.0 fps) |
| 832×480  | 80.9 ms (12.4 fps) | 79.8 ms (12.5 fps) | 93.1 ms (10.7 fps) |

**reduce-overhead does nothing useful on MPS** (CUDA graphs are CUDA-only): tied with
default compile at 720p, ~16% *slower* at 480p. Opposite of CUDA, MPS is **compute-bound** —
480p latency is 44% of 720p (pixel ratio 43%), i.e. it already scales linearly with pixels,
so there's no launch overhead left to remove. `torch.compile` itself barely helps (~10% at
720p, ~0% at 480p) because the FFT/complex ops run eager (inductor can't codegen them).

| Platform | scaling (eager) | binding | reduce-overhead |
|---|---|---|---|
| CUDA (B200 / RTX 6000) | flat (−6–17% for −57% px) | overhead-bound | up to −44% |
| MPS (Apple Silicon) | linear (−56% for −57% px) | compute-bound | none / slight regression |

## Full fp16 pipeline (fp16 FFT) — 1024×512

cuFFT supports half-precision FFT only for **power-of-2 transform sizes**. big-lama
downsamples ×8, so a **1024×512** input yields **128×64** feature maps (both pow-2) —
the FFT runs only on those downsampled maps (global/spectral channel subset), not the
full image. `FourierUnit` normally force-casts to fp32 around the transform; gating that
off (`LAMA_FP16_FFT=1`, fp16 input only) keeps the FFT in half (complex32). "before" =
fp16 model with fp32 FFT (default); "after" = fp16 model with fp16 FFT. Isolates FFT precision only.

### Accuracy (fp16 FFT vs fp32 FFT, same fp16 model)
**PSNR 64.3 dB**, mean abs diff 4.6e-5; worst single pixel ~19/255 (isolated). Visually identical.
(Spot-check, one 1024×512 frame.)

### Latency — 1024×512, fp16-half, compute-only (full 2×2)

**B200**

| FFT precision | eager | compiled + reduce-overhead |
|---|---|---|
| fp32 FFT | 23.69 ms (42.2 fps) | 5.61 ms (178.3 fps) |
| fp16 FFT | 21.02 ms (47.6 fps) | **5.40 ms (185.2 fps)** |

**RTX 6000**

| FFT precision | eager | compiled + reduce-overhead |
|---|---|---|
| fp32 FFT | 22.10 ms (45.2 fps) | 8.52 ms (117.3 fps) |
| fp16 FFT | 18.69 ms (53.5 fps) | **8.46 ms (118.3 fps)** |

### Finding: CUDA graphs absorb almost all of the fp16-FFT win
- **compile+RO dominates:** 4.4× (B200), 2.6× (RTX 6000) over eager → 185 / 118 fps.
- **fp16-FFT gain shrinks once compiled:** −11–15% in eager, but only **−3.7% (B200) / −0.7% (RTX)** with reduce-overhead.
- Why: the eager fp16-FFT win came mostly from skipping the `x.float()`→FFT→`.to(half)` **cast kernels** (extra launches + memory traffic). `torch.compile` fuses those casts and CUDA graphs remove launch overhead, so the two FFT precisions converge. The transform itself is small (128×64, channel subset).

### Recommendation
- **When compiling with reduce-overhead: keep fp32 FFT** — fp16 FFT trades 64 dB accuracy for ≤3.7% speedup.
- **fp16 FFT only pays off in eager** (no compile): a real ~11–15%.
- Caveat: `LAMA_FP16_FFT=1` is **pow-2 only** — it would make cuFFT error at non-pow-2 feature sizes (e.g. 1280×720 → 160×90). Flag defaults off; opt-in per run.

## Conclusion: latency is overhead-bound, not resolution-bound (on CUDA)

Cutting pixel count by **57% (2.3×)** dropped latency only **6% (B200)** to
**17% (RTX 6000)**. A pixel/compute-bound model would run 832×480 at ~0.43× the
time (≈5 ms B200, ≈6 ms RTX); it didn't come close.

The per-frame time is dominated by **fixed overhead** — kernel-launch overhead
across the many small ops in the ~18 FFC blocks, plus the FFTs. Torchinductor
cannot codegen the complex/FFT ops (`UserWarning: Torchinductor does not support
code generation for complex operators`), so those run eager. The B200 is *more*
overhead-bound (flatter response to resolution) because its compute is fast
enough that launch overhead dominates even more.

### Implications
- **Reducing resolution is a weak lever** — you pay mostly fixed overhead regardless of pixel count.
- **High-value lever (confirmed): kill launch overhead** → CUDA graphs via
  `torch.compile(mode="reduce-overhead")` gave up to −44% latency (see section above).
- **TensorRT won't help** — the FFTs are the wall and a big part of the fixed cost; TRT has no native FFT layer (would force many graph breaks).

## Raw logs (on hosts)
- `b200:~/lama-bench/lat_b200.log`, `b200:~/lama-bench/lat_b200_ro.log`
- `rtx43:~/lama-bench/lat_rtx43.log`, `rtx43:~/lama-bench/lat_rtx43_ro.log`
- fp16-FFT (1024×512): `{b200,rtx43}:~/lama-bench/lat_fp16fft_{b200,rtx43}.log` (eager),
  `lat_fp16fft_ro_{b200,rtx43}.log` (compiled + reduce-overhead)
- MPS (local): `mps_data/lat_mps.log`

## fp16-FFT code change
`saicinpainting/training/modules/ffc.py` — `FourierUnit` gains an opt-in half-precision
FFT path behind `LAMA_FP16_FFT=1` (fp16 input + pow-2 sizes only). Default (unset) is
byte-for-byte the original fp32-FFT behavior.
