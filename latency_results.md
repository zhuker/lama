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

Ran the same fp32 sweep on a local Mac — **Apple M5 Max**, MPS, torch 2.13.0 — comparing eager vs
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


---

# Round 2 (2026-09-09 / 09-11): RTX 5090, Apple M2, Quadro RTX 5000

Second sweep adding three more machines and a third resolution. Same model,
same bench scripts, same 80-frame dataset — but a newer stack and a
**1024×512** resolution added specifically to unlock the native fp16 FFT.

## Setup deltas vs round 1

- Framework: **torch 2.14.0** everywhere (`+cu130` / CUDA 13.0 on the NVIDIA hosts)
- Bench: `--frames 16 --warmup 20 --iters 200` (RTX 5090), `--iters 100` (Quadro, to shorten
  sustained load after its first run crashed), `--frames 16 --warmup 10 --iters 100` (MPS)
- `torch.compile` used throughout, per policy: **`mode="reduce-overhead"` on CUDA**,
  default mode on MPS (reduce-overhead also measured on MPS for reference).
- Dataset: `dataset16_lat80` — 80 warped/mask pairs, ids 00138–00217 (all past the
  degenerate-mask cutoff). Lower resolutions derived with `make_res_datasets.py`
  (aspect-preserving resize + center crop, never a squash):
  1280×720 → 1024×512 and → 832×480. All three are divisible by 8, so modulo-8
  padding is a no-op at every size.

| Host | GPU | Arch | Notes |
|---|---|---|---|
| `pop-os` | NVIDIA GeForce RTX 5090 (32 GB) | Blackwell, sm_120 | full sweep |
| `zhuker-mini` | Apple M2 (Mac mini, 10-core GPU, 16 GB) | — | full sweep, MPS |
| `zhuker-blade` | Quadro RTX 5000 Max-Q (16 GB) | Turing, sm_75 | full sweep (rerun 2026-09-11 after a crash) |

### The three fp16 flavors measured

Round 1 conflated two different things under "fp16". They are separated here:

| label | what it is |
|---|---|
| `fp16-amp` | `torch.autocast(float16)` over the fp32 model; FFT force-cast to fp32 |
| `fp16-half` (fp32 FFT island) | model + inputs `.half()`, but `FourierUnit` still upcasts around the transform |
| `fp16-half` (native) | `LAMA_FP16_FFT=1` — the transform itself runs in complex32. **Power-of-2 sizes only** |

⚠️ `LAMA_FP16_FFT=1` must **not** be set globally: it also captures the `fp16-amp`
path (whose tensors are fp16 at the FFT), making autocast fail at non-pow-2 sizes.
Set it per-run, only for the native-FFT measurement.

## RTX 5090 — compute-only latency (mean ms, fps)

| precision | mode | 1280×720 | 1024×512 | 832×480 |
|---|---|---|---|---|
| fp32 (TF32) | eager | 39.71 (25.2) | 25.16 (39.7) | 20.50 (48.8) |
| fp32 (TF32) | compile+RO | 29.36 (34.1) | 18.79 (53.2) | 15.42 (64.9) |
| fp16-amp | eager | 25.66 (39.0) | 17.55 (57.0) | 14.43 (69.3) |
| fp16-amp | compile+RO | 15.79 (63.3) | 10.77 (92.8) | 8.67 (115.3) |
| fp16-half, fp32 FFT | eager | 25.73 (38.9) | 17.73 (56.4) | 14.72 (67.9) |
| fp16-half, fp32 FFT | compile+RO | **15.56 (64.3)** | 10.49 (95.3) | **8.37 (119.4)** |
| fp16-half, native FFT | eager | n/a — non-pow-2 | 16.49 (60.7) | n/a — non-pow-2 |
| fp16-half, native FFT | compile+RO | n/a — non-pow-2 | **9.70 (103.1)** | n/a — non-pow-2 |

End-to-end (pinned H2D + forward + D2H), compile+RO: 720p 29.58 / 15.98 / 16.14 ms
(fp32 / amp / half); 1024×512 18.93 / 10.92 / 10.03 (native FFT) / 10.84 (fp32 FFT);
832×480 15.53 / 8.80 / 8.64. Copy cost stayed **0.1–0.6 ms** throughout.

No NaNs were flagged in any 5090 fp16 run.

### 5090 takeaways

- **fp16 is the fast mode here, unlike on B200.** fp16-amp+RO is **−43% to −46%** vs
  fp32+RO across all three resolutions. On B200 round 1, fp32/TF32 was competitive; on
  consumer Blackwell the fp32 (TF32) path is comparatively much weaker, so the
  precision choice matters far more.
- **reduce-overhead pays, but less than on B200:** −26% for fp32, −38–40% for
  fp16-amp. B200 saw up to −44%.
- **Resolution now scales close to linearly.** With compile+RO, 832×480
  (43.3% of the pixels) costs **−45%** (fp16-amp) / **−47%** (fp32) vs 720p.
  This is the same "kill the launch overhead and latency starts tracking pixels"
  effect round 1 found on B200/RTX 6000 — the 5090 shows it at every precision.
- **Native fp16 FFT still pays on the 5090 — more than round 1 suggested.**
  At 1024×512 with compile+RO it is **9.70 vs 10.49 ms = −7.5%** (and −7.0% eager).
  Round 1 measured only −3.7% (B200) / −0.7% (RTX 6000) once compiled and concluded
  "keep fp32 FFT when compiling". That conclusion is **hardware-dependent**: on the
  5090 the fp16 FFT is worth ~7.5%, making native-FFT the fastest config *at 1024×512*
  (103 fps compute / 100 fps end-to-end). In absolute terms 832×480 fp16-half is still
  lower-latency (8.37 ms) — it has 24% fewer pixels than 1024×512.

## Apple M2 (Mac mini, 16 GB) — MPS, compute-only latency (mean ms, fps)

⚠️ This is a **different Mac** from the round-1 MPS section, which was an
**M5 Max**. The M2 is ~6.7× slower (720p fp32 eager: 1214 ms vs 182 ms). Do not
compare the two MPS tables as if they were the same machine.

fp32, eager vs compile modes:

| Res | eager | compile-default | compile-reduce-overhead |
|---|---|---|---|
| 1280×720 | 1214.35 (0.8) | 1073.33 (0.9) | 1072.39 (0.9) |
| 1024×512 | 666.20 (1.5) | 571.70 (1.7) | 572.64 (1.7) |
| 832×480 | 534.89 (1.9) | 465.86 (2.1) | 464.74 (2.2) |

Precisions, all `torch.compile` (default mode):

| Res | fp32 | fp16-amp | fp16-half (fp32 FFT) | fp16-half (native FFT) |
|---|---|---|---|---|
| 1280×720 | 1071.42 (0.9) | 900.67 (1.1) | **881.02 (1.1)** | ❌ `NotImplementedError: Unsupported dtype Half` |
| 1024×512 | 574.04 (1.7) | 490.15 (2.0) | **475.38 (2.1)** | ❌ same |
| 832×480 | 466.21 (2.1) | 399.10 (2.5) | **387.92 (2.6)** | ❌ same |

End-to-end tracked compute within 0.5–3 ms (unified memory — no real copy cost).

### M2 / MPS takeaways

- **`torch.compile` gives a consistent −12% to −14%** at every resolution — larger and
  more reliable than round 1 saw on the M5 Max (~10% at 720p, ~0% at 480p).
- **`reduce-overhead` is a no-op on MPS**, confirming round 1: it lands within
  0.2% of default compile at all three resolutions (no CUDA graphs to capture).
  It is not a *regression* here, though — round 1's −16% at 480p did not reproduce.
- **fp16 does help on MPS, contradicting round 1's "fp32 only" note.** fp16-amp is
  **−14% to −16%** and fp16-half (with the fp32 FFT island) is **−17% to −18%** vs fp32. What is
  unavailable is the *native half FFT*: stock torch 2.14 has no MPS half-precision
  FFT kernel at all, so `LAMA_FP16_FFT=1` fails with `Unsupported dtype Half`
  regardless of whether the size is a power of two. Round 1's `RUN_FFT_INFERENCE_MPS.md`
  advice ("do NOT use fp16 on MPS") was about *quality*, not availability — speed-wise
  fp16 is a real ~18% on this hardware.
- **Compute-bound, as in round 1.** 832×480 (43.3% of the pixels) costs 43% of the
  720p time — dead-on linear. There is no launch overhead left to remove, which is
  exactly why reduce-overhead does nothing.

## Quadro RTX 5000 Max-Q — compute-only latency (mean ms, fps)

Run 2026-09-11, after the laptop was brought back up. `--iters 100` (vs 200 on the
5090) to shorten sustained load, plus a 60 s cooldown between runs.

| precision | mode | 1280×720 | 1024×512 | 832×480 |
|---|---|---|---|---|
| fp32 | eager | 377.26 (2.7) | 210.43 (4.8) | 168.49 (5.9) |
| fp32 | compile+RO | 389.33 (2.6) | 214.58 (4.7) | 169.85 (5.9) |
| fp16-amp | eager | 197.78 (5.1) | 109.79 (9.1) | 85.39 (11.7) |
| fp16-amp | compile+RO | 131.49 (7.6) | 74.79 (13.4) | 54.63 (18.3) |
| fp16-half, fp32 FFT | eager | 182.69 (5.5) | 102.38 (9.8) | 81.53 (12.3) |
| fp16-half, fp32 FFT | compile+RO | **124.25 (8.0)** | 69.99 (14.3) | **52.81 (18.9)** |
| fp16-half, native FFT | eager | n/a — non-pow-2 | 89.22 (11.2) | n/a — non-pow-2 |
| fp16-half, native FFT | compile+RO | n/a — non-pow-2 | **63.02 (15.9)** | n/a — non-pow-2 |

End-to-end, compile+RO: 720p 411.25 / 136.55 / 128.55 ms (fp32 / amp / half);
1024×512 223.41 / 76.50 / 64.27 (native FFT) / 72.22 (fp32 FFT); 832×480
173.90 / 55.65 / 54.03. Copy cost is 1–5 ms for fp16 but reaches ~22 ms for fp32 at
720p — much larger than the 5090's 0.1–0.6 ms (laptop PCIe link, fp32 readback).

No NaNs were flagged in any Quadro fp16 run.

### Thermal envelope during the run

GPU telemetry was sampled every 5 s (`bench_logs/gpu_telemetry_quadro_rtx5000.csv`).
Over 156 under-load samples: **SM clock 765–1830 MHz, mean 1053 MHz** (max boost is
2100), **peak 68 °C, peak 86.4 W**. The card spent the run power-limited at roughly
half its boost clock, so these are **sustained laptop numbers, not peak-boost numbers**.
The run completed without incident.

### Quadro takeaways

- **fp16 is worth 3× here.** With compile+RO, fp16-amp is **−65% to −68%** vs fp32
  (2.96× at 720p); on the 5090 the same gap is 1.86×. Turing has no TF32, so its fp32
  path gets no tensor-core help at all, while its fp16 path does.
- **`reduce-overhead` does nothing for fp32 and a lot for fp16.** fp32 is 1–3%
  *slower* compiled (pure compute-bound — 390 ms of fp32 math leaves no launch
  overhead to remove). fp16 drops **−32% to −36%** once compiled.
- **Native fp16 FFT pays the most here of any GPU tested:** at 1024×512 it is
  **−10.0%** with compile+RO (63.02 vs 69.99 ms) and −12.9% eager.
- **Latency scales linearly with pixels at every precision** — 832×480 costs 42–45% of
  the 720p time (pixel ratio 43.3%), and 1024×512 costs 56–57% (pixel ratio 56.9%).
  The card is compute-bound across the board.
- **vs the RTX 5090:** 7.7× slower in fp16-amp eager, 9.5× in fp32 eager, 8.0× on the
  best 720p configuration (124.25 vs 15.56 ms).

### Sep 9 partial run — discarded

The first attempt (2026-09-09) hard-crashed ~20 minutes in: the journal stops
mid-stream with no shutdown sequence and no GPU (Xid) errors. The 720p eager numbers
it produced before dying were **far slower and unstable**: fp32 1106 ms (vs 377 ms
on the rerun) and fp16-amp 270 ms (vs 198 ms). The machine was evidently already
degraded before it went down. Those numbers are **withdrawn**, along with the two
claims built on them in the first version of this doc: a "4.1× fp32→fp16 gap" and a
"28× fp32 gap vs the 5090". The real figures are 1.9× (eager) / 3.0× (compiled) and 9.5×.
The raw partial log is kept as `bench_logs/lat_quadro_rtx5000_partial.log` for reference.

## Round 2 conclusions

1. **Precision choice is hardware-dependent, and it is the biggest lever on
   consumer/older NVIDIA parts.** With compile+RO, fp16-amp is −43% to −46% on the
   5090 and −65% to −68% on the Quadro, but was not the winner on B200. Round 1's "fp32+compile is B200's fast
   mode" does not generalize.
2. **`reduce-overhead` is CUDA-only, and even there it only helps when kernels are
   fast.** −26% to −40% on the 5090, −32% to −36% for Quadro fp16 — but **zero for
   Quadro fp32** (compute-bound) and zero on MPS across three resolutions.
3. **The round-1 conclusion "native fp16 FFT isn't worth it once compiled" needs a
   hardware qualifier.** True on B200 (−3.7%) and RTX 6000 (−0.7%), but the 5090
   gets **−7.5%** and the Quadro **−10.0%**. The gain ranges from −0.7% to −10.0%
   across the four GPUs with no clean trend, so measure it on the target card.
   ⚠️ **big-lama only.** On `lama-fourier` the native fp16 FFT outputs all-NaN
   (see Round 3), so validate outputs, not just latency, before adopting it.
4. **1024×512 is the only tested resolution where the native fp16 FFT is legal**
   (128×64 feature maps). 1280×720 → 160×90 and 832×480 → 104×60 both fail the
   cuFFT power-of-2 requirement.
5. **Best measured configurations** (compute-only; all fp16-half):

   | GPU | 832×480 | 1024×512 (native FFT) | 1280×720 |
   |---|---|---|---|
   | RTX 5090 (compile+RO) | **8.37 ms (119 fps)** | 9.70 ms (103 fps) | 15.56 ms (64 fps) |
   | Quadro RTX 5000 Max-Q (compile+RO) | **52.81 ms (18.9 fps)** | 63.02 ms (15.9 fps) | 124.25 ms (8.0 fps) |
   | Apple M2 (compile) | **387.92 ms (2.6 fps)** | 475.38 ms (2.1 fps)¹ | 881.02 ms (1.1 fps) |

   ¹ fp32 FFT — MPS has no native half FFT.

## Round 2 raw logs

In-repo under `bench_logs/`:
- `lat_rtx5090.log` — full 5090 sweep (also at `pop-os:/tmp/bench_cuda.log`)
- `lat_m2_mps.log` — full M2 MPS sweep
- `lat_quadro_rtx5000.log` — full Quadro sweep, 2026-09-11 (also at `zhuker-blade:~/bench_cuda_blade.log`)
- `gpu_telemetry_quadro_rtx5000.csv` — 5 s GPU temperature/clock/power samples for that run
- `lat_quadro_rtx5000_partial.log` — the discarded 2026-09-09 partial run (see above)

Drivers: the originals were `pop-os:/tmp/run_cuda_bench.sh` (5090),
`zhuker-blade:~/run_cuda_bench_blade.sh` (Quadro; adds cooldowns + telemetry), and a
Mac `/tmp/run_mps_bench.sh` that no longer exists. `bench_logs/run_cuda_sweep.sh` and
`bench_logs/run_mps_sweep.sh` run the identical sweep; set `MODEL=big-lama` to reproduce
this round (plus `ITERS=100 COOLDOWN=60 TELEMETRY=1` for the Quadro).

## Round 2 code changes

- `bench_latency_cuda.py`: added `--compile-mode` (defaults to `reduce-overhead`),
  a NaN check on the first timed output, and an eager/compiled label in the header.
- `make_res_datasets.py` (new): derives lower-resolution copies of a
  `<id>_warped.png` / `<id>_mask.png` dataset by aspect-preserving resize +
  center crop.


---

# Round 3 (2026-09-11): `lama-fourier` on RTX 5090, Quadro RTX 5000, Apple M2

Same sweep as round 2, run on the smaller FFT model. `lama-fourier` is the Places
FFC generator with **9 blocks instead of big-lama's 18**. Its generator config is
otherwise identical: same `ngf=64`, `ratio_gin/gout=0.75`, and `fft_norm=ortho`.

## Setup

- Weights: `LaMa_models.zip` → `lama-places/lama-fourier/{config.yaml, models/best.ckpt}`,
  taken from the official LaMa Google Drive folder. `best.ckpt` is 313,967,681 bytes,
  SHA-256 `b2456284b604786a88a38549432ddf2e73a5911f9868ea692b5122e89a5f1f54`, identical
  to the `camenduru/big-lama` Hugging Face mirror. The full zip passes `unzip -t` and
  sits at `models/LaMa_models.zip` (git-ignored).
- Stack, datasets, and per-host settings are unchanged from round 2: torch 2.14.0,
  5090 at `--iters 200`, Quadro at `--iters 100` with 60 s cooldowns and telemetry,
  M2 at `--iters 100`.
- Drivers: `bench_logs/run_cuda_sweep.sh` (`MODEL=… ITERS=… COOLDOWN=… TELEMETRY=…`)
  and `bench_logs/run_mps_sweep.sh` (`MODEL=…`).

## ⚠️ Native fp16 FFT is broken on `lama-fourier`: all-NaN output

On both NVIDIA GPUs, the 1024×512 native-FFT runs (`LAMA_FP16_FFT=1`) printed a NaN
warning. A per-frame check on the 5090 shows the failure is total:

| model | native fp16 FFT | NaN frames | NaN pixel fraction |
|---|---|---|---|
| lama-fourier | on | **16 / 16** | **100%** |
| lama-fourier | off (fp32 FFT) | 0 / 16 | 0% |
| big-lama | on | 0 / 16 | 0% |
| big-lama | off (fp32 FFT) | 0 / 16 | 0% |

A step-through of the first `FourierUnit` (`model.5.conv1.ffc.convg2g.fu`) shows where
it breaks. The input is finite and small (max |x| = 19.7), yet **cuFFT's half-precision
`rfftn` returns inf/NaN**. Every later op carries the NaN forward: spectral conv,
BatchNorm, ReLU, `irfftn`. The weights and BN statistics are unremarkable.

A correctly computed ortho-normalized FFT can't overflow here: its largest possible
coefficient is ≈ 19.7 × √8192 ≈ 1,800, far below the fp16 limit of 65,504. The likely
cause — **inferred, not verified** — is that cuFFT forms the *unscaled* sum over the
64×128 = 8,192 points before applying the 1/√N scale, and that intermediate overflows
fp16 when activations aren't zero-centred. big-lama's first-unit input is smaller
(max |x| = 11.3) and evidently stays in range. If that's right, pre-scaling the FFT
input by 1/√N would fix it. Untested.

**The native-FFT timings for `lama-fourier` are therefore invalid and are excluded
from every comparison below.** They are shown struck through in the tables for completeness.

## RTX 5090 — `lama-fourier`, compute-only latency (mean ms, fps)

| precision | mode | 1280×720 | 1024×512 | 832×480 |
|---|---|---|---|---|
| fp32 (TF32) | eager | 25.57 (39.1) | 15.65 (63.9) | 12.46 (80.3) |
| fp32 (TF32) | compile+RO | 17.99 (55.6) | 11.24 (89.0) | 9.05 (110.5) |
| fp16-amp | eager | 15.95 (62.7) | 10.43 (95.8) | 8.50 (117.7) |
| fp16-amp | compile+RO | 9.17 (109.0) | 6.17 (162.0) | 4.89 (204.3) |
| fp16-half, fp32 FFT | eager | 16.02 (62.4) | 10.50 (95.3) | 8.48 (118.0) |
| fp16-half, fp32 FFT | compile+RO | **9.05 (110.6)** | **6.00 (166.7)** | **4.74 (210.8)** |
| fp16-half, native FFT | eager | n/a — non-pow-2 | ~~9.82~~ NaN | n/a — non-pow-2 |
| fp16-half, native FFT | compile+RO | n/a — non-pow-2 | ~~5.52~~ NaN | n/a — non-pow-2 |

End-to-end, compile+RO, best valid config: 9.55 / 6.32 / 5.01 ms (720p / 1024×512 /
832×480). Copy cost 0.1–0.5 ms.

## Quadro RTX 5000 Max-Q — `lama-fourier`, compute-only latency (mean ms, fps)

| precision | mode | 1280×720 | 1024×512 | 832×480 |
|---|---|---|---|---|
| fp32 | eager | 228.15 (4.4) | 128.79 (7.8) | 102.37 (9.8) |
| fp32 | compile+RO | 236.90 (4.2) | 130.11 (7.7) | 101.06 (9.9) |
| fp16-amp | eager | 116.05 (8.6) | 64.49 (15.5) | 50.64 (19.7) |
| fp16-amp | compile+RO | 78.46 (12.7) | 42.99 (23.3) | 32.19 (31.1) |
| fp16-half, fp32 FFT | eager | 110.43 (9.1) | 61.61 (16.2) | 48.63 (20.6) |
| fp16-half, fp32 FFT | compile+RO | **69.84 (14.3)** | **39.51 (25.3)** | **30.30 (33.0)** |
| fp16-half, native FFT | eager | n/a — non-pow-2 | ~~53.31~~ NaN | n/a — non-pow-2 |
| fp16-half, native FFT | compile+RO | n/a — non-pow-2 | ~~32.27~~ NaN | n/a — non-pow-2 |

End-to-end, compile+RO, best valid config: 72.61 / 40.57 / 30.94 ms.

Telemetry (`bench_logs/gpu_telemetry_lama-fourier_quadro_rtx5000.csv`), 95 under-load
samples: SM clock 825–1845 MHz, mean 1114 MHz; peak 67 °C, peak 81.8 W. No incidents.

Same patterns as big-lama on this card:
- **reduce-overhead does nothing for fp32** (−1% to +4%).
- **fp16 drops 32–36%** once compiled.
- **fp16-half is −70% vs fp32** when both are compiled.

## Apple M2 — `lama-fourier`, MPS compute-only latency (mean ms, fps)

fp32, eager vs compile modes:

| Res | eager | compile-default | compile-reduce-overhead |
|---|---|---|---|
| 1280×720 | 804.10 (1.2) | 747.28 (1.3) | 747.23 (1.3) |
| 1024×512 | 443.73 (2.3) | 398.29 (2.5) | 398.20 (2.5) |
| 832×480 | 358.09 (2.8) | 319.56 (3.1) | 319.19 (3.1) |

Precisions, all `torch.compile` (default mode):

| Res | fp32 | fp16-amp | fp16-half (fp32 FFT) | fp16-half (native FFT) |
|---|---|---|---|---|
| 1280×720 | 746.05 (1.3) | 590.38 (1.7) | **581.37 (1.7)** | ❌ `Unsupported dtype Half` |
| 1024×512 | 397.84 (2.5) | 323.79 (3.1) | **317.16 (3.2)** | ❌ same |
| 832×480 | 321.78 (3.1) | 259.47 (3.9) | **253.54 (3.9)** | ❌ same |

- **`torch.compile` saves 7–11%**, less than for big-lama (12–14%). `reduce-overhead` is
  again identical to default mode.
- **fp16-half is −20% to −22% vs fp32.**
- **Latency still scales linearly with pixels.** 832×480 takes 43.6% of the 720p time
  and has 43.3% of the pixels.

## `lama-fourier` vs big-lama

Like-for-like comparison, best valid mode for each platform: fp16-half with the FFT
kept in fp32, compiled (reduce-overhead on CUDA, default on MPS). Compute-only.

| GPU | Res | lama-fourier | big-lama | change |
|---|---|---|---|---|
| RTX 5090 | 1280×720 | 9.05 ms | 15.56 ms | **−42%** |
| RTX 5090 | 1024×512 | 6.00 ms | 10.49 ms | **−43%** |
| RTX 5090 | 832×480 | 4.74 ms | 8.37 ms | **−43%** |
| Quadro RTX 5000 | 1280×720 | 69.84 ms | 124.25 ms | **−44%** |
| Quadro RTX 5000 | 1024×512 | 39.51 ms | 69.99 ms | **−44%** |
| Quadro RTX 5000 | 832×480 | 30.30 ms | 52.81 ms | **−43%** |
| Apple M2 | 1280×720 | 581.37 ms | 881.02 ms | **−34%** |
| Apple M2 | 1024×512 | 317.16 ms | 475.38 ms | **−33%** |
| Apple M2 | 832×480 | 253.54 ms | 387.92 ms | **−35%** |

At 1024×512, big-lama's own best valid config is the native fp16 FFT (9.70 ms on the 5090,
63.02 ms on the Quadro). Against that, lama-fourier's advantage is −38% and −37%.

### Round 3 conclusions

1. **Halving the blocks cuts latency 42–44% on NVIDIA and 33–35% on the M2, not 50%.**
   The rest of the time goes to the layers the two models share: the downsampling/upsampling
   convs and the non-FFC layers.
2. **lama-fourier puts the Quadro laptop over 30 fps.** At 832×480 it runs 30.30 ms compute
   (33.0 fps) and 30.94 ms end-to-end (32.3 fps). big-lama managed 18.9 fps there.
3. **On the 5090, every resolution clears 100 fps.** 832×480 reaches 211 fps compute /
   200 fps end-to-end.
4. **Never adopt the native fp16 FFT without checking outputs.** It was a real speedup
   for big-lama and produced 100% NaN on lama-fourier, with a nearly identical architecture.
   The NaN check in `bench_latency_cuda.py` only inspects the first timed frame. That was
   enough to catch this case, but not enough to certify a config as safe.
5. **Quality was not measured.** These are latency numbers only. `lama-fourier` is the lower-quality
   model, and DATASET16.md says quality must be judged by eye.

### Best valid `lama-fourier` configurations

All fp16-half with fp32 FFT; compile+RO on CUDA, compile on MPS.

| GPU | 832×480 | 1024×512 | 1280×720 |
|---|---|---|---|
| RTX 5090 | **4.74 ms (211 fps)** | 6.00 ms (167 fps) | 9.05 ms (111 fps) |
| Quadro RTX 5000 Max-Q | **30.30 ms (33.0 fps)** | 39.51 ms (25.3 fps) | 69.84 ms (14.3 fps) |
| Apple M2 | **253.54 ms (3.9 fps)** | 317.16 ms (3.2 fps) | 581.37 ms (1.7 fps) |

## Round 3 raw logs

In `bench_logs/`:
- `lat_lama-fourier_rtx5090.log` (also `pop-os:~/bench_lama-fourier.log`)
- `lat_lama-fourier_quadro_rtx5000.log` + `gpu_telemetry_lama-fourier_quadro_rtx5000.csv`
  (also `zhuker-blade:~/bench_lama-fourier.log`)
- `lat_lama-fourier_m2_mps.log`
- NaN diagnostics (run on the 5090 with `LAMA_FP16_FFT=1`): `nan_check.py` (per-frame
  NaN count), `nan_trace.py` (first bad FourierUnit), `nan_step.py` (op-by-op step-through)


---

# Appendix: end-to-end latency, all runs

End-to-end = H2D copy + forward + D2H readback + sync, per frame (mean ms). The sections
above give compute-only latency for every run but end-to-end for only some. The tables
below are generated directly from the raw logs in `bench_logs/`. "n/a" means the run
failed: cuFFT half precision needs power-of-2 sizes, and MPS has no half FFT. Struck-through
values produced NaN output and are invalid. MPS numbers are compiled only; the MPS
eager-vs-compile comparison measures compute only. Full percentiles (median/p90/p99/min/max)
for every run are in the logs.

## big-lama — RTX 5090 (Round 2) · `lat_rtx5090.log`

| precision | 1280×720 eager | 1280×720 compile+RO | 1024×512 eager | 1024×512 compile+RO | 832×480 eager | 832×480 compile+RO |
|---|---|---|---|---|---|---|
| fp32 | 40.15 | 29.58 | 25.47 | 18.93 | 20.87 | 15.53 |
| fp16-amp | 26.06 | 15.98 | 17.84 | 10.92 | 14.65 | 8.80 |
| fp16-half, fp32 FFT | 26.33 | 16.14 | 18.08 | 10.84 | 15.05 | 8.64 |
| fp16-half, native FFT | n/a | n/a | 16.81 | 10.03 | n/a | n/a |

## big-lama — Quadro RTX 5000 Max-Q (Round 2) · `lat_quadro_rtx5000.log`

| precision | 1280×720 eager | 1280×720 compile+RO | 1024×512 eager | 1024×512 compile+RO | 832×480 eager | 832×480 compile+RO |
|---|---|---|---|---|---|---|
| fp32 | 401.60 | 411.25 | 220.55 | 223.41 | 175.66 | 173.90 |
| fp16-amp | 201.62 | 136.55 | 111.71 | 76.50 | 86.26 | 55.65 |
| fp16-half, fp32 FFT | 188.03 | 128.55 | 104.40 | 72.22 | 82.94 | 54.03 |
| fp16-half, native FFT | n/a | n/a | 91.86 | 64.27 | n/a | n/a |

## big-lama — Apple M2, compiled (Round 2) · `lat_m2_mps.log`

| precision | 1280×720 | 1024×512 | 832×480 |
|---|---|---|---|
| fp32 | 1074.90 | 575.95 | 468.66 |
| fp16-amp | 902.26 | 488.76 | 400.76 |
| fp16-half, fp32 FFT | 881.51 | 475.86 | 389.35 |
| fp16-half, native FFT | n/a | n/a | n/a |

## lama-fourier — RTX 5090 (Round 3) · `lat_lama-fourier_rtx5090.log`

| precision | 1280×720 eager | 1280×720 compile+RO | 1024×512 eager | 1024×512 compile+RO | 832×480 eager | 832×480 compile+RO |
|---|---|---|---|---|---|---|
| fp32 | 25.96 | 18.30 | 15.92 | 11.36 | 12.70 | 9.02 |
| fp16-amp | 16.44 | 9.47 | 10.71 | 6.30 | 8.72 | 5.01 |
| fp16-half, fp32 FFT | 16.54 | 9.55 | 10.83 | 6.32 | 8.75 | 5.01 |
| fp16-half, native FFT | n/a | n/a | ~~10.14~~ NaN | ~~5.74~~ NaN | n/a | n/a |

## lama-fourier — Quadro RTX 5000 Max-Q (Round 3) · `lat_lama-fourier_quadro_rtx5000.log`

| precision | 1280×720 eager | 1280×720 compile+RO | 1024×512 eager | 1024×512 compile+RO | 832×480 eager | 832×480 compile+RO |
|---|---|---|---|---|---|---|
| fp32 | 242.86 | 248.35 | 134.73 | 134.44 | 107.22 | 102.41 |
| fp16-amp | 119.36 | 80.19 | 65.90 | 43.79 | 51.48 | 32.66 |
| fp16-half, fp32 FFT | 113.06 | 72.61 | 63.03 | 40.57 | 49.66 | 30.94 |
| fp16-half, native FFT | n/a | n/a | ~~54.47~~ NaN | ~~32.93~~ NaN | n/a | n/a |

## lama-fourier — Apple M2, compiled (Round 3) · `lat_lama-fourier_m2_mps.log`

| precision | 1280×720 | 1024×512 | 832×480 |
|---|---|---|---|
| fp32 | 748.20 | 399.01 | 322.53 |
| fp16-amp | 592.30 | 325.33 | 260.60 |
| fp16-half, fp32 FFT | 583.17 | 318.80 | 254.31 |
| fp16-half, native FFT | n/a | n/a | n/a |
