#!/usr/bin/env python3
"""
Frame-to-frame latency benchmark for LaMa (big-lama, FFT/FFC) on MPS.

MPS analog of bench_latency_cuda.py. Loads the model, preloads real 720p
warped/mask frames, then times per-frame inference back-to-back with proper
MPS synchronization. Reports fp32 and two fp16 flavors.

Two latencies per precision:
  * compute      : forward pass only (inputs already resident on MPS) + sync
  * end_to_end   : H2D copy + forward + D2H readback + sync  (per frame)

Precisions:
  * fp32      : baseline.
  * fp16-amp  : torch.autocast('mps', float16) around the fp32 model -- convs run
                fp16, FFT stays fp32 (numerically stable). Realistic deployable path.
  * fp16-half : model + inputs .half() (true half everywhere). Needs a PyTorch build
                with the MPS half FFT kernel + LAMA_FP16_FFT=1 for native complex32
                FFT (no fp32 island). Fast but numerically unstable for big-lama
                (see biglama-fp16-mps-unstable). Flags NaN in output when detected.

Usage:
  LAMA_FP16_FFT=1 python bench_latency_mps_fp16.py --indir dataset16 --model big-lama \
      --frames 16 --warmup 10 --iters 60 --min-id 00138 [--compile]
"""
import argparse, contextlib, glob, os, statistics, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import cv2, numpy as np, torch, yaml
from omegaconf import OmegaConf
from saicinpainting.evaluation.data import ceil_modulo
from saicinpainting.training.trainers import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--indir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', default='best.ckpt')
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=60)
    p.add_argument('--pad-modulo', type=int, default=8)
    p.add_argument('--min-id', default='00138', help='skip ids below this (dataset16 degenerate masks)')
    p.add_argument('--precisions', default='fp32,fp16-amp,fp16-half')
    p.add_argument('--compile', action='store_true')
    return p.parse_args()


def find_pairs(indir, limit, min_id):
    pairs = []
    for mp in sorted(glob.glob(os.path.join(indir, '*_mask.png'))):
        stem = os.path.basename(mp)[:-len('_mask.png')]
        if min_id and stem < min_id:
            continue
        wp = os.path.join(indir, stem + '_warped.png')
        if os.path.exists(wp):
            pairs.append((wp, mp))
        if len(pairs) >= limit:
            break
    return pairs


def read_pair(wp, mp, pad_h, pad_w):
    img = cv2.cvtColor(cv2.imread(wp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
    mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
    if pad_h or pad_w:
        img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='symmetric')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='symmetric')
    return img, mask[None]


def load_model(model_dir, checkpoint, device):
    with open(os.path.join(model_dir, 'config.yaml')) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    cfg.training_model.predict_only = True
    cfg.visualizer.kind = 'noop'
    model = load_checkpoint(cfg, os.path.join(model_dir, 'models', checkpoint),
                            strict=False, map_location='cpu')
    model.freeze(); model.to(device); model.eval()
    return model


def summarize(name, lat):
    lat = sorted(lat); n = len(lat); mean = statistics.mean(lat)
    pct = lambda p: lat[min(n - 1, int(round(p / 100.0 * n)))]
    print(f'  {name:11s}  mean {mean:7.2f}  median {statistics.median(lat):7.2f}  '
          f'p90 {pct(90):7.2f}  p99 {pct(99):7.2f}  min {lat[0]:7.2f}  max {lat[-1]:7.2f}  '
          f'ms   |  {1000.0/mean:6.1f} fps')


def bench_precision(mode, model_dir, checkpoint, device, gpu_frames, host_frames,
                    warmup, iters, do_compile):
    print(f'[{mode}]' + ('  (compiled)' if do_compile else ''))
    use_half = (mode == 'fp16-half')
    use_amp = (mode == 'fp16-amp')

    if use_half:
        model = load_model(model_dir, checkpoint, device).half()
        frames = [(im.half(), mk.half()) for im, mk in gpu_frames]
        host = [(im.half(), mk.half()) for im, mk in host_frames]
    else:
        model = load_model(model_dir, checkpoint, device)
        frames, host = gpu_frames, host_frames

    if do_compile:
        model = torch.compile(model, dynamic=False)

    nf = len(frames)
    ctx = (lambda: torch.autocast('mps', dtype=torch.float16)) if use_amp else contextlib.nullcontext

    # ---- compute-only latency (inputs already on MPS) ----
    try:
        nan_seen = False
        with torch.no_grad():
            for i in range(warmup):
                im, mk = frames[i % nf]
                with ctx():
                    _ = model({'image': im, 'mask': mk})['inpainted']
            torch.mps.synchronize()
            comp = []
            for i in range(iters):
                im, mk = frames[i % nf]
                torch.mps.synchronize()
                t0 = time.perf_counter()
                with ctx():
                    out = model({'image': im, 'mask': mk})['inpainted']
                torch.mps.synchronize()
                comp.append((time.perf_counter() - t0) * 1000.0)
                if i == 0:
                    nan_seen = bool(torch.isnan(out).any().item())
    except Exception as e:
        print(f'  compute path FAILED: {type(e).__name__}: {e}')
        del model; torch.mps.empty_cache(); return

    # ---- end-to-end latency (H2D + forward + D2H readback) ----
    with torch.no_grad():
        for i in range(warmup):
            im, mk = host[i % nf]
            with ctx():
                o = model({'image': im.to(device), 'mask': mk.to(device)})['inpainted']
            _ = o.float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
        torch.mps.synchronize()
        e2e = []
        for i in range(iters):
            im, mk = host[i % nf]
            torch.mps.synchronize()
            t0 = time.perf_counter()
            with ctx():
                o = model({'image': im.to(device), 'mask': mk.to(device)})['inpainted']
            _ = o.float().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            torch.mps.synchronize()
            e2e.append((time.perf_counter() - t0) * 1000.0)

    summarize('compute', comp)
    summarize('end_to_end', e2e)
    if nan_seen:
        print('  ** NaN detected in output (numerically unstable) **')
    del model; torch.mps.empty_cache()


def main():
    args = parse_args()
    assert torch.backends.mps.is_available(), 'MPS not available'
    device = torch.device('mps')
    print(f'torch {torch.__version__}  device mps  LAMA_FP16_FFT={os.environ.get("LAMA_FP16_FFT")}')

    pairs = find_pairs(args.indir, args.frames, args.min_id)
    if not pairs:
        print(f'no warped/mask pairs in {args.indir}', file=sys.stderr); sys.exit(1)
    probe = cv2.imread(pairs[0][0]); h, w = probe.shape[:2]
    out_h = ceil_modulo(h, args.pad_modulo) if args.pad_modulo > 1 else h
    out_w = ceil_modulo(w, args.pad_modulo) if args.pad_modulo > 1 else w
    pad_h, pad_w = out_h - h, out_w - w
    print(f'{len(pairs)} frames from id>={args.min_id}  {w}x{h} (padded {out_w}x{out_h})  '
          f'warmup {args.warmup}  iters {args.iters}')

    gpu_frames, host_frames = [], []
    for wp, mp in pairs:
        img, mask = read_pair(wp, mp, pad_h, pad_w)
        im_t = torch.from_numpy(img)[None]; mk_t = torch.from_numpy(mask)[None]
        gpu_frames.append((im_t.to(device), mk_t.to(device)))
        host_frames.append((im_t, mk_t))   # MPS uses unified memory; no pin_memory

    print(f'loading model from {args.model} ...\n')
    for mode in [m.strip() for m in args.precisions.split(',') if m.strip()]:
        bench_precision(mode, args.model, args.checkpoint, device,
                        gpu_frames, host_frames, args.warmup, args.iters, args.compile)
        print()


if __name__ == '__main__':
    main()
