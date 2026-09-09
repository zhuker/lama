#!/usr/bin/env python3
"""
Frame-to-frame latency benchmark for LaMa (big-lama, FFT/FFC) on CUDA.

Loads the model, preloads a set of real 720p warped/mask frames, then times the
per-frame inference latency back-to-back (as in a streaming inpainting pipeline),
with proper CUDA synchronization. Reports fp32 and fp16 numbers.

Two latencies are reported per precision:
  * compute      : forward pass only (inputs already resident on GPU) + sync
  * end_to_end   : H2D copy (pinned) + forward + D2H readback + sync  (per frame)

fp16 modes:
  * fp16-amp  : torch.autocast(float16) around the fp32 model -- convs run in
                fp16 tensor cores, FFT stays fp32 (stable). This is the realistic
                deployable "fp16" path.
  * fp16-half : model + inputs cast to .half() (true half everywhere). May fail
                inside the FFC's cuFFT for non-supported sizes; skipped if it errors.

Usage:
  python bench_latency_cuda.py --indir frames --model big-lama \
      --frames 64 --warmup 20 --iters 300
"""

import argparse
import glob
import os
import statistics
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from saicinpainting.evaluation.data import ceil_modulo
from saicinpainting.training.trainers import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--indir', required=True,
                   help='folder with <id>_warped.png / <id>_mask.png pairs')
    p.add_argument('--model', required=True, help='model dir (config.yaml + models/best.ckpt)')
    p.add_argument('--checkpoint', default='best.ckpt')
    p.add_argument('--frames', type=int, default=64, help='distinct frames to cycle through')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=300, help='timed forward passes per precision')
    p.add_argument('--pad-modulo', type=int, default=8)
    p.add_argument('--precisions', default='fp32,fp16-amp,fp16-half',
                   help='comma list of: fp32, fp16-amp, fp16-half')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the model (fuses kernels; one-time compile cost)')
    return p.parse_args()


def find_pairs(indir, limit):
    mask_paths = sorted(glob.glob(os.path.join(indir, '*_mask.png')))
    pairs = []
    for mp in mask_paths:
        stem = os.path.basename(mp)[:-len('_mask.png')]
        wp = os.path.join(indir, stem + '_warped.png')
        if os.path.exists(wp):
            pairs.append((wp, mp))
        if len(pairs) >= limit:
            break
    return pairs


def read_pair(warped_path, mask_path, pad_h, pad_w):
    img = cv2.cvtColor(cv2.imread(warped_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
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
    model.freeze()
    model.to(device)
    model.eval()
    return model


def summarize(name, lat_ms):
    lat = sorted(lat_ms)
    n = len(lat)
    mean = statistics.mean(lat)
    def pct(p):
        return lat[min(n - 1, int(round(p / 100.0 * n)))]
    print(f'  {name:11s}  mean {mean:7.2f}  median {statistics.median(lat):7.2f}  '
          f'p90 {pct(90):7.2f}  p99 {pct(99):7.2f}  min {lat[0]:7.2f}  max {lat[-1]:7.2f}  '
          f'ms   |  {1000.0/mean:6.1f} fps')
    return mean


def bench_precision(mode, model_dir, checkpoint, device, gpu_frames, host_frames, warmup, iters,
                    do_compile=False):
    """gpu_frames: list of (image, mask) resident on GPU (fp32).
       host_frames: list of (pinned image, pinned mask) on CPU for end-to-end path."""
    print(f'[{mode}]')
    use_half = (mode == 'fp16-half')
    use_amp = (mode == 'fp16-amp')

    if use_half:
        model = load_model(model_dir, checkpoint, device).half()
        frames = [(im.half(), mk.half()) for im, mk in gpu_frames]
        host = [(im.half(), mk.half()) for im, mk in host_frames]
    else:
        model = load_model(model_dir, checkpoint, device)
        frames = gpu_frames
        host = host_frames

    if do_compile:
        model = torch.compile(model, dynamic=False)

    nf = len(frames)

    import contextlib
    def ctx():
        return torch.autocast('cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()

    # ---- compute-only latency (inputs already on GPU) ----
    try:
        with torch.no_grad():
            for i in range(warmup):
                im, mk = frames[i % nf]
                with ctx():
                    _ = model({'image': im, 'mask': mk})['inpainted']
            torch.cuda.synchronize()

            comp = []
            for i in range(iters):
                im, mk = frames[i % nf]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with ctx():
                    out = model({'image': im, 'mask': mk})['inpainted']
                torch.cuda.synchronize()
                comp.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        print(f'  compute path FAILED: {type(e).__name__}: {e}')
        del model
        torch.cuda.empty_cache()
        return

    # ---- end-to-end latency (H2D + forward + D2H) ----
    with torch.no_grad():
        for i in range(warmup):
            im, mk = host[i % nf]
            with ctx():
                o = model({'image': im.to(device, non_blocking=True),
                           'mask': mk.to(device, non_blocking=True)})['inpainted']
            _ = o.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()

        e2e = []
        for i in range(iters):
            im, mk = host[i % nf]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ctx():
                o = model({'image': im.to(device, non_blocking=True),
                           'mask': mk.to(device, non_blocking=True)})['inpainted']
            res = o.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t0) * 1000.0)

    summarize('compute', comp)
    summarize('end_to_end', e2e)
    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'CUDA not available'
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f'torch {torch.__version__}  cuda {torch.version.cuda}  '
          f'device {torch.cuda.get_device_name(0)}')

    pairs = find_pairs(args.indir, args.frames)
    if not pairs:
        print(f'no warped/mask pairs in {args.indir}', file=sys.stderr)
        sys.exit(1)

    probe = cv2.imread(pairs[0][0], cv2.IMREAD_COLOR)
    h, w = probe.shape[:2]
    out_h = ceil_modulo(h, args.pad_modulo) if args.pad_modulo > 1 else h
    out_w = ceil_modulo(w, args.pad_modulo) if args.pad_modulo > 1 else w
    pad_h, pad_w = out_h - h, out_w - w
    print(f'{len(pairs)} frames  {w}x{h} (padded {out_w}x{out_h})  '
          f'warmup {args.warmup}  iters {args.iters}')

    # Preload frames: GPU-resident (compute path) + pinned host (end-to-end path).
    gpu_frames, host_frames = [], []
    for wp, mp in pairs:
        img, mask = read_pair(wp, mp, pad_h, pad_w)
        im_t = torch.from_numpy(img)[None]   # (1,3,H,W)
        mk_t = torch.from_numpy(mask)[None]  # (1,1,H,W)
        gpu_frames.append((im_t.to(device), mk_t.to(device)))
        host_frames.append((im_t.pin_memory(), mk_t.pin_memory()))

    print(f'loading model from {args.model} ...\n')
    for mode in [m.strip() for m in args.precisions.split(',') if m.strip()]:
        bench_precision(mode, args.model, args.checkpoint, device,
                        gpu_frames, host_frames, args.warmup, args.iters,
                        do_compile=args.compile)
        print()


if __name__ == '__main__':
    main()
