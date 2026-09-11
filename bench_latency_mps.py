#!/usr/bin/env python3
"""MPS frame-latency bench for big-lama. fp32 only (MPS FFT has no fp16 path).
Compares eager vs torch.compile(default) vs torch.compile(reduce-overhead).
Each mode runs in isolation so a compile failure on one doesn't abort the rest."""
import argparse, glob, os, statistics, sys, time
import cv2, numpy as np, torch, yaml
from omegaconf import OmegaConf
from saicinpainting.evaluation.data import ceil_modulo
from saicinpainting.training.trainers import load_checkpoint


def find_pairs(indir, limit):
    out = []
    for mp in sorted(glob.glob(os.path.join(indir, '*_mask.png'))):
        stem = os.path.basename(mp)[:-len('_mask.png')]
        wp = os.path.join(indir, stem + '_warped.png')
        if os.path.exists(wp):
            out.append((wp, mp))
        if len(out) >= limit:
            break
    return out


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
    print(f'  {name:22s} mean {mean:7.2f}  median {statistics.median(lat):7.2f}  '
          f'p90 {pct(90):7.2f}  p99 {pct(99):7.2f}  min {lat[0]:7.2f}  max {lat[-1]:7.2f} ms'
          f'  | {1000.0/mean:6.1f} fps')


def run_mode(name, model_dir, ckpt, device, frames, warmup, iters, compile_mode):
    try:
        model = load_model(model_dir, ckpt, device)
        if compile_mode is not None:
            model = torch.compile(model, dynamic=False, mode=compile_mode)
        nf = len(frames)
        with torch.no_grad():
            for i in range(warmup):
                im, mk = frames[i % nf]
                _ = model({'image': im, 'mask': mk})['inpainted']
            torch.mps.synchronize()
            lat = []
            for i in range(iters):
                im, mk = frames[i % nf]
                torch.mps.synchronize()
                t0 = time.perf_counter()
                _ = model({'image': im, 'mask': mk})['inpainted']
                torch.mps.synchronize()
                lat.append((time.perf_counter() - t0) * 1000.0)
        summarize(name, lat)
    except Exception as e:
        print(f'  {name:22s} FAILED: {type(e).__name__}: {e}')
    finally:
        torch.mps.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', default='best.ckpt')
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--pad-modulo', type=int, default=8)
    args = p.parse_args()

    assert torch.backends.mps.is_available(), 'MPS not available'
    device = torch.device('mps')
    print(f'torch {torch.__version__}  device mps')

    pairs = find_pairs(args.indir, args.frames)
    if not pairs:
        print(f'no pairs in {args.indir}', file=sys.stderr); sys.exit(1)
    probe = cv2.imread(pairs[0][0]); h, w = probe.shape[:2]
    out_h = ceil_modulo(h, args.pad_modulo); out_w = ceil_modulo(w, args.pad_modulo)
    pad_h, pad_w = out_h - h, out_w - w
    print(f'{len(pairs)} frames  {w}x{h} (padded {out_w}x{out_h})  warmup {args.warmup}  iters {args.iters}')

    frames = []
    for wp, mp in pairs:
        img, mask = read_pair(wp, mp, pad_h, pad_w)
        frames.append((torch.from_numpy(img)[None].to(device),
                       torch.from_numpy(mask)[None].to(device)))

    print('loading model + timing (fp32)...\n', flush=True)
    run_mode('eager',                    args.model, args.checkpoint, device, frames, args.warmup, args.iters, None)
    run_mode('compile-default',          args.model, args.checkpoint, device, frames, args.warmup, args.iters, 'default')
    run_mode('compile-reduce-overhead',  args.model, args.checkpoint, device, frames, args.warmup, args.iters, 'reduce-overhead')


if __name__ == '__main__':
    main()
