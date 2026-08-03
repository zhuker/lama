#!/usr/bin/env python3
"""
Batch LaMa (big-lama, FFT/FFC) inpainting on MPS for the hybrid-reproj dataset.

This folder pairs files as ``<id>_warped.png`` (image) + ``<id>_mask.png`` (mask),
which LaMa's stock evaluation dataset does NOT handle: it would look for
``<id>.png`` next to ``<id>_mask.png``. So we do the pairing ourselves.

Optimizations (all images are assumed to share one resolution):
  - reusable preallocated host buffers for a whole batch (no per-image alloc),
  - multithreaded PNG decode straight into those buffers (cv2 releases the GIL),
  - batched forward passes on the Apple GPU (MPS), fp32.

Only ``*_warped.png`` and ``*_mask.png`` are touched; every other file
(``_src``, ``_ref``, ``_depth``, ``_meta.json``, ...) is ignored.

Usage:
  export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
  ./venv-latest/bin/python predict_batch_warped_mps.py \
    --indir  /Users/azhukov/projects/hybrid-reproj/dataset16 \
    --outdir $(pwd)/output-dataset16-biglama \
    --model  $(pwd)/big-lama \
    --device mps --batch-size 4
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Keep BLAS single-threaded; the GPU is doing the heavy lifting and oversubscribed
# CPU threads only hurt the decode pool. Must be set before numpy/torch import.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf

from saicinpainting.evaluation.data import ceil_modulo
from saicinpainting.training.trainers import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--indir', default='/Users/azhukov/projects/hybrid-reproj/dataset16',
                   help='folder containing <id>_warped.png / <id>_mask.png pairs')
    p.add_argument('--outdir', default=os.path.join(os.getcwd(), 'output-dataset16-biglama'),
                   help='where inpainted PNGs are written (<id>.png)')
    p.add_argument('--model', default=os.path.join(os.getcwd(), 'big-lama'),
                   help='model dir with config.yaml + models/<checkpoint> (big-lama = FFT/FFC)')
    p.add_argument('--checkpoint', default='best.ckpt')
    p.add_argument('--device', default='mps', choices=['mps', 'cpu', 'cuda'])
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--read-threads', type=int, default=8,
                   help='threads decoding PNGs into the preallocated batch buffer')
    p.add_argument('--pad-modulo', type=int, default=8,
                   help='pad H/W up to a multiple of this (LaMa default 8)')
    p.add_argument('--out-suffix', default='.png',
                   help='output filename = <id> + this suffix')
    p.add_argument('--limit', type=int, default=0,
                   help='process at most N pairs (0 = all); handy for a smoke test')
    return p.parse_args()


def find_pairs(indir):
    """Return sorted [(id, warped_path, mask_path)] for every id having both files."""
    mask_paths = glob.glob(os.path.join(indir, '*_mask.png'))
    pairs = []
    for mp in mask_paths:
        stem = os.path.basename(mp)[:-len('_mask.png')]
        wp = os.path.join(indir, stem + '_warped.png')
        if os.path.exists(wp):
            pairs.append((stem, wp, mp))
    pairs.sort(key=lambda t: t[0])
    return pairs


class BatchReader:
    """Decodes a batch of (warped, mask) pairs into reusable preallocated buffers.

    Buffers are sized to the padded resolution and filled in-place by a thread
    pool. Both dims of this dataset (720x1280) are already multiples of 8, so
    padding is a no-op here, but the symmetric-pad path is kept for generality.
    """

    def __init__(self, orig_h, orig_w, mod, batch_size, n_threads):
        self.orig_h, self.orig_w = orig_h, orig_w
        self.out_h = ceil_modulo(orig_h, mod) if mod > 1 else orig_h
        self.out_w = ceil_modulo(orig_w, mod) if mod > 1 else orig_w
        self.pad_h = self.out_h - orig_h
        self.pad_w = self.out_w - orig_w
        # Preallocated, reused every batch. from_numpy on a slice of these shares
        # memory, so the only per-batch copy is host->GPU.
        self.img_buf = np.empty((batch_size, 3, self.out_h, self.out_w), np.float32)
        self.mask_buf = np.empty((batch_size, 1, self.out_h, self.out_w), np.float32)
        self.pool = ThreadPoolExecutor(max_workers=n_threads)

    def _read_one(self, j, warped_path, mask_path):
        img = cv2.imread(warped_path, cv2.IMREAD_COLOR)          # BGR, HWC, uint8
        if img is None:
            raise RuntimeError(f'failed to read {warped_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))       # CHW
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)       # HW, uint8
        if mask is None:
            raise RuntimeError(f'failed to read {mask_path}')
        mask = (mask > 0).astype(np.float32)                     # binarized like predict.py

        if self.pad_h or self.pad_w:
            img = np.pad(img, ((0, 0), (0, self.pad_h), (0, self.pad_w)), mode='symmetric')
            mask = np.pad(mask, ((0, self.pad_h), (0, self.pad_w)), mode='symmetric')

        self.img_buf[j] = img
        self.mask_buf[j, 0] = mask

    def load(self, batch_pairs):
        """batch_pairs: list of (id, warped_path, mask_path). Returns filled views."""
        n = len(batch_pairs)
        futures = [self.pool.submit(self._read_one, j, wp, mp)
                   for j, (_id, wp, mp) in enumerate(batch_pairs)]
        for f in futures:
            f.result()  # propagate read errors
        return self.img_buf[:n], self.mask_buf[:n]

    def close(self):
        self.pool.shutdown(wait=True)


def load_model(model_dir, checkpoint, device):
    train_config_path = os.path.join(model_dir, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(model_dir, 'models', checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    return model


def main():
    args = parse_args()

    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    pairs = find_pairs(args.indir)
    if not pairs:
        print(f'No <id>_warped.png / <id>_mask.png pairs found in {args.indir}', file=sys.stderr)
        sys.exit(1)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f'Found {len(pairs)} warped/mask pairs in {args.indir}')

    # All images share one resolution (per the task); probe the first pair.
    probe = cv2.imread(pairs[0][1], cv2.IMREAD_COLOR)
    orig_h, orig_w = probe.shape[:2]
    print(f'Resolution: {orig_w}x{orig_h}  (padding to multiple of {args.pad_modulo})')

    print(f'Loading model from {args.model} on {device} ...')
    model = load_model(args.model, args.checkpoint, device)

    reader = BatchReader(orig_h, orig_w, args.pad_modulo, args.batch_size, args.read_threads)

    n = len(pairs)
    t0 = time.time()
    try:
        with torch.no_grad():
            for start in tqdm.trange(0, n, args.batch_size):
                batch_pairs = pairs[start:start + args.batch_size]
                img_np, mask_np = reader.load(batch_pairs)

                image = torch.from_numpy(img_np).to(device, non_blocking=True)
                mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
                batch = {'image': image, 'mask': mask}

                batch = model(batch)
                out = batch['inpainted']  # (B, 3, out_h, out_w), fp32 in [0, 1]
                # crop away symmetric padding back to original resolution
                out = out[:, :, :orig_h, :orig_w]
                out = (out.clamp(0, 1) * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

                for k, (stem, _wp, _mp) in enumerate(batch_pairs):
                    bgr = cv2.cvtColor(out[k], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(args.outdir, stem + args.out_suffix), bgr)
    finally:
        reader.close()

    dt = time.time() - t0
    print(f'Done: {n} images in {dt:.1f}s  ({dt / n * 1000:.0f} ms/image, '
          f'{n / dt:.2f} img/s) -> {args.outdir}')


if __name__ == '__main__':
    main()
