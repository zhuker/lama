#!/usr/bin/env python3
"""PNG decode benchmark using richgel999/fpng, mirroring bench_decode.py.

fpng can only decode PNGs it encoded, and only 3/4-channel images, so we:
  1. Load each dataset PNG with PIL in LaMa's mode (grayscale L expanded to RGB
     for fpng, colour kept as RGB) and re-encode it to an fpng PNG (cached).
  2. Time fpng decode of those files -> LaMa-style CHW float32/255 tensor,
     matching load_image's post-processing so it's an apples-to-apples decode.
"""
import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fpng_py"))
import fpng  # noqa: E402

MODES = {"depth": "L", "mask": "L", "ref": "RGB", "src": "RGB", "warped": "RGB"}


def reencode(dataset, cache, types, limit):
    """Re-encode originals to fpng PNGs; returns {type: [fpng file paths]}."""
    os.makedirs(cache, exist_ok=True)
    out = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(dataset, "*.png"))):
        t = os.path.basename(f).rsplit("_", 1)[-1][:-4]
        if t not in types or len(out[t]) >= limit:
            continue
        dst = os.path.join(cache, os.path.basename(f))
        if not os.path.exists(dst):
            arr = np.array(Image.open(f).convert(MODES[t]))
            if arr.ndim == 2:                       # grayscale -> 3ch for fpng
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            with open(dst, "wb") as fh:
                fh.write(fpng.encode(arr))
        out[t].append(dst)
    return out


def load_fpng(fname, mode):
    """fpng file -> CHW float32/255, matching load_image's output shape."""
    with open(fname, "rb") as fh:
        arr, _ = fpng.decode(fh.read(), desired_channels=3)  # HxWx3 uint8
    img = arr[:, :, 0] if mode == "L" else np.transpose(arr, (2, 0, 1))
    return img.astype("float32") / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--cache", default="/tmp/fpng_cache")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    fpng.init()
    print(f"fpng SSE4.1: {fpng.cpu_supports_sse41()}  (scalar fallback if False)")
    print("re-encoding originals to fpng PNGs (cached) ...")
    files_by_type = reencode(args.dataset, args.cache, MODES, args.limit)

    print(f"\ndataset: {args.dataset}  (fpng-encoded copies in {args.cache})")
    print(f"{'type':8} {'mode':4} {'n':>5} {'MB':>8} {'img/s':>9} {'MB/s':>8} {'ms/img':>8}")
    print("-" * 55)

    gn = gb = gt = 0
    for t in ["depth", "mask", "ref", "src", "warped"]:
        files = files_by_type.get(t, [])
        if not files:
            continue
        mode = MODES[t]
        for f in files[: args.warmup]:
            load_fpng(f, mode)
        nbytes = sum(os.path.getsize(f) for f in files)
        start = time.perf_counter()
        for f in files:
            load_fpng(f, mode)
        el = time.perf_counter() - start
        n, mb = len(files), nbytes / 1e6
        print(f"{t:8} {mode:4} {n:5d} {mb:8.1f} {n/el:9.1f} {mb/el:8.1f} {1000*el/n:8.2f}")
        gn += n; gb += nbytes; gt += el

    print("-" * 55)
    if gt:
        print(f"{'TOTAL':8} {'':4} {gn:5d} {gb/1e6:8.1f} {gn/gt:9.1f} "
              f"{gb/1e6/gt:8.1f} {1000*gt/gn:8.2f}")


if __name__ == "__main__":
    main()
