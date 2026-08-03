#!/usr/bin/env python3
"""Simple PNG decode benchmark using LaMa's own decode method.

Reuses saicinpainting.evaluation.data.load_image, i.e.
    np.array(Image.open(fname).convert(mode)) -> CHW float32 / 255
"""
import argparse
import glob
import os
import time
from collections import defaultdict

from saicinpainting.evaluation.data import load_image

# depth/mask are single-channel; ref/src/warped are colour.
MODES = {"depth": "L", "mask": "L", "ref": "RGB", "src": "RGB", "warped": "RGB"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder of *_<type>.png files")
    ap.add_argument("--limit", type=int, default=200,
                    help="max files per type to decode (default 200)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup files per type")
    args = ap.parse_args()

    by_type = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.dataset, "*.png"))):
        suffix = os.path.basename(f).rsplit("_", 1)[-1][:-4]  # strip _X.png
        if suffix in MODES:
            by_type[suffix].append(f)

    print(f"dataset: {args.dataset}")
    print(f"{'type':8} {'mode':4} {'n':>5} {'MB':>8} {'img/s':>9} {'MB/s':>8} {'ms/img':>8}")
    print("-" * 55)

    grand_n = grand_bytes = grand_time = 0
    for t in ["depth", "mask", "ref", "src", "warped"]:
        files = by_type.get(t, [])[: args.limit]
        if not files:
            continue
        mode = MODES[t]
        # warmup (populate OS page cache, JIT any lazy imports)
        for f in files[: args.warmup]:
            load_image(f, mode=mode)

        nbytes = sum(os.path.getsize(f) for f in files)
        start = time.perf_counter()
        for f in files:
            load_image(f, mode=mode)
        elapsed = time.perf_counter() - start

        n = len(files)
        mb = nbytes / 1e6
        print(f"{t:8} {mode:4} {n:5d} {mb:8.1f} {n/elapsed:9.1f} "
              f"{mb/elapsed:8.1f} {1000*elapsed/n:8.2f}")
        grand_n += n
        grand_bytes += nbytes
        grand_time += elapsed

    print("-" * 55)
    if grand_time:
        print(f"{'TOTAL':8} {'':4} {grand_n:5d} {grand_bytes/1e6:8.1f} "
              f"{grand_n/grand_time:9.1f} {grand_bytes/1e6/grand_time:8.1f} "
              f"{1000*grand_time/grand_n:8.2f}")


if __name__ == "__main__":
    main()
