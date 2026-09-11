#!/usr/bin/env python3
"""Derive lower-resolution copies of a <id>_warped.png / <id>_mask.png dataset.

Aspect-preserving resize + center crop (never a plain squash), so the content
stays geometrically sane while the tensor shape is exactly the target.
  1280x720 -> 1024x512 : resize to 1024x576, center-crop 32px off top/bottom
  1280x720 ->  832x480 : resize to  854x480, center-crop 11px off left/right
1024x512 matters for CUDA fp16-half: the FFC global branch runs at 1/8 res
(128x64), and cuFFT's half-precision FFT only supports power-of-two sizes.
"""
import argparse, glob, os, cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--size', required=True, help='WxH, e.g. 1024x512')
    a = p.parse_args()
    tw, th = (int(v) for v in a.size.lower().split('x'))
    os.makedirs(a.outdir, exist_ok=True)

    n = 0
    for mp in sorted(glob.glob(os.path.join(a.indir, '*_mask.png'))):
        stem = os.path.basename(mp)[:-len('_mask.png')]
        wp = os.path.join(a.indir, stem + '_warped.png')
        if not os.path.exists(wp):
            continue
        img = cv2.imread(wp, cv2.IMREAD_COLOR)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        s = max(tw / w, th / h)                      # cover the target, then crop
        rw, rh = max(tw, int(round(w * s))), max(th, int(round(h * s)))
        x0, y0 = (rw - tw) // 2, (rh - th) // 2
        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)[y0:y0+th, x0:x0+tw]
        mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)[y0:y0+th, x0:x0+tw]
        cv2.imwrite(os.path.join(a.outdir, stem + '_warped.png'), img)
        cv2.imwrite(os.path.join(a.outdir, stem + '_mask.png'), (mask > 0).astype('uint8') * 255)
        n += 1
    print(f'{n} pairs -> {a.outdir} @ {tw}x{th}')

if __name__ == '__main__':
    main()
