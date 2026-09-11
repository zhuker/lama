import os, sys
fp16_fft = os.environ.get("LAMA_FP16_FFT") == "1"
import torch
sys.path.insert(0, os.path.expanduser("~/git/lama"))
from bench_latency_cuda import find_pairs, read_pair, load_model
dev = torch.device("cuda")
pairs = find_pairs(os.path.expanduser("~/dataset16_lat80_1024x512"), 16)
for name in ["lama-fourier", "big-lama"]:
    m = load_model(os.path.expanduser(f"~/git/lama/{name}"), "best.ckpt", dev).half()
    bad, fracs = 0, []
    with torch.no_grad():
        for wp, mp in pairs:
            im, mk = read_pair(wp, mp, 0, 0)
            im = torch.from_numpy(im)[None].to(dev).half(); mk = torch.from_numpy(mk)[None].to(dev).half()
            out = m({"image": im, "mask": mk})["inpainted"]
            f = torch.isnan(out).float().mean().item(); fracs.append(f); bad += f > 0
    print(f"{name:13s} native_fp16_fft={fp16_fft!s:5s}  NaN frames {bad}/{len(pairs)}  "
          f"NaN pixel fraction mean {sum(fracs)/len(fracs):.4f}  max {max(fracs):.4f}")
    del m; torch.cuda.empty_cache()
