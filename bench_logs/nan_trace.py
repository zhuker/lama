import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/git/lama"))
from bench_latency_cuda import find_pairs, read_pair, load_model
dev = torch.device("cuda")
wp, mp = find_pairs(os.path.expanduser("~/dataset16_lat80_1024x512"), 1)[0]
for name in ["lama-fourier", "big-lama"]:
    m = load_model(os.path.expanduser(f"~/git/lama/{name}"), "best.ckpt", dev).half()
    fus = [(n, mod) for n, mod in m.generator.named_modules() if type(mod).__name__ == "FourierUnit"]
    log = []
    for idx, (n, mod) in enumerate(fus):
        def hook(mod, inp, out, idx=idx, n=n):
            x = inp[0]
            log.append((idx, n, x.abs().max().item(), torch.isinf(out).any().item(), torch.isnan(out).any().item(),
                        out[torch.isfinite(out)].abs().max().item() if torch.isfinite(out).any() else float("nan")))
        mod.register_forward_hook(hook)
    im, mk = read_pair(wp, mp, 0, 0)
    with torch.no_grad():
        m({"image": torch.from_numpy(im)[None].to(dev).half(), "mask": torch.from_numpy(mk)[None].to(dev).half()})
    print(f"== {name}: {len(fus)} FourierUnits, fft_norm={fus[0][1].fft_norm}")
    first_bad = next((r for r in log if r[3] or r[4]), None)
    for r in log[:4]: print(f"   FU#{r[0]:2d} in_absmax {r[2]:9.1f}  out_inf {r[3]!s:5s} out_nan {r[4]!s:5s} out_finite_absmax {r[5]:9.1f}")
    if first_bad:
        r = first_bad; print(f"   FIRST BAD: FU#{r[0]} ({r[1]})  in_absmax {r[2]:.1f}  inf={r[3]} nan={r[4]}")
    else:
        print(f"   no inf/NaN in any FourierUnit; max in_absmax {max(r[2] for r in log):.1f}")
    del m; torch.cuda.empty_cache()
