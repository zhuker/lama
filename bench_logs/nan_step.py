import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/git/lama"))
from bench_latency_cuda import find_pairs, read_pair, load_model
dev = torch.device("cuda")
wp, mp = find_pairs(os.path.expanduser("~/dataset16_lat80_1024x512"), 1)[0]
def st(t):
    f = torch.view_as_real(t).float() if t.is_complex() else t.float()
    fin = f[torch.isfinite(f)]
    amax = fin.abs().max().item() if fin.numel() else float("nan")
    return "absmax %10.2f  inf %-5s nan %-5s" % (amax, torch.isinf(f).any().item(), torch.isnan(f).any().item())
for name in ["lama-fourier", "big-lama"]:
    m = load_model(os.path.expanduser("~/git/lama/" + name), "best.ckpt", dev).half()
    fu = next(mod for mod in m.generator.modules() if type(mod).__name__ == "FourierUnit")
    cap = {}
    fu.register_forward_pre_hook(lambda mod, inp: cap.setdefault("x", inp[0].detach().clone()))
    im, mk = read_pair(wp, mp, 0, 0)
    with torch.no_grad():
        m({"image": torch.from_numpy(im)[None].to(dev).half(), "mask": torch.from_numpy(mk)[None].to(dev).half()})
        x = cap["x"]; b = x.shape[0]
        print("== %s  FU#0  in %s  spectral_pos_enc=%s use_se=%s" % (name, tuple(x.shape), fu.spectral_pos_encoding, fu.use_se))
        print("   bn.running_var min %.3e  bn.weight absmax %.2f  conv.weight absmax %.3f  bn.eps %g" % (
            fu.bn.running_var.float().min().item(), fu.bn.weight.float().abs().max().item(),
            fu.conv_layer.weight.float().abs().max().item(), fu.bn.eps))
        print("   input          " + st(x))
        ff = torch.fft.rfftn(x, dim=(-2, -1), norm=fu.fft_norm);                      print("   rfftn (c32)    " + st(ff))
        ff = torch.stack((ff.real, ff.imag), dim=-1).permute(0, 1, 4, 2, 3).contiguous()
        ff = ff.view((b, -1) + ff.size()[3:]).to(x.dtype)
        y = fu.conv_layer(ff);                                                       print("   spectral conv  " + st(y))
        y = fu.bn(y);                                                                print("   batchnorm      " + st(y))
        y = fu.relu(y);                                                              print("   relu           " + st(y))
        y = y.view((b, -1, 2) + y.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        out = torch.fft.irfftn(torch.view_as_complex(y), s=x.shape[-2:], dim=(-2, -1), norm=fu.fft_norm)
        print("   irfftn (c32)   " + st(out))
        out32 = torch.fft.irfftn(torch.complex(y[..., 0].float(), y[..., 1].float()), s=x.shape[-2:], dim=(-2, -1), norm=fu.fft_norm)
        print("   irfftn fp32, same spectrum: " + st(out32) + "   (fp16 max is 65504)")
    del m; torch.cuda.empty_cache()
