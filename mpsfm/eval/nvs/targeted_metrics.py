"""Masked PSNR/SSIM/LPIPS for one gs run. Runs standalone under the 3DGS
venv python (torch + lpips; NO mpsfm imports) — invoked by scripts/eval_nvs.py.

All metrics are computed as full-image spatial maps, then averaged over the
edge band / rest region (masking restricts where the map is read, not what
the neighborhood-based metrics see). Writes targeted.json next to results.json.

  /opt/gs/bin/python mpsfm/eval/nvs/targeted_metrics.py \
      --run-dir <gs/<mode>/<scene>/<testset>/<conf>> --masks-dir <../masks> --iterations 30000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

INVALID, REST, EDGE = 0, 128, 255


def gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).float()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim_map(x, y, window):
    """x, y: 1x3xHxW in [0,1] -> HxW map (channel-averaged), reflect-padded."""
    c1, c2 = 0.01**2, 0.03**2
    pad = window.shape[-1] // 2
    x, y = [torch.nn.functional.pad(t, [pad] * 4, mode="reflect") for t in (x, y)]
    conv = lambda t: torch.nn.functional.conv2d(t, window, groups=3)
    mx, my = conv(x), conv(y)
    sxx = conv(x * x) - mx * mx
    syy = conv(y * y) - my * my
    sxy = conv(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.squeeze(0).mean(0)


def masked_psnr(x, y, region):
    mse = ((x - y) ** 2).squeeze(0).mean(0)[region].mean()
    return float(-10 * torch.log10(mse.clamp_min(1e-12)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    args = parser.parse_args()

    import lpips  # deferred: slow import

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="vgg", spatial=True).to(device)
    window = gaussian_window().to(device)

    it = f"ours_{args.iterations}"
    renders_dir = args.run_dir / "model" / "test" / it / "renders"
    gt_dir = args.run_dir / "model" / "test" / it / "gt"
    test_names = sorted((args.run_dir / "source" / "sparse" / "0" / "test.txt").read_text().split())

    per_view = {}
    load = lambda p: torch.from_numpy(np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    for idx, name in enumerate(test_names):
        rpath = renders_dir / f"{idx:05d}.png"
        mpath = args.masks_dir / f"{name}.png"
        if not rpath.exists() or not mpath.exists():
            continue
        render, gt = load(rpath), load(gt_dir / f"{idx:05d}.png")
        coded = torch.from_numpy(np.array(Image.open(mpath))).to(device)
        if coded.shape != render.shape[-2:]:
            print(f"  {name}: mask {tuple(coded.shape)} vs render {tuple(render.shape[-2:])}, skipping")
            continue
        with torch.no_grad():
            smap = ssim_map(render, gt, window)
            dmap = lpips_fn(render * 2 - 1, gt * 2 - 1).squeeze()
        view = {"mask_frac": float((coded == EDGE).sum() / max((coded != INVALID).sum(), 1))}
        for region_name, region in (("edge", coded == EDGE), ("rest", coded == REST)):
            if region.sum() == 0:
                continue
            view[region_name] = {
                "PSNR": masked_psnr(render, gt, region),
                "SSIM": float(smap[region].mean()),
                "LPIPS": float(dmap[region].mean()),
            }
        per_view[name] = view

    complete = [v for v in per_view.values() if "edge" in v and "rest" in v]
    if not complete:
        raise SystemExit(f"no evaluable views in {args.run_dir}")
    entry = {
        region: {m: sum(v[region][m] for v in complete) / len(complete) for m in ("PSNR", "SSIM", "LPIPS")}
        for region in ("edge", "rest")
    }
    entry["mask_frac"] = sum(v["mask_frac"] for v in complete) / len(complete)
    entry["n_views"] = len(complete)
    entry["per_view"] = per_view

    out_file = args.run_dir / "model" / "targeted.json"
    existing = json.load(open(out_file)) if out_file.exists() else {}
    existing[it] = entry
    with open(out_file, "w") as fp:
        json.dump(existing, fp, indent=2)
    print(f"{args.run_dir.name}: edge PSNR {entry['edge']['PSNR']:.2f}  rest PSNR {entry['rest']['PSNR']:.2f}  "
          f"mask_frac {entry['mask_frac']:.3f}  ({len(complete)} views)")


if __name__ == "__main__":
    main()
