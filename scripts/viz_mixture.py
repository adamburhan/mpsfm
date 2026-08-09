"""Visualize the K=2 patch-GMM depth mixtures fitted at keypoints.

Builds the scene exactly like reconstruct.py (extraction is cached, so this is
fast) but stops before mapping, then renders per-keypoint debug panels:
RGB patch | prior depth patch (+ continuity mask) | log-depth histogram with
the fitted mixture that the max-mixture BA factor consumes.

    python3 scripts/viz_mixture.py --data_dir local/example -c sp-lg_m3dv2-maxmix
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from mpsfm.data_proc.simple import SimpleParser
from mpsfm.sfm.mapper import MpsfmMapper
from mpsfm.sfm.scene.image.mixture import _disk
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

MODE0_C = "#3b82f6"  # anchor mode (blue)
MODE1_C = "#f59e0b"  # second mode (orange)
EDGE_C = "#dc2626"  # continuity-mask discontinuities (red)


def build_scene(data_dir, conf_name):
    conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{conf_name}.yaml", return_name=False)
    conf.verbose = 0
    conf.extract = []
    scene_parser = SimpleParser(data_dir=Path(data_dir))
    MpsfmMapper.freeze_conf = False
    mapper = MpsfmMapper(
        conf=conf,
        references=scene_parser.imnames,
        cache_dir=Path(data_dir) / "cache_dir",
        sfm_outputs_dir=Path(data_dir) / "sfm_outputs",
        scene_parser=scene_parser,
        ref_imids=list(scene_parser.rec.images.keys()),
    )
    return mapper, scene_parser


def plot_sample(axes, rgb, depth_obj, kp_idx, radius):
    depth = depth_obj.data_prior
    h, w = depth.shape
    sx, sy = depth_obj.camera.sx, depth_obj.camera.sy
    kp = depth_obj.kps[kp_idx]
    u = int(np.clip(np.floor(kp[0] * sx), 0, w - 1))
    v = int(np.clip(np.floor(kp[1] * sy), 0, h - 1))
    modes, weights, sigmas = (
        depth_obj.mixture["modes"][kp_idx],
        depth_obj.mixture["weights"][kp_idx],
        depth_obj.mixture["sigmas"][kp_idx],
    )

    # --- RGB patch (original resolution) ---
    r_rgb = int(round(3 * radius / sx))
    x0, x1 = int(kp[0]) - r_rgb, int(kp[0]) + r_rgb
    y0, y1 = int(kp[1]) - r_rgb, int(kp[1]) + r_rgb
    crop = rgb[max(y0, 0) : y1, max(x0, 0) : x1]
    axes[0].imshow(crop)
    axes[0].add_patch(
        plt.Circle((kp[0] - max(x0, 0), kp[1] - max(y0, 0)), radius / sx, fill=False, color="w", lw=1.5)
    )
    axes[0].plot(kp[0] - max(x0, 0), kp[1] - max(y0, 0), "+", color="w", ms=10)
    axes[0].set_title("image", fontsize=8)

    # --- depth patch (map resolution) ---
    r_d = 3 * radius
    dy0, dy1 = max(v - r_d, 0), min(v + r_d, h)
    dx0, dx1 = max(u - r_d, 0), min(u + r_d, w)
    dcrop = depth[dy0:dy1, dx0:dx1]
    im = axes[1].imshow(dcrop, cmap="viridis")
    cont = getattr(depth_obj, "continuity_mask", None)
    if cont is not None:
        axes[1].contour(~cont[dy0:dy1, dx0:dx1], levels=[0.5], colors=EDGE_C, linewidths=0.8)
    axes[1].add_patch(plt.Circle((u - dx0, v - dy0), radius, fill=False, color="w", lw=1.5))
    axes[1].plot(u - dx0, v - dy0, "+", color="w", ms=10)
    axes[1].set_title("prior depth", fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # --- patch log-depth distribution + fitted mixture ---
    dv, du = _disk(radius)
    vv, uu = v + dv, u + du
    inb = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
    dpatch = depth[vv[inb], uu[inb]]
    ok = depth_obj.valid[vv[inb], uu[inb]] & (dpatch > 0)
    y = np.log(dpatch[ok])
    axes[2].hist(y, bins=30, density=True, color="#9ca3af", alpha=0.7)
    ys = np.linspace(y.min() - 0.1, y.max() + 0.1, 300)
    for k, (m, wt, s, c) in enumerate(zip(modes, weights, sigmas, [MODE0_C, MODE1_C])):
        pdf = wt * np.exp(-0.5 * ((ys - np.log(m)) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        axes[2].plot(ys, pdf, color=c, lw=2)
        axes[2].axvline(np.log(m), color=c, lw=1, ls="--")
        axes[2].text(
            np.log(m), axes[2].get_ylim()[1] * (0.95 - 0.12 * k),
            f"m{k}: z={m:.2f} w={wt:.2f} s={s:.3f}", color=c, fontsize=7, ha="center",
        )
    sep = abs(np.log(modes[1]) - np.log(modes[0]))
    axes[2].set_title(f"log-depth in patch (sep={sep:.3f})", fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks([]), ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="local/example")
    ap.add_argument("-c", "--conf", default="sp-lg_m3dv2-maxmix")
    ap.add_argument("--imname", default=None, help="image to sample from (default: first)")
    ap.add_argument("-n", "--num_samples", type=int, default=10)
    ap.add_argument("--out", default=None, help="output dir (default: <data_dir>/mixture_viz)")
    args = ap.parse_args()

    mapper, scene_parser = build_scene(args.data_dir, args.conf)
    out_dir = Path(args.out or Path(args.data_dir) / "mixture_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = mapper.mpsfm_rec.images
    imid = next(
        (i for i, im in images.items() if args.imname is None or im.name == args.imname), None
    )
    assert imid is not None, f"image {args.imname} not found"
    image = images[imid]
    depth_obj = image.depth
    assert depth_obj.mixture is not None, "no mixture fitted -- use a maxmix config with depth.mixture: true"

    radius = depth_obj.conf.mixture_radius
    modes = depth_obj.mixture["modes"]
    sep = np.abs(np.log(modes[:, 1].clip(1e-12)) - np.log(modes[:, 0].clip(1e-12)))
    ambiguous = np.flatnonzero(sep > 1e-6)
    print(f"{image.name}: {len(ambiguous)}/{len(modes)} keypoints with a genuine second mode")
    # spectrum from strongest to weakest separation
    order = ambiguous[np.argsort(-sep[ambiguous])]
    picks = order[np.linspace(0, len(order) - 1, min(args.num_samples, len(order))).astype(int)]

    rgb = np.array(PILImage.open(Path(scene_parser.rgb_dir) / image.name).convert("RGB"))
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, kp_idx in enumerate(picks):
        plot_sample(axes[row], rgb, depth_obj, int(kp_idx), radius)
    fig.suptitle(f"{image.name} — patch-GMM mixtures (strongest→weakest separation)", fontsize=10)
    fig.tight_layout()
    fn = out_dir / f"{Path(image.name).stem}_mixtures.png"
    fig.savefig(fn, dpi=150)
    print(f"saved {fn}")


if __name__ == "__main__":
    main()
