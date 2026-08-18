"""Visualize fitted depth mixtures on a Replica scene, scored against GT.

Same panels as viz_mixture.py (RGB patch | prior depth + continuity edges |
patch log-depth histogram + fitted modes), plus GT depth overlaid on the
histogram: green line = GT at the keypoint pixel, green band = GT min/max in
the 3x3 patch (shows both surfaces when the kp straddles an edge). Sensible
mode detection = the band endpoints sit on the two modes.

    python scripts/viz_mixture_replica.py -s office0 --imname frame000550.jpg
    python scripts/viz_mixture_replica.py -s office0 --kps 1234 5678   # specific p2D ids (e.g. from snapshot CSVs)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from viz_mixture import plot_sample  # noqa: E402
from mpsfm.data_proc.replica import ReplicaParser  # noqa: E402
from mpsfm.sfm.mapper import MpsfmMapper  # noqa: E402
from mpsfm.utils.tools import load_cfg  # noqa: E402
from mpsfm.vars import gvars  # noqa: E402

GT_C = "#16a34a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scene", default="office0")
    ap.add_argument("-m", "--mode", default="mini")
    ap.add_argument("--testset_id", type=int, default=0)
    ap.add_argument("-c", "--conf", default="paper/repr-sp-lg_m3dv2-maxmix")
    ap.add_argument("--imname", default=None, help="image to sample (default: middle of window)")
    ap.add_argument("--kps", type=int, nargs="+", help="explicit keypoint ids (default: separation spectrum)")
    ap.add_argument("-n", "--num_samples", type=int, default=10)
    ap.add_argument("--out", default="local/diag/mixture_viz")
    args = ap.parse_args()

    parser = ReplicaParser(args.scene)
    with open(parser.dataset.testsets / args.scene / f"{args.mode}.yaml") as f:
        ref_imids = yaml.safe_load(f)[args.testset_id]
    conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml", return_name=False)
    conf.verbose = 0
    conf.extract = []
    MpsfmMapper.freeze_conf = False
    mapper = MpsfmMapper(
        conf=conf,
        references=[parser.rec.images[i].name for i in ref_imids],
        cache_dir=parser.dataset.default_cache_dir / args.scene,
        sfm_outputs_dir=Path(args.out) / "sfm_tmp",
        scene_parser=parser,
        ref_imids=ref_imids,
    )

    images = mapper.mpsfm_rec.images
    if args.imname is None:
        imid = sorted(images)[len(images) // 2]
    else:
        imid = next(i for i, im in images.items() if im.name == args.imname)
    image = images[imid]
    depth_obj = image.depth
    assert depth_obj.mixture is not None, "no mixture fitted — conf needs depth.mixture: true"

    gt = parser.gt_depth(next(i for i, im in parser.rec.images.items() if im.name == image.name))
    modes = depth_obj.mixture["modes"]
    sep = np.abs(np.log(modes[:, 1].clip(1e-12)) - np.log(modes[:, 0].clip(1e-12)))
    if args.kps:
        picks = np.array(args.kps)
    else:
        order = np.flatnonzero(sep > 1e-6)[np.argsort(-sep[np.flatnonzero(sep > 1e-6)])]
        picks = order[np.linspace(0, len(order) - 1, min(args.num_samples, len(order))).astype(int)]
    print(f"{image.name}: {int((sep > 1e-6).sum())}/{len(modes)} bimodal kps; showing {len(picks)}")

    rgb = np.array(PILImage.open(parser.rgb_dir / image.name).convert("RGB"))
    radius = depth_obj.conf.mixture_radius
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    H, W = gt.shape
    for row, kp_idx in enumerate(picks):
        plot_sample(axes[row], rgb, depth_obj, kp_idx, radius)
        u = int(np.clip(round(depth_obj.kps[kp_idx][0]), 0, W - 1))
        v = int(np.clip(round(depth_obj.kps[kp_idx][1]), 0, H - 1))
        patch = gt[max(v - 1, 0) : v + 2, max(u - 1, 0) : u + 2]
        patch = patch[patch > 0]
        ax = axes[row][2]
        if gt[v, u] > 0:
            ax.axvline(np.log(gt[v, u]), color=GT_C, lw=2, label="GT@px")
        if len(patch):
            ax.axvspan(np.log(patch.min()), np.log(patch.max()), color=GT_C, alpha=0.15)
        ax.legend(fontsize=6, loc="upper right")
        axes[row][0].set_ylabel(f"kp {kp_idx}", fontsize=8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{args.scene}_{Path(image.name).stem}_mixtures.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"saved {fname}")


if __name__ == "__main__":
    main()
