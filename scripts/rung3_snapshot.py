"""Rung-3 final-state snapshot: per-keypoint mode selection vs Replica GT.

Joins the last BA diag dump per image (MPSFM_DIAG_DIR, see mpsfm/utils/diag.py)
from two arms — maxmix and the passive-mixture unimodal baseline
(repr-sp-lg_m3dv2-diagmix) — with Replica GT depth, and reports per keypoint:
the two extractor modes, the GT-correct mode, what maxmix selected, what
unimodal converged to, and the uncertainties used. Rows are matched across
arms by (image name, point2D idx); both arms share cached SuperPoint features
so keypoint indexing is identical.

Reconstruction depths/modes live in the recon scale frame; each arm is aligned
to metric GT with a single global log-shift (median over all keypoints), with
the per-image spread reported as a sanity check. The GT-correct mode is the
majority argmin over a 3x3 GT patch (robust to which side of a discontinuity
the sub-pixel keypoint rounds to).

  python scripts/rung3_snapshot.py -s office0 \
      --maxmix-dir local/diag/office0-maxmix --base-dir local/diag/office0-base \
      --csv local/diag/office0_snapshot.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_last_ba(diag_dir):
    """Last dumped BA record per image name (files are seq-ordered)."""
    recs = {}
    for f in sorted(Path(diag_dir).glob("*_ba_*.npz")):
        d = dict(np.load(f))
        recs[str(d["name"])] = d
    if not recs:
        raise RuntimeError(f"no *_ba_*.npz dumps in {diag_dir}")
    return recs


def kp_pixels(kps, shape):
    x = np.clip(np.round(kps[:, 0]).astype(int), 0, shape[1] - 1)
    y = np.clip(np.round(kps[:, 1]).astype(int), 0, shape[0] - 1)
    return x, y


def align_to_gt(recs, gt_by_name):
    """Global log-shift a with log(gt) ~ log(d3d) + a; returns (a, per-image spread)."""
    diffs, per_im = [], []
    for name, d in recs.items():
        gt = gt_by_name[name]
        x, y = kp_pixels(d["kps"], gt.shape)
        g = gt[y, x]
        ok = (g > 0) & (d["d3d_post"] > 0)
        if ok.sum() < 10:
            continue
        dd = np.log(g[ok]) - np.log(d["d3d_post"][ok])
        per_im.append(np.median(dd))
        diffs.append(dd)
    a = float(np.median(np.concatenate(diffs)))
    return a, float(np.std(per_im))


def gt_mode_vote(gt, x, y, log_modes_gtframe):
    """Majority GT-mode over the 3x3 patch around each keypoint.

    Returns (gt_mode, clear, gt_center): winning mode index, whether >=78% of
    >=7 valid patch pixels agree, and the center-pixel GT depth.
    """
    n, K = log_modes_gtframe.shape
    votes = np.zeros((n, K), dtype=int)
    valid = np.zeros(n, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            px = gt[np.clip(y + dy, 0, gt.shape[0] - 1), np.clip(x + dx, 0, gt.shape[1] - 1)]
            ok = px > 0
            assign = np.abs(np.log(px.clip(1e-6, None))[:, None] - log_modes_gtframe).argmin(axis=1)
            for k in range(K):
                votes[:, k] += ok & (assign == k)
            valid += ok
    gt_mode = votes.argmax(axis=1)
    top = votes.max(axis=1)
    clear = (valid >= 7) & (top >= 0.78 * valid)
    return gt_mode, clear, gt[y, x]


def maxmix_score(log_d3d, log_modes, sigmas, weights):
    """Selection rule of the C++ factor: whitened^2 + 2 log(sigma/w)."""
    sig = sigmas.clip(1e-6, None)
    w = weights.clip(1e-6, None)
    return ((log_d3d[:, None] - log_modes) / sig) ** 2 + 2 * np.log(sig / w)


def build_rows(rec, gt, align):
    """Per-keypoint dict arrays for one image of one arm."""
    x, y = kp_pixels(rec["kps"], gt.shape)
    log_modes = np.log(rec["modes"].clip(1e-6, None)) + align  # -> GT frame
    log_d3d = np.log(rec["d3d_post"].clip(1e-6, None)) + align
    multi = np.abs(log_modes[:, 1] - log_modes[:, 0]) > 1e-9
    gt_mode, gt_clear, gt_center = gt_mode_vote(gt, x, y, log_modes)
    sel = maxmix_score(log_d3d, log_modes, rec["mode_sigmas"], rec["mode_weights"]).argmin(axis=1)
    nearest = np.abs(log_d3d[:, None] - log_modes).argmin(axis=1)
    return {
        "p2D": rec["p2Ds"],
        "x": x,
        "y": y,
        "multi": multi,
        "log_modes": log_modes,
        "sigmas": rec["mode_sigmas"],
        "weights": rec["mode_weights"],
        "var_cal": rec["variances"],
        "gt_center": gt_center,
        "gt_mode": gt_mode,
        "gt_clear": gt_clear & (gt_center > 0),
        "sel": sel,
        "nearest": nearest,
        "log_d3d": log_d3d,
        "log_obs": np.log(rec["obs_depths"].clip(1e-6, None)) + align,
        "log_prior": np.log(rec["prior_depths"].clip(1e-6, None)) + align,
        "in_ba": rec["in_ba"].astype(bool),
    }


def med(a):
    return float(np.median(a)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scene", default="office0")
    ap.add_argument("--maxmix-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--csv", help="Optional per-keypoint CSV output path")
    args = ap.parse_args()

    from mpsfm.data_proc.replica import ReplicaParser  # container-only imports

    parser = ReplicaParser(args.scene)
    name_to_gtimid = {im.name: imid for imid, im in parser.rec.images.items()}

    arms = {}
    for arm, ddir in [("maxmix", args.maxmix_dir), ("base", args.base_dir)]:
        recs = load_last_ba(ddir)
        factor = {str(d["depth_factor"]) for d in recs.values()}
        print(f"[{arm}] {len(recs)} images from {ddir} (depth_factor={sorted(factor)})")
        gt_by_name = {name: parser.gt_depth(name_to_gtimid[name]) for name in recs}
        align, spread = align_to_gt(recs, gt_by_name)
        print(f"[{arm}] GT alignment log-shift {align:+.4f} (scale {np.exp(align):.4f}x), per-image spread {spread:.4f}")
        arms[arm] = {
            name: build_rows(rec, gt_by_name[name], align) for name, rec in recs.items() if "modes" in rec
        }
        if not arms[arm]:
            raise RuntimeError(f"[{arm}] no mixture in dumps — was the config's depth.mixture on?")

    # match rows across arms by (name, p2D)
    joined = []
    for name in sorted(set(arms["maxmix"]) & set(arms["base"])):
        mm, bs = arms["maxmix"][name], arms["base"][name]
        common, mi, bi = np.intersect1d(mm["p2D"], bs["p2D"], return_indices=True)
        joined.append((name, mm, bs, mi, bi))
    n_match = sum(len(mi) for _, _, _, mi, _ in joined)
    print(f"\nmatched keypoints across arms: {n_match}")

    def pool(field, arm_idx):
        return np.concatenate([(mm if arm_idx == 0 else bs)[field][(mi if arm_idx == 0 else bi)]
                               for _, mm, bs, mi, bi in joined])

    multi = pool("multi", 0) & pool("multi", 1)
    clear = pool("gt_clear", 0) & pool("gt_clear", 1)
    gt_mode_mm, gt_mode_bs = pool("gt_mode", 0), pool("gt_mode", 1)
    agree = gt_mode_mm == gt_mode_bs
    eval_mask = multi & clear & agree
    print(f"bimodal (both arms): {multi.sum()} | GT-clear: {(multi & clear).sum()} "
          f"| arms agree on GT mode: {eval_mask.sum()}  <- evaluation set")

    sel, nearest = pool("sel", 0)[eval_mask], pool("nearest", 1)[eval_mask]
    gt_mode = gt_mode_mm[eval_mask]
    lm_mm = pool("log_modes", 0)[eval_mask]
    log_gt = np.log(pool("gt_center", 0)[eval_mask].clip(1e-6, None))
    ld3d_mm, ld3d_bs = pool("log_d3d", 0)[eval_mask], pool("log_d3d", 1)[eval_mask]
    n = eval_mask.sum()
    idx = np.arange(n)

    print("\n=== mode selection at bimodal, GT-clear keypoints ===")
    print(f"GT prefers alt (mode 1): {(gt_mode == 1).mean():.1%}")
    print(f"maxmix selected-mode accuracy:      {(sel == gt_mode).mean():.1%}")
    print(f"  selected alt: {(sel == 1).mean():.1%} | P(correct | selected alt): "
          f"{(gt_mode[sel == 1] == 1).mean() if (sel == 1).any() else float('nan'):.1%}"
          f" | P(correct | kept anchor): {(gt_mode[sel == 0] == 0).mean() if (sel == 0).any() else float('nan'):.1%}")
    print(f"unimodal converged-nearest accuracy: {(nearest == gt_mode).mean():.1%}")

    print("\n=== |log err| vs GT of converged depth (median) ===")
    print(f"maxmix d3d:   {med(np.abs(ld3d_mm - log_gt)):.4f}")
    print(f"unimodal d3d: {med(np.abs(ld3d_bs - log_gt)):.4f}")
    print(f"anchor mode:  {med(np.abs(lm_mm[idx, 0] - log_gt)):.4f}")
    print(f"oracle mode:  {med(np.abs(lm_mm[idx, gt_mode] - log_gt)):.4f}")

    print("\n=== what BA was fed (D* feedback), bimodal kps ===")
    for arm_idx, arm in [(0, "maxmix"), (1, "base")]:
        lo = pool("log_obs", arm_idx)[eval_mask]
        lp = pool("log_prior", arm_idx)[eval_mask]
        lm = pool("log_modes", arm_idx)[eval_mask]
        obs_on = np.abs(lo[:, None] - lm).argmin(axis=1)
        print(f"[{arm}] median |log obs - log prior| {med(np.abs(lo - lp)):.4f} "
              f"| obs nearest anchor: {(obs_on == 0).mean():.1%} "
              f"| in_ba rate: {pool('in_ba', arm_idx)[eval_mask].mean():.1%}")

    # control: unimodal keypoints
    uni = ~multi & clear
    if uni.sum():
        print(f"\ncontrol (unimodal kps, n={uni.sum()}): median |log err| "
              f"maxmix {med(np.abs(pool('log_d3d', 0)[uni] - np.log(pool('gt_center', 0)[uni].clip(1e-6, None)))):.4f} "
              f"| base {med(np.abs(pool('log_d3d', 1)[uni] - np.log(pool('gt_center', 1)[uni].clip(1e-6, None)))):.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["name", "p2D", "x", "y", "mu0_gtframe", "mu1_gtframe", "w0", "w1", "sig0", "sig1",
                         "var_cal", "gt_depth", "gt_mode", "gt_clear", "multi", "sel_maxmix", "d3d_maxmix",
                         "d3d_base", "obs_maxmix", "prior_maxmix", "in_ba_maxmix", "in_ba_base"])
            for name, mm, bs, mi, bi in joined:
                for i, j in zip(mi, bi):
                    wr.writerow([name, mm["p2D"][i], mm["x"][i], mm["y"][i],
                                 f"{np.exp(mm['log_modes'][i, 0]):.4f}", f"{np.exp(mm['log_modes'][i, 1]):.4f}",
                                 f"{mm['weights'][i, 0]:.3f}", f"{mm['weights'][i, 1]:.3f}",
                                 f"{mm['sigmas'][i, 0]:.4f}", f"{mm['sigmas'][i, 1]:.4f}",
                                 f"{mm['var_cal'][i]:.6f}", f"{mm['gt_center'][i]:.4f}",
                                 mm["gt_mode"][i], int(mm["gt_clear"][i] and bs["gt_clear"][j]),
                                 int(mm["multi"][i] and bs["multi"][j]), mm["sel"][i],
                                 f"{np.exp(mm['log_d3d'][i]):.4f}", f"{np.exp(bs['log_d3d'][j]):.4f}",
                                 f"{np.exp(mm['log_obs'][i]):.4f}", f"{np.exp(mm['log_prior'][i]):.4f}",
                                 int(mm["in_ba"][i]), int(bs["in_ba"][j])])
        print(f"\nper-keypoint CSV written to {args.csv}")


if __name__ == "__main__":
    main()
