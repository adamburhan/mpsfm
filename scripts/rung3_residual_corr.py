"""Spatial correlation of mono-prior residuals.

Measures whether the network's depth errors are spatially correlated within
surfaces (the independence assumption BA makes when summing per-point depth
residuals). Primary field: prior residual e = log(gt) - log(prior) at keypoints
(no mode-fit machinery). Same-surface pair proxy: |log gt_i - log gt_j| < 0.05;
cross-surface: > 0.15. Also reports the differential dispersion
std(e_i - e_j)/sqrt(2) per bin — the quantity a correlated model would trust.

  python scripts/rung3_residual_corr.py --glob "local/diag/replica-battery/*/snapshot.csv"
"""

import argparse
import glob as globmod
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BINS = [(2, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
MAX_KPS = 1500  # per image subsample cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--demean", action="store_true", help="remove per-image mean residual first")
    args = ap.parse_args()

    pairs = {b: {"same": [[], []], "cross": [[], []]} for b in BINS}
    n_img, marg = 0, []
    for f in sorted(globmod.glob(args.glob)):
        r = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
        for name in np.unique(r["name"]):
            s = (r["name"] == name) & (r["gt_depth"] > 0) & (r["prior_maxmix"] > 0)
            if s.sum() < 50:
                continue
            n_img += 1
            idx = np.flatnonzero(s)
            if len(idx) > MAX_KPS:
                idx = idx[np.linspace(0, len(idx) - 1, MAX_KPS).astype(int)]
            x, y = r["x"][idx].astype(float), r["y"][idx].astype(float)
            lgt = np.log(r["gt_depth"][idx])
            e = lgt - np.log(r["prior_maxmix"][idx])
            if args.demean:
                e = e - np.median(e)
            marg.append(e)
            tree = cKDTree(np.stack([x, y], 1))
            for lo, hi in BINS:
                ii, jj = np.array(sorted(tree.query_pairs(hi))).T if tree.query_pairs(hi) else (np.array([], int),) * 2
                if len(ii) == 0:
                    continue
                d = np.hypot(x[ii] - x[jj], y[ii] - y[jj])
                inbin = d >= lo
                ii, jj = ii[inbin], jj[inbin]
                dg = np.abs(lgt[ii] - lgt[jj])
                for key, m in [("same", dg < 0.05), ("cross", dg > 0.15)]:
                    pairs[(lo, hi)][key][0].append(e[ii[m]])
                    pairs[(lo, hi)][key][1].append(e[jj[m]])

    marg = np.concatenate(marg)
    print(f"images: {n_img} | marginal residual: median {np.median(marg):+.4f}  "
          f"robust sigma {1.4826 * np.median(np.abs(marg - np.median(marg))):.4f}")
    print(f"\n{'dist(px)':>10} {'n_same':>9} {'rho_same':>9} {'diff_disp':>10} {'n_cross':>9} {'rho_cross':>10}")
    for b in BINS:
        row = [f"{b[0]}-{b[1]}".rjust(10)]
        for key in ["same", "cross"]:
            a = np.concatenate(pairs[b][key][0]) if pairs[b][key][0] else np.array([])
            c = np.concatenate(pairs[b][key][1]) if pairs[b][key][1] else np.array([])
            if len(a) < 200:
                row += ["-".rjust(9), "-".rjust(9 if key == "same" else 10)]
                if key == "same":
                    row.insert(3, "-".rjust(10))
                continue
            rho = float(np.corrcoef(a, c)[0, 1])
            if key == "same":
                diff = 1.4826 * np.median(np.abs((a - c) - np.median(a - c))) / np.sqrt(2)
                row += [f"{len(a):9d}", f"{rho:9.3f}", f"{diff:10.4f}"]
            else:
                row += [f"{len(a):9d}", f"{rho:10.3f}"]
        print(" ".join(row))
    print("\npre-registered CONFIRM: rho_same(8-16) > 0.5 decaying with r; rho_cross(8-16) < 0.15;"
          "\n                        diff_disp(8-16) << marginal sigma (differentials clean).")


if __name__ == "__main__":
    main()
