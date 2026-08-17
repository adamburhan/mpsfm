"""Re-score mode selection against the track's verified GT surface (searched in a
small window around the kp) instead of the pixel's GT vote. First-hit radius
measures the detection/matching offset; rendered GT has sharp edges, so a
depth match certifies a real surface (floating tracks match nothing).

  python scripts/rung3_rescore.py -s office0 --csv local/diag/replica-mini/office0_snapshot.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def first_hit(gt, x, y, log_d, radius, tol):
    """Smallest Chebyshev radius r <= radius with a valid GT pixel matching
    log_d within tol; returns (r, log depth of best pixel at that ring) or (-1, nan)."""
    H, W = gt.shape
    for r in range(radius + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        patch = gt[y0:y1, x0:x1]
        valid = patch > 0
        if not valid.any():
            continue
        diffs = np.abs(np.log(patch[valid]) - log_d)
        i = diffs.argmin()
        if diffs[i] < tol:
            return r, float(np.log(patch[valid][i]))
    return -1, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scene", default="office0")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    from mpsfm.data_proc.replica import ReplicaParser

    parser = ReplicaParser(args.scene)
    name_to_imid = {im.name: imid for imid, im in parser.rec.images.items()}

    r = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=None, encoding=None)
    m = (r["multi"] == 1) & (r["gt_clear"] == 1)
    lmu = np.stack([np.log(r["mu0_gtframe"]), np.log(r["mu1_gtframe"])], 1)
    ld_mm, ld_bs = np.log(r["d3d_maxmix"]), np.log(r["d3d_base"])
    near_alt = (np.abs(ld_bs - lmu[:, 1]) < np.abs(ld_bs - lmu[:, 0])) & (
        np.abs(ld_mm - lmu[:, 1]) < np.abs(ld_mm - lmu[:, 0])
    )
    structural = m & near_alt
    anchor_sit = m & ~near_alt

    gt_maps = {name: parser.gt_depth(name_to_imid[name]) for name in np.unique(r["name"][m])}

    hit_r = np.full(len(r), -2)
    surf = np.full(len(r), np.nan)
    for i in np.flatnonzero(m):
        hit_r[i], surf[i] = first_hit(
            gt_maps[r["name"][i]], int(r["x"][i]), int(r["y"][i]), ld_mm[i], args.radius, args.tol
        )

    print(f"eval set: {m.sum()} bimodal GT-clear kps | structural (near-alt both arms): {structural.sum()} "
          f"| anchor-sitting: {anchor_sit.sum()}")

    print(f"\n=== GT-surface verification of the track (tol {args.tol}, max radius {args.radius}) ===")
    for lab, s in [("structural", structural), ("anchor-sitting", anchor_sit)]:
        hr = hit_r[s]
        print(f"{lab:>15}: verified {np.mean(hr >= 0):.1%} | hit-radius histogram "
              + " ".join(f"r{k}:{(hr == k).sum()}" for k in range(args.radius + 1))
              + f" none:{(hr == -1).sum()}")

    ver = structural & (hit_r >= 0)
    corrected = np.abs(surf[:, None] - lmu).argmin(axis=1)
    sel = r["sel_maxmix"]
    sep = np.abs(lmu[:, 1] - lmu[:, 0])

    print("\n=== does the ALT mode capture the verified surface? (structural, verified) ===")
    alt_match = np.abs(lmu[ver, 1] - surf[ver]) < 0.1
    print(f"|log mu1 - log surface| < 0.1 at {alt_match.mean():.1%} of {ver.sum()} kps "
          f"(median {np.median(np.abs(lmu[ver, 1] - surf[ver])):.4f})")

    print("\n=== selector re-scored (maxmix arm) ===")
    old_key, new_key = r["gt_mode"].astype(int), np.where(ver, corrected, r["gt_mode"].astype(int))
    for lab, s in [("structural kps", structural), ("all bimodal GT-clear", m)]:
        print(f"{lab:>22}: pixel-key acc {(sel[s] == old_key[s]).mean():.1%}  ->  "
              f"track-key acc {(sel[s] == new_key[s]).mean():.1%}")
    mid = m & (sep >= 0.1) & (sep < 0.2)
    alt = mid & (sel == 1)
    print(f"mid-bin [0.1,0.2) P(ok|fired alt): pixel-key {(old_key[alt] == 1).mean():.1%}  ->  "
          f"track-key {(new_key[alt] == 1).mean():.1%}  (n fired {alt.sum()})")
    print(f"mid-bin GT-alt rate: pixel-key {(old_key[mid] == 1).mean():.1%} -> track-key {(new_key[mid] == 1).mean():.1%}")


if __name__ == "__main__":
    main()
