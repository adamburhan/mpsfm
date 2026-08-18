"""Win/loss geometry anatomy under the TRACK key (pre-registered revision #2).

Pixel-key anatomy penalizes correct routing at structural keypoints (track on
the adjacent surface). Corrected reference depth per keypoint:
  - structural (near-alt in BOTH arms): GT surface matched to the ALT MODE by
    ring search (mode-seeded -> arm-neutral); no GT match => excluded, counted.
  - non-structural: GT at the pixel (unchanged).
Both keys are reported side by side.

  python scripts/rung3_anatomy_trackkey.py --glob "local/diag/replica-battery/*/snapshot.csv"
  python scripts/rung3_anatomy_trackkey.py --glob "local/diag/replica-battery/office*/csig_snapshot.csv"
"""

import argparse
import glob as globmod
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rung3_rescore import first_hit  # noqa: E402


def win_loss(em, eb, s):
    return int((em[s] < eb[s] - 0.01).sum()), int((eb[s] < em[s] - 0.01).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="snapshot CSVs (window dir name = <scene>-<testset>)")
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    from mpsfm.data_proc.replica import ReplicaParser  # container-only import

    parsers, gt_cache = {}, {}
    tot_px, tot_tr, tot_excl = [0, 0], [0, 0], 0
    print(f"{'window':>12} {'pixel-key':>12} {'track-key':>12} {'structural':>10} {'no-hit':>7}")
    for f in sorted(globmod.glob(args.glob)):
        scene = Path(f).parent.name.split("-")[0]
        if scene not in parsers:
            parsers[scene] = ReplicaParser(scene)
        parser = parsers[scene]
        r = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)

        ok = r["gt_depth"] > 0
        lgt_px = np.log(r["gt_depth"].clip(1e-6, None))
        lmu = np.stack([np.log(r["mu0_gtframe"]), np.log(r["mu1_gtframe"])], 1)
        lm, lb = np.log(r["d3d_maxmix"]), np.log(r["d3d_base"])
        multi, clear = r["multi"] == 1, r["gt_clear"] == 1
        near_alt = (np.abs(lb - lmu[:, 1]) < np.abs(lb - lmu[:, 0])) & (
            np.abs(lm - lmu[:, 1]) < np.abs(lm - lmu[:, 0])
        )
        structural = multi & clear & ok & near_alt
        dis = np.abs(lm - lb) > 0.02
        base_set = dis & ok & multi & clear

        # corrected reference: pixel GT by default, alt-mode-matched surface at structural kps
        ref = lgt_px.copy()
        excl = np.zeros(len(r), bool)
        for i in np.flatnonzero(structural & dis):
            name = str(r["name"][i])
            key = (scene, name)
            if key not in gt_cache:
                imid = next(im_i for im_i, im in parser.rec.images.items() if im.name == name)
                gt_cache[key] = parser.gt_depth(imid)
            hit_r, surf = first_hit(gt_cache[key], int(r["x"][i]), int(r["y"][i]), lmu[i, 1],
                                    args.radius, args.tol)
            if hit_r >= 0:
                ref[i] = surf
            else:
                excl[i] = True

        em_px, eb_px = np.abs(lm - lgt_px), np.abs(lb - lgt_px)
        em_tr, eb_tr = np.abs(lm - ref), np.abs(lb - ref)
        wpx, lpx = win_loss(em_px, eb_px, base_set)
        wtr, ltr = win_loss(em_tr, eb_tr, base_set & ~excl)
        tot_px[0] += wpx; tot_px[1] += lpx
        tot_tr[0] += wtr; tot_tr[1] += ltr
        tot_excl += int(excl.sum())
        print(f"{Path(f).parent.name:>12} {wpx:5d}/{lpx:<5d} {wtr:5d}/{ltr:<5d} "
              f"{int((structural & dis).sum()):10d} {int(excl.sum()):7d}")

    for lab, (w, l) in [("pixel-key", tot_px), ("track-key", tot_tr)]:
        n = w + l
        z = (w - n / 2) / max(np.sqrt(n / 4), 1e-9)
        print(f"\nPOOLED {lab}: {w}/{l} = {w / max(n, 1):.1%} (z={z:+.1f})", end="")
    print(f"   excluded (no GT hit): {tot_excl}")


if __name__ == "__main__":
    main()
