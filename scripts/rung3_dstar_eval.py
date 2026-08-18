"""Pre-registered D* endpoint: dense depth accuracy vs GT at discontinuity
bands, paired base vs bimodal. Reads dstar_<imid>.npz diag dumps (last per
imid wins), aligns each image to GT by median log-ratio over NON-band pixels,
reports band/rest |log err| medians + pixel-level win rates.

  python scripts/rung3_dstar_eval.py -s office0 \
      --base-dir local/diag/dstar/office0/base --bimodal-dir local/diag/dstar/office0/bimodal
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_last(ddir):
    out = {}
    for f in sorted(glob.glob(f"{ddir}/*_dstar_*.npz")):
        imid = int(Path(f).stem.split("_dstar_")[1])
        out[imid] = f
    return {k: dict(np.load(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scene", default="office0")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--bimodal-dir", required=True)
    ap.add_argument("--band-radius", type=int, default=4)
    args = ap.parse_args()

    from mpsfm.data_proc.replica import ReplicaParser

    parser = ReplicaParser(args.scene)
    base, bim = load_last(args.base_dir), load_last(args.bimodal_dir)
    imids = sorted(set(base) & set(bim))
    print(f"{args.scene}: {len(imids)} paired images")

    stats = {"band": [[], []], "rest": [[], []]}
    wins = np.zeros(2, dtype=np.int64)  # band pixels where [bimodal, base] is closer
    for imid in imids:
        gt_full = parser.gt_depth(imid)
        H, W = base[imid]["dstar"].shape
        iy = np.clip(np.round((np.arange(H) + 0.5) * gt_full.shape[0] / H - 0.5).astype(int), 0, gt_full.shape[0] - 1)
        ix = np.clip(np.round((np.arange(W) + 0.5) * gt_full.shape[1] / W - 0.5).astype(int), 0, gt_full.shape[1] - 1)
        gt = gt_full[iy[:, None], ix[None, :]]  # nearest sampling preserves edges
        rows = []
        for arm in (base, bim):
            d = arm[imid]["dstar"].astype(np.float64)
            assert d.shape == (H, W)
            cont = arm[imid].get("continuity")
            band = ndimage.binary_dilation(~cont, iterations=args.band_radius) if cont is not None else None
            ok = (d > 0) & (gt > 0)
            if band is None or ok.sum() < 1000:
                rows = None
                break
            e = np.log(gt[ok]) - np.log(d[ok])
            a = np.median(e[~band[ok]])  # align on smooth regions
            err = np.abs(np.log(gt) - np.log(d) - a)
            rows.append((err, band, ok))
        if rows is None:
            continue
        (eb, bandb, okb), (em, bandm, okm) = rows
        band = bandb & bandm & okb & okm  # shared band, both valid
        rest = ~bandb & ~bandm & okb & okm
        stats["band"][0].append(eb[band]); stats["band"][1].append(em[band])
        stats["rest"][0].append(eb[rest]); stats["rest"][1].append(em[rest])
        wins[0] += int((em[band] < eb[band] - 0.005).sum())
        wins[1] += int((eb[band] < em[band] - 0.005).sum())

    for region in ("band", "rest"):
        b = np.concatenate(stats[region][0]); m = np.concatenate(stats[region][1])
        print(f"{region:>5}: n={len(b)/1e6:.2f}M  base med {np.median(b):.4f}  bimodal med {np.median(m):.4f}  "
              f"delta {np.median(b) - np.median(m):+.4f}  p90 {np.percentile(b, 90):.4f}->{np.percentile(m, 90):.4f}")
    n = wins.sum()
    print(f"band pixel win rate (bimodal closer, >0.5% margin): {wins[0]}/{n} = {wins[0]/max(n,1):.1%}")


if __name__ == "__main__":
    main()
