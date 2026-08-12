"""Targeted (masked) NVS evaluation over completed gs runs.

Two-interpreter orchestration (like benchmark_gs.py): masks are built in this
python (h5py/cv2, reads the cached mono depth), metrics run under GS_PYTHON
(torch + lpips). Idempotent: masks skip if meta.json exists, metrics skip if
targeted.json already has the ours_<iterations> entry.

Run inside the container/sif:
  python scripts/eval_nvs.py -d eth3d -c repr-sp-lg_m3dv2 repr-sp-lg_m3dv2-maxmix-comp
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpsfm.eval.nvs.targeted import build_testset_masks
from mpsfm.vars import lvars

GS_PYTHON = os.environ.get("GS_PYTHON", "/opt/gs/bin/python")
METRICS_SCRIPT = Path(__file__).resolve().parent.parent / "mpsfm" / "eval" / "nvs" / "targeted_metrics.py"

DIRS = {
    "eth3d": (lvars.ETH3D_EXP_DIR, lvars.ETH3D_CACHE_DIR),
    "smerf": (lvars.SMERF_EXP_DIR, lvars.SMERF_CACHE_DIR),
    "scannetpp": (lvars.SCANNETPP_EXP_DIR, lvars.SCANNETPP_CACHE_DIR),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=sorted(DIRS), default="eth3d")
    parser.add_argument("-c", "--configs", nargs="+", required=True)
    parser.add_argument("-m", "--mode", default="all")
    parser.add_argument("-s", "--scenes", nargs="+")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--band-frac", type=float, default=0.01)
    parser.add_argument("--depth-h5", type=str, help="cache h5 stem (default: first metric3dv2*.h5 found)")
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    exp_dir, cache_dir = DIRS[args.dataset]
    root = exp_dir / "gs" / args.mode
    scenes = args.scenes or (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else [])

    for scene in scenes:
        for testset_dir in sorted((root / scene).glob("[0-9]*")):
            meta = build_testset_masks(
                testset_dir, cache_dir / scene, depth_h5=args.depth_h5,
                band_frac=args.band_frac, overwrite=args.overwrite,
            )
            if meta is None:
                print(f"{scene}/{testset_dir.name}: no completed source tree yet, skipping")
                continue
            for conf in args.configs:
                run_dir = testset_dir / conf
                if not (run_dir / "model" / "test" / f"ours_{args.iterations}" / "renders").exists():
                    print(f"{scene}/{testset_dir.name}/{conf}: no renders yet, skipping")
                    continue
                tfile = run_dir / "model" / "targeted.json"
                if not args.overwrite and tfile.exists() and f"ours_{args.iterations}" in json.load(open(tfile)):
                    print(f"{scene}/{testset_dir.name}/{conf}: targeted.json exists, skipping")
                    continue
                subprocess.run(
                    [GS_PYTHON, str(METRICS_SCRIPT), "--run-dir", str(run_dir),
                     "--masks-dir", str(testset_dir / "masks"), "--iterations", str(args.iterations)],
                    check=True,
                )


if __name__ == "__main__":
    main()
