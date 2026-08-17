"""One-shot aggregate tables for a set of arms: pose AUC, full-frame NVS,
and targeted Edge/Rest NVS. Skips gracefully where a stage wasn't run.

Login-node friendly (stdlib + lvars paths only, no torch/container).

  python scripts/aggregate_all.py -d scannetpp -c repr-sp-lg_m3dv2 repr-sp-lg_m3dv2-maxmix-gates
  python scripts/aggregate_all.py -d eth3d -m leq30 -c repr-sp-lg_m3dv2 repr-sp-lg_m3dv2-maxmix
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpsfm.eval.nvs.general import METRICS, collect, summarize
from mpsfm.eval.nvs.targeted import collect_targeted, summarize_targeted
from mpsfm.vars import lvars

EXP_DIRS = {
    "eth3d": lvars.ETH3D_EXP_DIR,
    "smerf": lvars.SMERF_EXP_DIR,
    "scannetpp": lvars.SCANNETPP_EXP_DIR,
    "replica": lvars.REPLICA_EXP_DIR,
}

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", choices=sorted(EXP_DIRS), default="eth3d")
parser.add_argument("-c", "--configs", nargs="+", required=True)
parser.add_argument("-m", "--mode", default="all")
parser.add_argument("-s", "--scenes", nargs="+", help="Subset of scenes (default: all found)")
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--per-scene", action="store_true")
args = parser.parse_args()

exp_dir = EXP_DIRS[args.dataset]
width = max(len(c) for c in args.configs) + 2


def pose_rows(conf):
    """AUC columns per scene/testset from summary.txt (one row per testset)."""
    rows = {}
    for f in (exp_dir / "reconstruction" / args.mode).glob(f"*/*/{conf}/results/summary.txt"):
        scene, testset = f.parts[-5], f.parts[-4]
        if args.scenes and scene not in args.scenes:
            continue
        rows[f"{scene}/{testset}"] = [float(x) for x in f.read_text().split("|")]
    return rows


print("=== pose AUC (mean/median over testsets) ===")
for conf in args.configs:
    rows = pose_rows(conf)
    if not rows:
        print(f"{conf:<{width}} no results found")
        continue
    if args.per_scene:
        for key in sorted(rows):
            print(f"  {key:<20} " + " | ".join(f"{v:6.2f}" for v in rows[key]))
    cols = list(zip(*rows.values()))
    print(f"{conf:<{width}} n={len(rows):<4}" + "  ".join(f"{mean(c):6.2f}/{median(c):6.2f}" for c in cols))

print("\n=== NVS (full-frame, mean/median over scenes) ===")
for conf in args.configs:
    rows = collect(exp_dir, args.mode, conf, scenes=args.scenes, iterations=args.iterations)
    s = summarize(rows)
    if s is None:
        print(f"{conf:<{width}} no results found")
        continue
    print(f"{conf:<{width}} n={s['n_scenes']:<4}" + "  ".join(
        f"{m} {s[f'{m}_mean']:6.3f}/{s[f'{m}_median']:6.3f}" for m in METRICS
    ) + f"  coverage {s['rendered']}/{s['test_views']}")

print("\n=== targeted NVS (Edge/Rest, mean/median over scenes) ===")
for conf in args.configs:
    rows = collect_targeted(exp_dir, args.mode, conf, scenes=args.scenes, iterations=args.iterations)
    s = summarize_targeted(rows)
    if s is None:
        print(f"{conf:<{width}} no targeted results found")
        continue
    for reg in ("edge", "rest"):
        print(f"{conf:<{width}} {reg:<5} n={s['n_scenes']:<4}" + "  ".join(
            f"{m} {s[f'{reg}_{m}_mean']:6.3f}/{s[f'{reg}_{m}_median']:6.3f}" for m in METRICS
        ) + (f"  mask_frac {s['mask_frac_mean']:.3f}" if reg == "edge" else ""))
