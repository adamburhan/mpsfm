"""Aggregate general NVS metrics across scenes for one or more configs.

Login-node friendly (stdlib + lvars paths only, no torch/container).

  python scripts/aggregate_nvs.py -d scannetpp -c repr-sp-lg_m3dv2 repr-sp-lg_m3dv2-maxmix-comp
  python scripts/aggregate_nvs.py -d eth3d -c repr-sp-lg_m3dv2 --per-scene
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpsfm.eval.nvs.general import METRICS, collect, summarize
from mpsfm.eval.nvs.targeted import collect_targeted, summarize_targeted
from mpsfm.vars import lvars

EXP_DIRS = {
    "eth3d": lvars.ETH3D_EXP_DIR,
    "smerf": lvars.SMERF_EXP_DIR,
    "scannetpp": lvars.SCANNETPP_EXP_DIR,
}

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", choices=sorted(EXP_DIRS), default="eth3d")
parser.add_argument("-c", "--configs", nargs="+", required=True)
parser.add_argument("-m", "--mode", default="all")
parser.add_argument("-s", "--scenes", nargs="+", help="Subset of scenes (default: all found)")
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--per-scene", action="store_true", help="Print per-scene rows too")
parser.add_argument("--targeted", action="store_true", help="Aggregate targeted.json (Edge/Rest) instead of results.json")
args = parser.parse_args()

if args.targeted:
    for conf in args.configs:
        rows = collect_targeted(EXP_DIRS[args.dataset], args.mode, conf, scenes=args.scenes, iterations=args.iterations)
        if args.per_scene:
            for key, r in sorted(rows.items()):
                print(f"{key:<16} {conf:<30} " + "  ".join(
                    f"{reg[:1].upper()}.{m} {r[reg][m]:6.3f}" for reg in ("edge", "rest") for m in METRICS
                ) + f"  frac {r['mask_frac']:.3f}")
        s = summarize_targeted(rows)
        if s is None:
            print(f"{conf:<30} no targeted results found")
            continue
        for reg in ("edge", "rest"):
            print(f"{conf:<30} {reg:<5} n={s['n_scenes']:<3} " + "  ".join(
                f"{m} {s[f'{reg}_{m}_mean']:6.3f}/{s[f'{reg}_{m}_median']:6.3f}" for m in METRICS
            ) + (f"  mask_frac {s['mask_frac_mean']:.3f}" if reg == "edge" else "") + "  (mean/median)")
    raise SystemExit

for conf in args.configs:
    rows = collect(EXP_DIRS[args.dataset], args.mode, conf, scenes=args.scenes, iterations=args.iterations)
    if args.per_scene:
        for key, r in sorted(rows.items()):
            flag = "  <-- coverage!" if r["n_rendered"] != r["n_test"] else ""
            print(
                f"{key:<16} {conf:<30} "
                + "  ".join(f"{m} {r[m]:6.3f}" for m in METRICS)
                + f"  {r['n_rendered']}/{r['n_test']}{flag}"
            )
    s = summarize(rows)
    if s is None:
        print(f"{conf:<30} no completed runs found")
        continue
    print(
        f"{conf:<30} n={s['n_scenes']:<3} "
        + "  ".join(f"{m} {s[f'{m}_mean']:6.3f}/{s[f'{m}_median']:6.3f}" for m in METRICS)
        + f"  coverage {s['rendered']}/{s['test_views']}  (mean/median)"
    )
