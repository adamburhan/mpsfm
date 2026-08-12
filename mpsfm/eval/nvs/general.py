"""General NVS aggregation: PSNR/SSIM/LPIPS from gaussian-splatting's
results.json, per scene and aggregated per dataset (macro mean/median over
scenes), with render coverage against the frozen test split.

"""

import json
import statistics
from pathlib import Path

METRICS = ("PSNR", "SSIM", "LPIPS")


def scene_result(run_dir: Path, iterations: int = 30000):
    """Read one gs/<mode>/<scene>/<testset>/<conf> run. Returns None if the
    run hasn't produced metrics for this iteration count yet."""
    results_file = run_dir / "model" / "results.json"
    if not results_file.exists():
        return None
    metrics = json.load(open(results_file)).get(f"ours_{iterations}")
    if metrics is None:
        return None
    test_txt = run_dir / "source" / "sparse" / "0" / "test.txt"
    n_test = len(test_txt.read_text().split()) if test_txt.exists() else 0
    renders = run_dir / "model" / "test" / f"ours_{iterations}" / "renders"
    n_rendered = len(list(renders.glob("*.png"))) if renders.exists() else 0
    return {
        **{m: metrics[m] for m in METRICS},
        "n_rendered": n_rendered,
        "n_test": n_test,
    }


def collect(exp_dir: Path, mode: str, conf: str, scenes=None, iterations: int = 30000):
    """Rows for one config across scenes. scenes=None -> every scene dir found."""
    root = exp_dir / "gs" / mode
    if scenes is None:
        scenes = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    rows = {}
    for scene in scenes:
        for testset_dir in sorted((root / scene).glob("[0-9]*")):
            res = scene_result(testset_dir / conf, iterations)
            if res is not None:
                rows[f"{scene}/{testset_dir.name}"] = res
    return rows


def summarize(rows: dict):
    """Macro mean/median over scenes + total coverage."""
    if not rows:
        return None
    out = {"n_scenes": len(rows)}
    for m in METRICS:
        vals = [r[m] for r in rows.values()]
        out[f"{m}_mean"] = sum(vals) / len(vals)
        out[f"{m}_median"] = statistics.median(vals)
    out["rendered"] = sum(r["n_rendered"] for r in rows.values())
    out["test_views"] = sum(r["n_test"] for r in rows.values())
    return out
