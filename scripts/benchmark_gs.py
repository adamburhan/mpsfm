"""One-click 3DGS benchmark stage.

Consumes sparse models saved by `scripts/benchmark.py --save_sparse`, then per
scene: builds a 3DGS source tree (images symlink + sparse + frozen test split),
trains gaussian splatting, renders the held-out views, computes metrics.

Every stage checks for its output and skips if present, so the script is safe
to relaunch after a crash and fast-forwards to where it died. Run inside the
container. Pin the GPU with CUDA_VISIBLE_DEVICES.

  python scripts/benchmark_gs.py -c repr-sp-lg_m3dv2 -m all

-c takes the config *name* (the directory name under reconstruction/, i.e. the
`name:` field of the sfm config), not the config file path.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pycolmap
import yaml

from mpsfm.test import get_test

GS_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "gaussian-splatting"
# 3DGS runs in its own env (torch cu128 matching system nvcc; see mpsfm-dev-setup notes)
GS_PYTHON = os.environ.get("GS_PYTHON", "/opt/gs/bin/python")


def run(cmd, cwd):
    print(f"+ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["eth3d", "smerf", "scannetpp"], default="eth3d")
    parser.add_argument("-c", "--conf", type=str, required=True, help="Config name, e.g. repr-sp-lg_m3dv2")
    parser.add_argument("-m", "--mode", type=str, default="all")
    parser.add_argument("-s", "--scene", type=str, help="Single scene (default: all scenes with a gs_test.yaml)")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("-t", "--terminate", action="store_true", help="Terminate on first error")
    args = parser.parse_args()

    test_cls = get_test(args.dataset)
    dataset = test_cls.dataset
    scenes = [args.scene] if args.scene else sorted(dataset.scenes)

    for scene in scenes:
        gs_test_yaml = dataset.testsets / scene / "gs_test.yaml"
        rec_dir = dataset.default_exp_dir / "reconstruction" / args.mode / scene / "0" / args.conf
        sparse_dir = rec_dir / "sparse" / "0"
        gs_dir = dataset.default_exp_dir / "gs" / args.mode / scene / "0" / args.conf
        src_dir = gs_dir / "source"
        model_dir = gs_dir / "model"

        if not gs_test_yaml.exists():
            print(f"{scene}: no gs_test.yaml, skipping")
            continue
        if not sparse_dir.exists():
            print(f"{scene}/{args.conf}: no sparse model at {sparse_dir} — run benchmark.py --save_sparse first")
            continue

        try:
            # stage 1: source tree (test.txt is written last = stage marker)
            test_names_file = src_dir / "sparse" / "0" / "test.txt"
            if not test_names_file.exists():
                src_dir.mkdir(parents=True, exist_ok=True)
                images_link = src_dir / "images"
                if not images_link.exists():
                    images_link.symlink_to(dataset.data_dir / scene / "images")
                if not (src_dir / "sparse" / "0").exists():
                    shutil.copytree(sparse_dir, src_dir / "sparse" / "0")
                    # mpsfm skips COLMAP's color-extraction pass, leaving all
                    # points3D black — 3DGS initializes SH from these colors
                    rec = pycolmap.Reconstruction(src_dir / "sparse" / "0")
                    rec.extract_colors_for_all_images(dataset.data_dir / scene / "images")
                    rec.write(src_dir / "sparse" / "0")
                    # gaussian-splatting caches a points3D.ply next to the bins
                    (src_dir / "sparse" / "0" / "points3D.ply").unlink(missing_ok=True)
                with open(gs_test_yaml) as f:
                    test_imids = yaml.safe_load(f)[0]
                scene_parser = test_cls.parser(scene)
                test_names = sorted(scene_parser.rec.images[imid].name for imid in test_imids)
                test_names_file.write_text("\n".join(test_names) + "\n")
                print(f"{scene}/{args.conf}: built source tree ({len(test_names)} test views)")

            # stage 2: train
            ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
            if not ply.exists():
                run(
                    [GS_PYTHON, "train.py", "-s", src_dir, "-m", model_dir, "--eval", "-r", "1",
                     "--iterations", args.iterations, "--save_iterations", args.iterations,
                     "--test_iterations", args.iterations, "--quiet", "--disable_viewer"],
                    cwd=GS_ROOT,
                )
            else:
                print(f"{scene}/{args.conf}: trained model exists, skipping")

            # stage 3: render held-out views
            renders_dir = model_dir / "test" / f"ours_{args.iterations}" / "renders"
            if not renders_dir.exists() or not any(renders_dir.iterdir()):
                run([GS_PYTHON, "render.py", "-m", model_dir, "--iteration", args.iterations, "--skip_train"],
                    cwd=GS_ROOT)
            else:
                print(f"{scene}/{args.conf}: renders exist, skipping")

            # stage 4: standard metrics (PSNR/SSIM/LPIPS over test renders).
            # Marker must be iteration-aware: results.json may exist from a
            # run at a different --iterations (metrics.py scores every
            # test/ours_* dir it finds, so rerunning refreshes all entries).
            results_file = model_dir / "results.json"
            if not results_file.exists() or f"ours_{args.iterations}" not in json.load(open(results_file)):
                run([GS_PYTHON, "metrics.py", "-m", model_dir], cwd=GS_ROOT)
            else:
                print(f"{scene}/{args.conf}: metrics exist, skipping")

        except Exception as e:
            if args.terminate:
                raise e
            print(f"{scene}/{args.conf}: FAILED — {e}")


if __name__ == "__main__":
    main()
