import yaml
from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["eth3d", "smerf"], default="eth3d")
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    path = Path(__file__).parent.parent / "local/testsets" / args.dataset
    for scene_dir in sorted(path.iterdir()):
        all_yaml = scene_dir / "all.yaml"
        if not scene_dir.is_dir() or not all_yaml.exists():
            continue
        out_yaml = scene_dir / "gs_test.yaml"
        if out_yaml.exists() and not args.overwrite:
            print(f"{scene_dir.name}: gs_test.yaml exists, skipping (frozen split)")
            continue

        with open(all_yaml) as f:
            testsets = yaml.safe_load(f)
        test_imids = sorted(testsets[0])[:: args.every]

        with open(out_yaml, "w") as f:
            yaml.safe_dump({0: test_imids}, f, default_flow_style=None)
        print(f"{scene_dir.name}: {len(test_imids)}/{len(sorted(testsets[0]))} test views")


if __name__ == "__main__":
    main()
