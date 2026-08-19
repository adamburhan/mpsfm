"""Write Replica GT depth into the extraction cache as the cache-only
'gtdepth' depth prior (<cache_dir>/<scene>/gtdepth.h5), for the GT-depth
oracle BA arms (extractors.depth: gtdepth).

Covers the union of frames over each scene's testset yamls, so the h5 stays
small even though data/<scene>/images symlinks the full trajectory. The
written depth_variance is a nominal (0.03*d)^2; the oracle BA arms override
all sigmas via ba.fixed_log_sigma anyway.

Run inside the container (needs pycolmap for the scene parsers):
  python scripts/write_gt_depth_replica.py            # all scenes, all modes
  python scripts/write_gt_depth_replica.py -s room0 office3 -m mini
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpsfm.data_proc.replica import ReplicaDataset, ReplicaParser

REL_SIGMA = 0.03

parser = ArgumentParser()
parser.add_argument("-s", "--scenes", nargs="+", help="Subset of scenes (default: all with testsets)")
parser.add_argument("-m", "--mode", help="Only frames from this testset yaml (default: union of all modes)")
parser.add_argument("-o", "--overwrite", action="store_true")
args = parser.parse_args()

scenes = args.scenes or ReplicaDataset.scenes
for scene in scenes:
    testset_dir = ReplicaDataset.testsets / scene
    yamls = [testset_dir / f"{args.mode}.yaml"] if args.mode else sorted(testset_dir.glob("*.yaml"))
    imids = set()
    for y in yamls:
        with open(y) as f:
            for window in yaml.safe_load(f).values():
                imids.update(window)

    scene_parser = ReplicaParser(scene)
    out_dir = ReplicaDataset.default_cache_dir / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with h5py.File(out_dir / "gtdepth.h5", "a", libver="latest") as fd:
        for imid in sorted(imids):
            name = Path(scene_parser.image_name(imid)).name
            if name in fd:
                if not args.overwrite:
                    continue
                del fd[name]
            depth = scene_parser.gt_depth(imid).astype(np.float32)
            grp = fd.create_group(name)
            grp.create_dataset("depth", data=depth)
            grp.create_dataset("depth_variance", data=(REL_SIGMA * depth) ** 2)
            grp.create_dataset("valid", data=depth > 0)
            n_written += 1
    print(f"{scene}: {n_written}/{len(imids)} frames written to {out_dir / 'gtdepth.h5'}")
