import contextlib
from argparse import ArgumentParser

from tqdm import tqdm

from mpsfm.data_proc.scannetpp import ScanNetPPDataset
from mpsfm.eval.sfm.dataset_aggregators.scannetpp import ScanNetPPAggregator

parser = ArgumentParser()
parser.add_argument("-s", "--scenes", nargs="+", help="Scenes to aggregate (default: all with testsets)")
parser.add_argument("-c", "--configs", nargs="+", help="Configs to aggregate", default=["repr-sp-lg_m3dv2"])
parser.add_argument("-m", "--modes", nargs="+", help="Modes to aggregate", default=["all"])
args = parser.parse_args()

scenes = ScanNetPPDataset.scenes if args.scenes is None else args.scenes

out = {}
for mode in tqdm(args.modes):
    agg = ScanNetPPAggregator(scenes=scenes, sfm_configs=args.configs, mode=mode)
    out[mode] = agg.all_scenes()

print("Modes:".ljust(19), "     ".join(args.modes))
for conf in args.configs:
    with contextlib.suppress(Exception):
        print(f"{conf.ljust(15)}     {'     '.join([out[mode][conf] for mode in args.modes])}")
