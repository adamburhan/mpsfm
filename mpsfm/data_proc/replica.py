"""Replica SLAM RGB-D dataset and parser for MP-SfM pipeline."""

import json

import numpy as np
import torch
from PIL import Image

from mpsfm.vars import gvars, lvars

from .basedataset import BaseDataset, BaseDatasetParser


class ReplicaDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for Replica SLAM RGB-D trajectories."""

    data_dir = lvars.REPLICA_DATA_DIR
    default_exp_dir = lvars.REPLICA_EXP_DIR
    default_cache_dir = lvars.REPLICA_CACHE_DIR
    testsets = gvars.TESTSETS_DIR / "replica"
    scenes = [el.name for el in testsets.iterdir()] if testsets.exists() else []


class ReplicaParser(BaseDatasetParser):
    """Parser for Replica scenes prepared by mpsfm/data_proc/prepare/replica.py."""

    dataset = ReplicaDataset

    def __init__(self, scene=None, *args, **kwargs):
        super().__init__(scene, *args, **kwargs)
        with open(self.dataset.data_dir / self.scene / "meta.json") as f:
            self.meta = json.load(f)

    def gt_depth(self, imid):
        """Canonical GT depth loader: metric z-depth (float64), 0 = invalid.

        PNG scale and z-depth semantics were verified at prep time (see meta.json).
        """
        depth_name = self.image_name(imid).replace("frame", "depth").replace(".jpg", ".png")
        arr = np.array(Image.open(self.rgb_dir / depth_name))
        # Pillow reads 16-bit PNGs as uint16 or int32 (mode 'I') depending on version
        assert arr.dtype in (np.uint16, np.int32), f"{depth_name}: expected 16-bit PNG, got {arr.dtype}"
        return arr.astype(np.float64) / self.meta["depth_png_scale"]
