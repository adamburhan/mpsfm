"""ScanNet++ (DSLR) dataset and parser for MP-SfM pipeline."""

import torch

from mpsfm.vars import gvars, lvars

from .basedataset import BaseDataset, BaseDatasetParser


class ScanNetPPDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for ScanNet++ DSLR data."""

    data_dir = lvars.SCANNETPP_DATA_DIR
    default_exp_dir = lvars.SCANNETPP_EXP_DIR
    default_cache_dir = lvars.SCANNETPP_CACHE_DIR
    testsets = gvars.TESTSETS_DIR / "scannetpp"
    scenes = [el.name for el in testsets.iterdir()] if testsets.exists() else []


class ScanNetPPParser(BaseDatasetParser):
    """Parser for ScanNet++ DSLR dataset."""

    dataset = ScanNetPPDataset
