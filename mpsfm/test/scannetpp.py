from mpsfm.data_proc.scannetpp import ScanNetPPDataset, ScanNetPPParser

from .basetest import BaseTest


class ScanNetPPTest(BaseTest):
    """ScanNet++ Test class for evaluating reconstruction performance on ScanNet++ DSLR testsets."""

    dataset = ScanNetPPDataset
    parser = ScanNetPPParser

    def _auto_download(self):
        raise RuntimeError(
            "ScanNet++ is license-gated and cannot be auto-downloaded. Prepare it with "
            "mpsfm/data_proc/prepare/scannetpp.py on the machine holding the raw dataset, "
            f"then place data under {self.dataset.data_dir} and testsets under {self.dataset.testsets}."
        )
