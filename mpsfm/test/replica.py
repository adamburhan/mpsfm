from mpsfm.data_proc.replica import ReplicaDataset, ReplicaParser

from .basetest import BaseTest


class ReplicaTest(BaseTest):
    """Replica Test class for evaluating reconstruction performance on Replica mini-sequences."""

    dataset = ReplicaDataset
    parser = ReplicaParser

    def _auto_download(self):
        raise RuntimeError(
            "Replica cannot be auto-downloaded. Prepare it with "
            "mpsfm/data_proc/prepare/replica.py on the machine holding the raw dataset, "
            f"then place data under {self.dataset.data_dir} and testsets under {self.dataset.testsets}."
        )
