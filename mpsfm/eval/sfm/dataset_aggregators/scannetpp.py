from mpsfm.data_proc.scannetpp import ScanNetPPDataset
from mpsfm.eval.sfm.dataset_aggregators.base_dataset_aggegator import BaseDatasetAggregator
from mpsfm.eval.sfm.relative_pose import AggregateRelativePose
from mpsfm.test.scannetpp import ScanNetPPTest


class ScanNetPPAggregator(BaseDatasetAggregator):
    dataset = ScanNetPPDataset
    aggregation_approach = AggregateRelativePose
    dataset_benchmark = ScanNetPPTest
    default_conf = {}

    def _init(self, mode):
        self.recdescs = {}
        self.testset_type = f"{mode}"
        for scene in self.scenes:
            testset_dict = self.benchmark_obj.read_testsets(scene, self.testset_type)
            if testset_dict is None:
                print(f"{scene}-{mode} testset does not exist")
                continue
            self.recdescs[scene] = testset_dict["testsets"]
