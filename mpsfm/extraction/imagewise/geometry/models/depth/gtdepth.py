"""Cache-only GT-depth "extractor" for oracle experiments: the per-scene
gtdepth.h5 must be pre-populated (scripts/write_gt_depth_replica.py), after
which extraction finds every image cached and never instantiates this model.
It exists only to fail loudly if the cache is incomplete."""

from mpsfm.extraction.base_model import BaseModel


class GTDepth(BaseModel):
    """Placeholder model; gtdepth priors are never extracted at runtime."""

    default_conf = {"require_download": False}

    def _init(self, conf):
        raise RuntimeError(
            "gtdepth is cache-only: images are missing from <cache_dir>/<scene>/gtdepth.h5. "
            "Pre-populate it with scripts/write_gt_depth_replica.py before running."
        )

    def _forward(self, data):
        raise NotImplementedError
