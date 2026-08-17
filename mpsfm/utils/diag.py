"""Env-gated diagnostic dumps. All hooks are no-ops unless MPSFM_DIAG_DIR is set."""

import os
from pathlib import Path

import numpy as np

_seq = 0


def enabled():
    return "MPSFM_DIAG_DIR" in os.environ


def dump(tag, **data):
    """Write one compressed npz event; None values are skipped.

    Files are prefixed with a global sequence number so per-keypoint traces
    can be ordered causally across BA calls.
    """
    global _seq
    if not enabled():
        return
    out = Path(os.environ["MPSFM_DIAG_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{_seq:06d}_{tag}.npz", **{k: v for k, v in data.items() if v is not None})
    _seq += 1
