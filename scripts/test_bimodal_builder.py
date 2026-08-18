"""Known-answer tests for the bimodal integration candidate builder.

Synthetic step scene with a smear ramp, a speckle outlier, and a flat scene.
Run in the container:  python scripts/test_bimodal_builder.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mpsfm.sfm.scene.image.integration import Integration  # noqa: E402


class Stub:
    class conf:
        bimodal_radius = 4
        bimodal_sep_min = 0.1
        bimodal_contour_dilation = 1

    class depth:
        pass


def make_stub(d, cont):
    s = Stub()
    s.depth = Stub.depth()
    s.depth.data_prior = d
    s.depth.valid = np.ones_like(d, bool)
    s.depth.continuity_mask = cont
    return s


def continuity_from(d):
    gx = np.abs(np.diff(np.log(d), axis=1, prepend=np.log(d[:, :1])))
    gy = np.abs(np.diff(np.log(d), axis=0, prepend=np.log(d[:1])))
    return ~((gx > 0.0149) | (gy > 0.0149))


def step_scene(H=80, W=120, edge=60, ramp=6):
    d = np.full((H, W), 2.0)
    d[:, edge:] = 5.0
    for i in range(ramp):  # linear smear across the edge
        d[:, edge - ramp // 2 + i] = 2.0 + (5.0 - 2.0) * (i + 0.5) / ramp
    return d


def main():
    # --- step: own de-smears, other crosses ---
    d = step_scene()
    cands = Integration._build_alt_priors(make_stub(d, continuity_from(d)))
    assert cands is not None
    own, other = cands
    left_band = np.s_[10:70, 52:56]  # clean-left side of the band
    assert np.abs(np.log(own[left_band] / 2.0)).max() < 0.02, "own not de-smeared on left band"
    cross = np.abs(np.log(other[left_band] / 5.0)) < 0.05
    assert cross.mean() > 0.9, f"other-surface coverage {cross.mean():.0%} on left band"
    far = np.s_[:, :40]
    assert (own[far] == d[far]).all() and (other[far] == d[far]).all(), "off-band candidates must equal raw"
    print("step scene PASSED")

    # --- speckle: small outlier blob must not become a candidate source ---
    d2 = step_scene()
    d2[30:33, 20:23] = 0.5
    cands = Integration._build_alt_priors(make_stub(d2, continuity_from(d2)))
    own, other = cands
    near_speckle = np.s_[25:40, 14:30]
    for name, c in [("own", own), ("other", other)]:
        contaminated = (np.abs(c[near_speckle] - 0.5) < 0.01) & (np.abs(d2[near_speckle] - 0.5) > 0.01)
        assert contaminated.sum() == 0, f"{name}: {contaminated.sum()} pixels contaminated by speckle"
    print("speckle immunity PASSED")

    # --- flat scene: no candidates ---
    d3 = np.full((80, 120), 3.0)
    assert Integration._build_alt_priors(make_stub(d3, continuity_from(d3))) is None
    print("flat scene PASSED")
    print("ALL BUILDER TESTS PASSED")


if __name__ == "__main__":
    main()
