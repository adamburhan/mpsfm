"""Targeted (masked) NVS evaluation — mask construction + aggregation.

Edge mask = MP-SfM's own depth-continuity detector applied to the cached
monocular depth of each held-out test image, dilated to a band, intersected
with the extractor's valid mask. Arm-independent by construction: it depends
only on the image, so every SfM arm is scored on identical pixels.

The continuity test (ratio of neighboring inverse depths > 1.015) is
reimplemented verbatim from mpsfm.sfm.scene.image.utils so this module
imports without the SfM stack (cupy). It is equivalent to thresholding
|grad log D| at log(1.015) per pixel step.

Masks are written once per (mode, scene, testset), shared by all arms:
    gs/<mode>/<scene>/<testset>/masks/<image>.png   coded: 0=invalid, 128=rest, 255=edge
    gs/<mode>/<scene>/<testset>/masks/meta.json     params + per-view mask_frac
"""

import json
import statistics
from pathlib import Path

# numpy/cv2/h5py imported lazily inside the mask-building functions so that
# the aggregation half of this module stays importable on bare login-node pythons

RATIO_T = 1.015  # inherited from mpsfm get_continuity_mask, NOT a free parameter

INVALID, REST, EDGE = 0, 128, 255

TARGETED_METRICS = ("PSNR", "SSIM", "LPIPS")


def _fgbg_depth(d, t):
    # from mpsfm.sfm.scene.image.utils.fgbg_depth
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    return left_is_big_enough, top_is_big_enough, right_is_big_enough, bottom_is_big_enough


def continuity_mask(depth):
    # from mpsfm.sfm.scene.image.utils.get_continuity_mask
    import numpy as np

    continuity = np.ones_like(depth, dtype=bool)
    inv_depth = 1.0 / np.clip(depth, 1e-6, None)
    l1, t1, r1, b1 = [~el for el in _fgbg_depth(inv_depth, RATIO_T)]
    continuity[:, 1:] &= l1 & r1
    continuity[:, :-1] &= l1 & r1
    continuity[1:, :] &= t1 & b1
    continuity[:-1, :] &= t1 & b1
    return continuity


def build_view_mask(depth, valid, render_wh, band_frac):
    """Coded mask (uint8, render resolution) for one view from its mono depth."""
    import cv2
    import numpy as np

    edge = ~continuity_mask(depth)
    radius = max(1, round(band_frac * depth.shape[1]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)
    coded = np.full(depth.shape, REST, dtype=np.uint8)
    coded[band] = EDGE
    coded[~valid] = INVALID
    return cv2.resize(coded, render_wh, interpolation=cv2.INTER_NEAREST)


def build_testset_masks(testset_dir: Path, cache_scene_dir: Path, depth_h5: str = None,
                        band_frac: float = 0.01, overwrite: bool = False):
    """Build masks for all test views of one gs/<mode>/<scene>/<testset>.
    Uses any completed arm's source tree for the test list and image resolution."""
    import cv2
    import h5py
    import numpy as np

    masks_dir = testset_dir / "masks"
    meta_file = masks_dir / "meta.json"
    if meta_file.exists() and not overwrite:
        return json.load(open(meta_file))

    sources = sorted(testset_dir.glob("*/source/sparse/0/test.txt"))
    if not sources:
        return None  
    source_dir = sources[0].parents[2]
    test_names = sorted(sources[0].read_text().split())

    if depth_h5 is None:
        candidates = sorted(cache_scene_dir.glob("metric3dv2*.h5"))
        if not candidates:
            raise FileNotFoundError(f"no metric3dv2*.h5 in {cache_scene_dir}")
        depth_h5 = candidates[0].stem

    masks_dir.mkdir(parents=True, exist_ok=True)
    meta = {"ratio_t": RATIO_T, "band_frac": band_frac, "depth_h5": depth_h5, "mask_frac": {}}
    with h5py.File(cache_scene_dir / f"{depth_h5}.h5", "r") as f:
        for name in test_names:
            if name not in f:
                print(f"  {name}: missing in {depth_h5}.h5, skipping view")
                continue
            depth = f[name]["depth"][...].squeeze().astype(np.float64)
            valid = f[name]["valid"][...].squeeze().astype(bool) if "valid" in f[name] else np.ones(depth.shape, bool)
            image = cv2.imread(str(source_dir / "images" / name))
            if image is None:
                print(f"  {name}: source image unreadable, skipping view")
                continue
            coded = build_view_mask(depth, valid, (image.shape[1], image.shape[0]), band_frac)
            cv2.imwrite(str(masks_dir / f"{name}.png"), coded)
            n_valid = int((coded != INVALID).sum())
            meta["mask_frac"][name] = float((coded == EDGE).sum() / max(n_valid, 1))

    with open(meta_file, "w") as fp:
        json.dump(meta, fp, indent=2)
    return meta


def collect_targeted(exp_dir: Path, mode: str, conf: str, scenes=None, iterations: int = 30000):
    """Per-scene targeted rows for one config (reads targeted.json files)."""
    root = exp_dir / "gs" / mode
    if scenes is None:
        scenes = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    rows = {}
    for scene in scenes:
        for testset_dir in sorted((root / scene).glob("[0-9]*")):
            tfile = testset_dir / conf / "model" / "targeted.json"
            if not tfile.exists():
                continue
            entry = json.load(open(tfile)).get(f"ours_{iterations}")
            if entry is not None:
                rows[f"{scene}/{testset_dir.name}"] = entry
    return rows


def summarize_targeted(rows: dict):
    """Macro mean/median over scenes, per region."""
    if not rows:
        return None
    out = {"n_scenes": len(rows)}
    for region in ("edge", "rest"):
        for m in TARGETED_METRICS:
            vals = [r[region][m] for r in rows.values()]
            out[f"{region}_{m}_mean"] = sum(vals) / len(vals)
            out[f"{region}_{m}_median"] = statistics.median(vals)
    fracs = [r["mask_frac"] for r in rows.values()]
    out["mask_frac_mean"] = sum(fracs) / len(fracs)
    return out
