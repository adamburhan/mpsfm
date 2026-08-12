"""Prepare ScanNet++ DSLR scenes for the MP-SfM benchmark layout.

Standalone (no mpsfm imports) so it can run on the cluster where the raw
dataset lives. For each scene it builds:

    <out>/data/<scene>/images   -> symlink to dslr/resized_undistorted_images
    <out>/data/<scene>/rec      -> COLMAP text model (PINHOLE camera from
                                   transforms_undistorted.json, poses for all
                                   non-bad train+test frames)
    <out>/testsets/<scene>/all.yaml      {0: [all image ids]}
    <out>/testsets/<scene>/gs_test.yaml  {0: [official NVS test-frame ids]}

Copy testsets/ into the repo's local/testsets/scannetpp/ and rsync data/
(with -L to follow the image symlinks) to the machine running the benchmark.

The GT model is written as COLMAP text files rather than through pycolmap's
Reconstruction API: pose setters on pycolmap Image objects are incompatible
across pycolmap versions, so pycolmap is only used to read (stable API).

Poses in transforms_undistorted.json are camera-to-world in the toolkit's
nerfstudio/instant-ngp convention (camera-axis flip + world-frame swap); we
invert that exactly and verify against dslr/colmap/images.txt per scene,
aborting the scene on disagreement. Recovering the COLMAP world frame matters
beyond the check itself: the scans/ mesh lives in that frame.

  python scannetpp.py --root ~/scratch/datasets/scannetpp --out ~/scratch/mpsfm_scannetpp \
      --scenes 09c1414f1b 0d2ee665be ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from PIL import Image

# The ScanNet++ toolkit (prepare_transforms_json) converts COLMAP c2w to
# nerfstudio/instant-ngp convention as:
#   c2w[0:3, 1:3] *= -1          -> right-multiply by CAM_FLIP (camera y,z flip)
#   c2w = c2w[[1, 0, 2, 3], :]   \
#   c2w[2, :] *= -1              -> left-multiply by WORLD_SWAP (world frame change)
# Both matrices are involutions, so undoing is applying them on the same sides.
CAM_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
WORLD_SWAP = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)


def cam_from_world(frame):
    """4x4 COLMAP world-to-camera matrix from a transforms_undistorted frame,
    inverting the toolkit's COLMAP->nerfstudio conversion (see above)."""
    c2w_ns = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w = WORLD_SWAP @ c2w_ns @ CAM_FLIP
    return np.linalg.inv(c2w)


def rotmat_to_quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), no scipy dependency."""
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
            [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
            [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
        ]
    ) / 3.0
    K = K + K.T - np.diag(np.diag(K))
    vals, vecs = np.linalg.eigh(K)
    x, y, z, w = vecs[:, np.argmax(vals)]
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


def write_text_model(rec_dir, meta, images):
    """images: list of (image_id, name, w2c 4x4). Empty points3D."""
    with open(rec_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {meta['w']} {meta['h']} {meta['fl_x']} {meta['fl_y']} {meta['cx']} {meta['cy']}\n")
    with open(rec_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for imid, name, w2c in images:
            qw, qx, qy, qz = rotmat_to_quat_wxyz(w2c[:3, :3])
            tx, ty, tz = w2c[:3, 3]
            f.write(f"{imid} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}\n\n")
    (rec_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    )


def verify_against_colmap(scene_dir, images):
    """Compare converted poses with the official COLMAP model (same poses,
    different convention/source file). Returns max rotation angle diff in deg."""
    ref = pycolmap.Reconstruction(scene_dir / "dslr" / "colmap")
    ref_by_name = {im.name: im for im in ref.images.values()}
    max_dr = 0.0
    for _, name, w2c in images:
        if name not in ref_by_name:
            continue
        cfw = ref_by_name[name].cam_from_world
        if callable(cfw):  # method vs property across pycolmap versions
            cfw = cfw()
        R_ref = cfw.rotation.matrix()
        cos = np.clip((np.trace(w2c[:3, :3] @ R_ref.T) - 1) / 2, -1, 1)
        max_dr = max(max_dr, np.degrees(np.arccos(cos)))
    return max_dr


def prepare_scene(scene_dir, out_data, out_testsets, overwrite):
    scene = scene_dir.name
    rec_dir = out_data / scene / "rec"
    if rec_dir.exists() and not overwrite:
        print(f"{scene}: exists, skipping")
        return

    with open(scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json") as f:
        meta = json.load(f)
    assert meta["camera_model"] == "PINHOLE", f"{scene}: unexpected camera model {meta['camera_model']}"
    assert all(meta[k] == 0.0 for k in ["k1", "k2", "k3", "k4"]), f"{scene}: nonzero distortion"

    images_dir = scene_dir / "dslr" / "resized_undistorted_images"
    # sanity: json resolution must match the pixels we serve
    w, h = Image.open(next(images_dir.iterdir())).size
    assert (w, h) == (meta["w"], meta["h"]), f"{scene}: json {meta['w']}x{meta['h']} vs images {w}x{h}"

    train = [fr for fr in meta["frames"] if not fr.get("is_bad", False)]
    test = [fr for fr in meta["test_frames"] if not fr.get("is_bad", False)]
    n_bad = len(meta["frames"]) + len(meta["test_frames"]) - len(train) - len(test)

    frames = sorted(train + test, key=lambda fr: fr["file_path"])
    test_names = {fr["file_path"] for fr in test}
    images, test_ids = [], []
    for imid, fr in enumerate(frames, start=1):
        images.append((imid, fr["file_path"], cam_from_world(fr)))
        if fr["file_path"] in test_names:
            test_ids.append(imid)

    max_dr = verify_against_colmap(scene_dir, images)
    if max_dr > 0.1:
        print(f"{scene}: POSE MISMATCH vs colmap model ({max_dr:.3f} deg) — SKIPPED, investigate")
        return

    rec_dir.mkdir(parents=True, exist_ok=True)
    write_text_model(rec_dir, meta, images)
    rec = pycolmap.Reconstruction(rec_dir)  # sanity: model parses
    assert rec.num_images() == len(images), f"{scene}: wrote {len(images)} images, read {rec.num_images()}"

    images_link = out_data / scene / "images"
    if not images_link.exists():
        images_link.symlink_to(images_dir.resolve())

    testset_dir = out_testsets / scene
    testset_dir.mkdir(parents=True, exist_ok=True)
    with open(testset_dir / "all.yaml", "w") as f:
        yaml.safe_dump({0: [imid for imid, _, _ in images]}, f, default_flow_style=None)
    with open(testset_dir / "gs_test.yaml", "w") as f:
        yaml.safe_dump({0: sorted(test_ids)}, f, default_flow_style=None)

    print(
        f"{scene}: {len(images)} images ({len(test_ids)} test, {n_bad} bad excluded), "
        f"pose check {max_dr:.4f} deg"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="ScanNet++ root (contains data/<scene>/)")
    parser.add_argument("--out", type=Path, required=True, help="Output root (gets data/ and testsets/)")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    for scene in args.scenes:
        prepare_scene(args.root / "data" / scene, args.out / "data", args.out / "testsets", args.overwrite)


if __name__ == "__main__":
    main()
