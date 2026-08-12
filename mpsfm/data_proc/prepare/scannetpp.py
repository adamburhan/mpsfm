"""Prepare ScanNet++ DSLR scenes for the MP-SfM benchmark layout.

Standalone (no mpsfm imports) so it can run on the cluster where the raw
dataset lives. For each scene it builds:

    <out>/data/<scene>/images   -> symlink to dslr/resized_undistorted_images
    <out>/data/<scene>/rec      -> pycolmap GT reconstruction (PINHOLE camera
                                   from transforms_undistorted.json, poses for
                                   all non-bad train+test frames)
    <out>/testsets/<scene>/all.yaml      {0: [all image ids]}
    <out>/testsets/<scene>/gs_test.yaml  {0: [official NVS test-frame ids]}

Copy testsets/ into the repo's local/testsets/scannetpp/ and rsync data/
(with -L to follow the image symlinks) to the machine running the benchmark.

Poses in transforms_undistorted.json are nerfstudio-convention camera-to-world
(OpenGL axes: y up, z back). We convert to COLMAP cam_from_world and verify
against dslr/colmap/images.txt per scene, aborting the scene on disagreement.

  python scannetpp.py --root ~/scratch/datasets/scannetpp --out ~/scratch/mpsfm_scannetpp \
      --scenes 09c1414f1b 0d2ee665be ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
import yaml

# right-multiply a nerfstudio/OpenGL c2w to get an OpenCV/COLMAP c2w
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def cam_from_world(frame):
    c2w = np.array(frame["transform_matrix"], dtype=np.float64) @ GL_TO_CV
    w2c = np.linalg.inv(c2w)
    return pycolmap.Rigid3d(w2c[:3])


def verify_against_colmap(scene_dir, images):
    """Compare converted poses with the official COLMAP model (same poses,
    different convention/source file). Returns max rotation angle diff in deg."""
    ref = pycolmap.Reconstruction(scene_dir / "dslr" / "colmap")
    ref_by_name = {im.name: im for im in ref.images.values()}
    max_dr = 0.0
    for im in images.values():
        if im.name not in ref_by_name:
            continue
        R_ours = im.cam_from_world.rotation.matrix()
        R_ref = ref_by_name[im.name].cam_from_world.rotation.matrix()
        cos = np.clip((np.trace(R_ours @ R_ref.T) - 1) / 2, -1, 1)
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
    sample = next(images_dir.iterdir())
    from PIL import Image

    w, h = Image.open(sample).size
    assert (w, h) == (meta["w"], meta["h"]), f"{scene}: json {meta['w']}x{meta['h']} vs images {w}x{h}"

    train = [fr for fr in meta["frames"] if not fr.get("is_bad", False)]
    test = [fr for fr in meta["test_frames"] if not fr.get("is_bad", False)]
    n_bad = len(meta["frames"]) + len(meta["test_frames"]) - len(train) - len(test)

    rec = pycolmap.Reconstruction()
    camera = pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=meta["w"],
        height=meta["h"],
        params=[meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]],
    )
    rec.add_camera(camera)

    frames = sorted(train + test, key=lambda fr: fr["file_path"])
    test_names = {fr["file_path"] for fr in test}
    test_ids = []
    for imid, fr in enumerate(frames, start=1):
        image = pycolmap.Image(image_id=imid, camera_id=1, name=fr["file_path"])
        image.cam_from_world = cam_from_world(fr)
        rec.add_image(image)
        if fr["file_path"] in test_names:
            test_ids.append(imid)

    max_dr = verify_against_colmap(scene_dir, rec.images)
    if max_dr > 0.1:
        print(f"{scene}: POSE MISMATCH vs colmap model ({max_dr:.3f} deg) — SKIPPED, investigate")
        return

    rec_dir.mkdir(parents=True, exist_ok=True)
    rec.write(rec_dir)
    images_link = out_data / scene / "images"
    if not images_link.exists():
        images_link.symlink_to(images_dir.resolve())

    testset_dir = out_testsets / scene
    testset_dir.mkdir(parents=True, exist_ok=True)
    with open(testset_dir / "all.yaml", "w") as f:
        yaml.safe_dump({0: sorted(rec.images.keys())}, f, default_flow_style=None)
    with open(testset_dir / "gs_test.yaml", "w") as f:
        yaml.safe_dump({0: sorted(test_ids)}, f, default_flow_style=None)

    print(
        f"{scene}: {len(frames)} images ({len(test_ids)} test, {n_bad} bad excluded), "
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
