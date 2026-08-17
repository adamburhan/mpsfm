"""Prepare Replica SLAM RGB-D trajectories for the MP-SfM benchmark layout.

Standalone (no mpsfm imports) so it can run on the machine holding the raw
dataset. For each scene it builds:

    <out>/data/<scene>/images    -> symlink to <root>/<scene>/results
                                    (frame%06d.jpg + depth%06d.png live together)
    <out>/data/<scene>/rec       -> COLMAP text model (PINHOLE camera from
                                    cam_params.json, poses for ALL frames)
    <out>/data/<scene>/meta.json -> verified depth PNG scale, traj convention,
                                    convention-check numbers (read by ReplicaParser)
    <out>/testsets/<scene>/<mode>.yaml   {0: [imids of the frozen frame window]}

Copy testsets/ into the repo's local/testsets/replica/ and place data/ under
local/benchmarks/replica/data (symlinks resolve on the same machine).

Image ids are frame_index + 1 (COLMAP ids must be >= 1); pose eval matches by
image name, so the offset is cosmetic. The GT model contains every frame so new
testset windows only need a new yaml, not a re-prep.

traj.txt convention (camera-to-world vs world-to-camera) and the depth PNG
scale are NOT assumed: both are validated jointly by cross-frame depth
consistency — back-project GT depth of frame i (z-depth semantics), transform
into frame j under each convention hypothesis, and compare projected z against
frame j's GT depth. The correct hypothesis yields sub-percent median log error;
the wrong one is catastrophic. A scene is aborted unless there is a clear
winner. The same check validates z-depth (vs ray-distance) semantics and the
metric consistency of depth with the trajectory translations.

  python replica.py --root ~/scratch/datasets/replica/Replica --out <bench_root> \
      --scenes office0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

CANONICAL_START, CANONICAL_STRIDE, CANONICAL_NUM = 500, 5, 20


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


def find_cam_params(args, root, scene_dir):
    candidates = [args.cam_params] if args.cam_params else []
    candidates += [root / "cam_params.json", root.parent / "cam_params.json", scene_dir / "cam_params.json"]
    for path in candidates:
        if path is not None and Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            cam = data.get("camera", data)
            missing = [k for k in ["w", "h", "fx", "fy", "cx", "cy", "scale"] if k not in cam]
            if missing:
                raise RuntimeError(f"{path}: missing camera keys {missing}")
            return cam, path
    raise RuntimeError(
        f"cam_params.json not found near {root} — pass --cam-params <path> "
        "(expects keys w h fx fy cx cy scale, optionally nested under 'camera')"
    )


def load_depth(path, scale):
    """Raw depth PNG -> metric z-depth (0 = invalid). dtype is asserted, not assumed.

    Pillow reads 16-bit grayscale PNGs as uint16 or int32 (mode 'I') depending
    on version; both are accepted, but values must be in 16-bit range.
    """
    arr = np.array(Image.open(path))
    assert arr.dtype in (np.uint16, np.int32), f"{path}: expected 16-bit depth PNG, got {arr.dtype}"
    assert arr.min() >= 0 and arr.max() <= 65535, f"{path}: values outside 16-bit range"
    return arr.astype(np.float64) / scale


def cross_depth_error(K, w2c_i, w2c_j, depth_i, depth_j, stride=8):
    """Median |log| depth-consistency error projecting frame i's GT depth into frame j."""
    h, w = depth_i.shape
    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    us, vs = us.ravel(), vs.ravel()
    d = depth_i[vs, us]
    ok = d > 0
    us, vs, d = us[ok], vs[ok], d[ok]
    rays = np.linalg.inv(K) @ np.stack([us, vs, np.ones_like(us)]).astype(np.float64)
    T = w2c_j @ np.linalg.inv(w2c_i)
    X_j = T[:3, :3] @ (rays * d) + T[:3, 3:4]
    z_j = X_j[2]
    front = z_j > 1e-6
    uv = (K @ (X_j[:, front] / z_j[front]))[:2]
    u_j, v_j = np.round(uv[0]).astype(int), np.round(uv[1]).astype(int)
    inb = (u_j >= 0) & (u_j < w) & (v_j >= 0) & (v_j < h)
    d_j = depth_j[v_j[inb], u_j[inb]]
    valid = d_j > 0
    if valid.sum() < 100:
        return np.inf, int(valid.sum())
    err = np.abs(np.log(z_j[front][inb][valid] / d_j[valid]))
    return float(np.median(err)), int(valid.sum())


def verify_convention(K, traj, scene_dir, frame_ids, depth_scale):
    """Test c2w vs w2c on two frame pairs from the window; return (convention, report, scale_mult).

    The convention decision is robust to a wrong depth scale, but the absolute
    error threshold alone is NOT scale-sensitive at small baselines (a 6.5x
    scale error only lifts the near-pair error to ~0.01). So after picking the
    convention, sweep a global depth multiplier: the consistency error is
    minimized at the true scale (the parallax term breaks the degeneracy), and
    the argmin must land at ~1x for the assumed cam_params scale to be correct.
    """
    pairs = [(frame_ids[0], frame_ids[min(2, len(frame_ids) - 1)]), (frame_ids[0], frame_ids[-1])]
    depths = {
        fid: load_depth(scene_dir / "results" / f"depth{fid:06d}.png", depth_scale)
        for fid in sorted({f for pair in pairs for f in pair})
    }
    report = {}
    for name, to_w2c in [("c2w", np.linalg.inv), ("w2c", lambda M: M)]:
        errs = [cross_depth_error(K, to_w2c(traj[i]), to_w2c(traj[j]), depths[i], depths[j])[0] for i, j in pairs]
        report[name] = float(np.mean(errs))
    best = min(report, key=report.get)
    other = max(report, key=report.get)
    if not (report[best] < 0.02 and report[other] > 5 * report[best]):
        raise RuntimeError(
            f"no clear traj convention winner (median |log depth err| c2w={report['c2w']:.4f} "
            f"w2c={report['w2c']:.4f}) — check depth scale / depth semantics / traj format"
        )

    to_w2c = np.linalg.inv if best == "c2w" else (lambda M: M)
    mults = np.exp(np.linspace(-2.3, 2.3, 47))  # 0.1 log-steps covering ~1/10x..10x
    sweep = [
        np.mean(
            [cross_depth_error(K, to_w2c(traj[i]), to_w2c(traj[j]), depths[i] * m, depths[j] * m)[0] for i, j in pairs]
        )
        for m in mults
    ]
    scale_mult = float(mults[int(np.argmin(sweep))])
    if abs(np.log(scale_mult)) > 0.05:
        raise RuntimeError(
            f"depth scale check failed: consistency error is minimized at {scale_mult:.3f}x the assumed "
            f"scale ({depth_scale}) — cam_params depth scale is likely wrong by that factor"
        )
    return best, report, scale_mult


def write_text_model(rec_dir, cam, images):
    """images: list of (image_id, name, w2c 4x4). Empty points3D."""
    with open(rec_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {cam['w']} {cam['h']} {cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']}\n")
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


def prepare_scene(args, scene, out_data, out_testsets):
    root = args.root
    scene_dir = root / scene
    rec_dir = out_data / scene / "rec"
    frame_ids = list(range(args.start, args.start + args.stride * args.num_frames, args.stride))

    cam, cam_path = find_cam_params(args, root, scene_dir)
    results_dir = scene_dir / "results"
    n_rgb = len(list(results_dir.glob("frame*.jpg")))
    n_depth = len(list(results_dir.glob("depth*.png")))
    w, h = Image.open(results_dir / "frame000000.jpg").size
    assert (w, h) == (cam["w"], cam["h"]), f"{scene}: cam_params {cam['w']}x{cam['h']} vs images {w}x{h}"

    traj = np.loadtxt(scene_dir / "traj.txt").reshape(-1, 4, 4)
    assert len(traj) == n_rgb == n_depth, f"{scene}: {len(traj)} poses vs {n_rgb} rgb vs {n_depth} depth"
    assert np.allclose(traj[:, 3], [0, 0, 0, 1]), f"{scene}: traj rows are not homogeneous 4x4"
    assert frame_ids[-1] < len(traj), f"{scene}: window end {frame_ids[-1]} >= {len(traj)} frames"

    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    convention, report, scale_mult = verify_convention(K, traj, scene_dir, frame_ids, cam["scale"])
    w2c_all = np.linalg.inv(traj) if convention == "c2w" else traj

    centers = np.linalg.inv(w2c_all[frame_ids])[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    print(
        f"{scene}: traj is {convention} (median |log depth err| c2w={report['c2w']:.4f} w2c={report['w2c']:.4f}, "
        f"scale argmin {scale_mult:.3f}x); "
        f"window step mean={steps.mean():.4f} max={steps.max():.4f} path={steps.sum():.4f}"
    )

    if rec_dir.exists() and not args.overwrite:
        print(f"{scene}: rec exists, skipping model write (testset yaml still written)")
    else:
        images = [(fidx + 1, f"frame{fidx:06d}.jpg", w2c_all[fidx]) for fidx in range(len(traj))]
        rec_dir.mkdir(parents=True, exist_ok=True)
        write_text_model(rec_dir, cam, images)
        import pycolmap  # deferred: only needed for the parse sanity check

        rec = pycolmap.Reconstruction(rec_dir)
        assert rec.num_images() == len(images), f"{scene}: wrote {len(images)} images, read {rec.num_images()}"

        images_link = out_data / scene / "images"
        if not images_link.exists():
            images_link.symlink_to(results_dir.resolve())

        with open(out_data / scene / "meta.json", "w") as f:
            json.dump(
                {
                    "scene": scene,
                    "cam_params_source": str(cam_path),
                    "intrinsics": {k: cam[k] for k in ["w", "h", "fx", "fy", "cx", "cy"]},
                    "depth_png_scale": cam["scale"],
                    "depth_png_dtype": "uint16",
                    "depth_invalid_value": 0,
                    "depth_semantics": "z-depth (validated by cross-frame consistency check)",
                    "traj_convention": convention,
                    "convention_check_median_log_err": report,
                    "depth_scale_check_argmin_multiplier": scale_mult,
                    "imid_offset": 1,
                },
                f,
                indent=2,
            )

    testset_dir = out_testsets / scene
    testset_dir.mkdir(parents=True, exist_ok=True)
    with open(testset_dir / f"{args.mode_name}.yaml", "w") as f:
        yaml.safe_dump({0: [fidx + 1 for fidx in frame_ids]}, f, default_flow_style=None)
    print(f"{scene}: {len(traj)} poses in rec, testset '{args.mode_name}' = frames {frame_ids[0]}:{args.stride}:{frame_ids[-1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Replica root (contains <scene>/results, traj.txt)")
    parser.add_argument("--out", type=Path, required=True, help="Output root (gets data/ and testsets/)")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--cam-params", type=Path, help="Explicit cam_params.json path")
    parser.add_argument("--start", type=int, default=CANONICAL_START)
    parser.add_argument("--stride", type=int, default=CANONICAL_STRIDE)
    parser.add_argument("--num-frames", type=int, default=CANONICAL_NUM)
    parser.add_argument("--mode-name", type=str, default="mini", help="Testset yaml name (= benchmark -m mode)")
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    failed = []
    for scene in args.scenes:
        try:
            prepare_scene(args, scene, args.out / "data", args.out / "testsets")
        except Exception as e:
            print(f"{scene}: FAILED — {type(e).__name__}: {e}")
            failed.append(scene)
    if failed:
        print(f"\n{len(failed)} scenes failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
