"""GT point-geometry analysis for the Replica fixed-sigma battery.

Reads the BA diag dumps (MPSFM_DIAG_DIR reruns), the saved sparse models
(--save_sparse) and Replica GT to measure what pose AUC cannot: per-track
depth error against the GT surface the track actually represents.

Per observation (final-BA row of each registered image):
  z_est   depth of the final 3D point in the image's GT camera (rec Sim3-
          aligned to GT poses via camera centers)
  z_trk   depth of the track's GT-pose triangulation in the same camera
          ("track-key GT": the physical surface the 2D observations agree on,
          independent of any arm's BA; gated by GT reprojection error)
  z_pix   bilinear GT depth at the detector pixel ("pixel-key GT")

Subsets: unimodal rows, ambiguous rows (fitted second mode), structural rows
(|log z_trk - log z_pix| > thresh: track surface != detector-center surface).
For maxmix arms: winning-mode recomputation (score = whitened^2 + 2 ln(s/w))
and FG/BG selection accuracy on GT-clear ambiguous rows. With --pair, rows
are matched across two arms by (window, image, keypoint) for paired deltas.

Run inside the container:
  python scripts/analyze_geometry_replica.py -s room0 office0 office3 \
      -c repr-sp-lg_m3dv2-uni-cauchy003-noint repr-sp-lg_m3dv2-maxmix-cauchy003-noint \
         repr-sp-lg_m3dv2-gt-uni-cauchy003-noint repr-sp-lg_m3dv2-gt-maxmix-cauchy003-noint \
      --pair repr-sp-lg_m3dv2-uni-cauchy003-noint:repr-sp-lg_m3dv2-maxmix-cauchy003-noint \
      --pair repr-sp-lg_m3dv2-gt-uni-cauchy003-noint:repr-sp-lg_m3dv2-gt-maxmix-cauchy003-noint
"""

import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpsfm.data_proc.replica import ReplicaDataset, ReplicaParser
from mpsfm.vars import lvars

FIXED_SIGMA = 0.03
GT_REPROJ_GATE = 3.0  # px; GT-pose triangulation must explain the track
STRUCTURAL_THRESH = 0.05  # |log z_trk - log z_pix| above this = structural row
GT_CLEAR_THRESH = 0.05  # GT within this log distance of a mode = clear
MIN_SEP = 0.05  # min mode separation for selection-accuracy rows


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cs, cd = src - mu_s, dst - mu_d
    cov = cd.T @ cs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, 1.0, d])
    s = np.trace(np.diag(S) @ D) / (cs**2).sum() * len(src)
    R = U @ D @ Vt
    t = mu_d - s * R @ mu_s
    return s, R, t


def w2c_matrix(cam_from_world):
    m = np.eye(4)
    m[:3, :3] = cam_from_world.rotation.matrix()
    m[:3, 3] = cam_from_world.translation
    return m


def cam_center(cam_from_world):
    R, t = cam_from_world.rotation.matrix(), cam_from_world.translation
    return -R.T @ t


def triangulate_gt(obs, projs):
    """DLT over (xy, P) observations; returns (X, mean reproj err in px)."""
    A = []
    for (x, y), P in zip(obs, projs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    X = np.linalg.svd(np.asarray(A))[2][-1]
    X = X[:3] / X[3]
    errs = []
    for (x, y), P in zip(obs, projs):
        p = P @ np.append(X, 1.0)
        if p[2] <= 0:
            return X, np.inf
        errs.append(np.hypot(p[0] / p[2] - x, p[1] / p[2] - y))
    return X, float(np.mean(errs))


def bilinear(img, x, y):
    h, w = img.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
        return -1.0
    patch = img[y0 : y0 + 2, x0 : x0 + 2]
    if (patch <= 0).any():
        return -1.0
    fx, fy = x - x0, y - y0
    return float(
        patch[0, 0] * (1 - fx) * (1 - fy)
        + patch[0, 1] * fx * (1 - fy)
        + patch[1, 0] * (1 - fx) * fy
        + patch[1, 1] * fx * fy
    )


def load_final_dumps(diag_dir, window_imids):
    """Last global-BA dump per imid, assigned to its window."""
    by_key = {}
    for f in sorted(diag_dir.glob("*_ba_*.npz")):
        seq = int(f.name.split("_")[0])
        imid = int(f.stem.split("_ba_")[1])
        tids = [tid for tid, ims in window_imids.items() if imid in ims]
        assert len(tids) == 1, f"imid {imid} in windows {tids}; windows must be disjoint"
        d = dict(np.load(f, allow_pickle=True))
        if str(d["ba_mode"]) != "global":
            continue
        key = (tids[0], imid)
        if key not in by_key or seq > by_key[key][0]:
            by_key[key] = (seq, d)
    return {k: v[1] for k, v in by_key.items()}


def analyze_conf(conf, scenes, mode, diag_root, exp_dir):
    import pycolmap

    rows = defaultdict(list)  # column name -> values, one entry per observation
    for scene in scenes:
        testset_dir = ReplicaDataset.testsets / scene
        with open(testset_dir / f"{mode}.yaml") as f:
            window_imids = {tid: set(v) for tid, v in yaml.safe_load(f).items()}
        parser = ReplicaParser(scene)
        dumps = load_final_dumps(diag_root / conf / scene, window_imids)

        for tid in sorted(window_imids):
            sparse = exp_dir / "reconstruction" / mode / scene / str(tid) / conf / "sparse" / "0"
            if not sparse.exists():
                print(f"WARNING: no sparse model {sparse}, skipping")
                continue
            rec = pycolmap.Reconstruction(sparse)
            reg = [imid for imid in rec.reg_image_ids() if (tid, imid) in dumps]
            if len(reg) < 3:
                continue

            # Sim3 rec -> GT via camera centers
            src = np.array([cam_center(rec.images[i].cam_from_world) for i in reg])
            dst = np.array([cam_center(parser.pose(i)) for i in reg])
            s_al, R_al, t_al = umeyama(src, dst)

            K = parser.camera(reg[0]).calibration_matrix()
            gt_w2c = {i: w2c_matrix(parser.pose(i)) for i in window_imids[tid]}
            gt_proj = {i: K @ gt_w2c[i][:3] for i in gt_w2c}
            gt_depth_maps = {}

            # GT-pose triangulation per track, cached across images
            tri_cache = {}

            def track_gt(p3d_id):
                if p3d_id in tri_cache:
                    return tri_cache[p3d_id]
                track = rec.points3D[p3d_id].track
                obs, projs = [], []
                for el in track.elements:
                    if el.image_id not in gt_proj:
                        continue
                    obs.append(rec.images[el.image_id].points2D[el.point2D_idx].xy)
                    projs.append(gt_proj[el.image_id])
                res = triangulate_gt(obs, projs) if len(obs) >= 2 else (None, np.inf)
                tri_cache[p3d_id] = res
                return res

            for imid in reg:
                d = dumps[(tid, imid)]
                p3Ds, kps, in_ba = d["p3Ds"], d["kps"], d["in_ba"]
                modes = d.get("modes")
                weights = d.get("mode_weights")
                if imid not in gt_depth_maps:
                    gt_depth_maps[imid] = parser.gt_depth(imid)
                gt_map = gt_depth_maps[imid]
                w2c = gt_w2c[imid]
                rec_pose = rec.images[imid].cam_from_world

                for r in range(len(p3Ds)):
                    pid = int(p3Ds[r])
                    if pid not in rec.points3D:
                        continue
                    X = rec.points3D[pid].xyz
                    z_rec = (rec_pose.rotation.matrix() @ X + rec_pose.translation)[2]
                    if z_rec <= 0:
                        continue
                    X_tri, reproj = track_gt(pid)
                    if reproj > GT_REPROJ_GATE:
                        continue
                    X_gt = s_al * (R_al @ X) + t_al
                    z_est = (w2c[:3, :3] @ X_gt + w2c[:3, 3])[2]
                    z_trk = (w2c[:3, :3] @ X_tri + w2c[:3, 3])[2]
                    z_pix = bilinear(gt_map, kps[r, 0], kps[r, 1])
                    if z_est <= 0 or z_trk <= 0:
                        continue

                    multi, sel_ok, alt_correct, sep = False, np.nan, np.nan, 0.0
                    if modes is not None:
                        m = modes[r]
                        sep = abs(np.log(max(m[1], 1e-9)) - np.log(max(m[0], 1e-9)))
                        multi = sep > 1e-9
                        if multi:
                            # selection in rec units, exactly the factor's rule
                            whit2 = ((np.log(z_rec) - np.log(m.clip(1e-9))) / FIXED_SIGMA) ** 2
                            score = whit2 + 2 * np.log(FIXED_SIGMA / weights[r].clip(1e-9))
                            winner = int(np.argmin(score))
                            # GT-correct mode in aligned units
                            dist_gt = np.abs(np.log(m.clip(1e-9) * s_al) - np.log(z_trk))
                            h_gt = int(np.argmin(dist_gt))
                            if sep >= MIN_SEP and dist_gt[h_gt] < GT_CLEAR_THRESH:
                                sel_ok = float(winner == h_gt)
                                alt_correct = float(h_gt == 1)

                    rows["key"].append((tid, imid, round(kps[r, 0], 1), round(kps[r, 1], 1)))
                    rows["scene"].append(scene)
                    rows["e_trk"].append(abs(np.log(z_est) - np.log(z_trk)))
                    rows["e_pix"].append(-1.0 if z_pix <= 0 else abs(np.log(z_est) - np.log(z_pix)))
                    rows["structural"].append(z_pix > 0 and abs(np.log(z_trk) - np.log(z_pix)) > STRUCTURAL_THRESH)
                    rows["multi"].append(multi)
                    rows["in_ba"].append(bool(in_ba[r]))
                    rows["sel_ok"].append(sel_ok)
                    rows["alt_correct"].append(alt_correct)
    return {k: (np.array(v, dtype=object) if k == "key" else np.asarray(v)) for k, v in rows.items()}


def subset_masks(data):
    return {
        "all": np.ones(len(data["e_trk"]), bool),
        "unimodal": ~data["multi"],
        "ambiguous": data["multi"],
        "structural": data["structural"] & data["multi"],
    }


def report(conf, data):
    print(f"\n### {conf}  (n={len(data['e_trk'])} obs, {data['in_ba'].mean()*100:.1f}% in BA)")
    for name, m in subset_masks(data).items():
        if m.sum() == 0:
            print(f"  {name:<11} n=0")
            continue
        e = data["e_trk"][m]
        print(
            f"  {name:<11} n={m.sum():<7} median|log dX-dGT,trk|={np.median(e):.4f}  "
            f"mean={e.mean():.4f}  p90={np.quantile(e, 0.9):.4f}  >0.05: {(e > 0.05).mean()*100:.1f}%"
        )
    sel = data["sel_ok"][~np.isnan(data["sel_ok"])]
    alt = data["alt_correct"][~np.isnan(data["alt_correct"])]
    if len(sel):
        print(
            f"  mode selection (GT-clear ambiguous, n={len(sel)}): "
            f"correct {sel.mean()*100:.1f}% | GT-correct mode is ALT {alt.mean()*100:.1f}%"
        )


def report_pair(conf_a, conf_b, data_a, data_b):
    keys_a = {k: i for i, k in enumerate(data_a["key"])}
    ia, ib = [], []
    for j, k in enumerate(data_b["key"]):
        if k in keys_a:
            ia.append(keys_a[k])
            ib.append(j)
    ia, ib = np.asarray(ia), np.asarray(ib)
    print(f"\n=== paired: {conf_b} vs {conf_a}  ({len(ia)} matched obs) ===")
    if len(ia) == 0:
        return
    delta = data_b["e_trk"][ib] - data_a["e_trk"][ia]
    masks = subset_masks(data_b)
    for name in ["all", "unimodal", "ambiguous", "structural"]:
        m = masks[name][ib]
        if m.sum() == 0:
            print(f"  {name:<11} n=0")
            continue
        dd = delta[m]
        print(
            f"  {name:<11} n={m.sum():<7} improved={(dd < 0).mean()*100:5.1f}%  "
            f"median d(err)={np.median(dd):+.4f}  mean={dd.mean():+.4f}"
        )


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-c", "--confs", nargs="+", required=True)
    ap.add_argument("-s", "--scenes", nargs="+", required=True)
    ap.add_argument("-m", "--mode", default="mini")
    ap.add_argument("--diag-root", type=Path, default=Path("local/benchmarks/replica/diag"))
    ap.add_argument("--pair", action="append", default=[], help="uni_conf:maxmix_conf (repeatable)")
    args = ap.parse_args()

    results = {}
    for conf in args.confs:
        results[conf] = analyze_conf(conf, args.scenes, args.mode, args.diag_root, lvars.REPLICA_EXP_DIR)
        report(conf, results[conf])
    for pair in args.pair:
        a, b = pair.split(":")
        report_pair(a, b, results[a], results[b])
