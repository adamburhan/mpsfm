"""K=2 anchored patch-GMM over log depth at keypoints, for the max-mixture
depth factor. Ported from depth-aware-BA's gmm_patch extractor, adapted to
mpsfm: fixed patch radius (SuperPoint keypoints carry no scale), candidate
gating via the depth continuity mask, and per-mode sigmas fused with the
calibrated per-pixel uncertainties.
"""

import numpy as np

_TINY = 1e-12


def _disk(r):
    """Integer (dv, du) offsets of a filled disk of radius r."""
    d = np.arange(-r, r + 1)
    dv, du = np.meshgrid(d, d, indexing="ij")
    m = dv * dv + du * du <= r * r
    return dv[m], du[m]


def _weighted_em(y, sw, mu0, sig_floor, wmin, sep_min, max_iter):
    """Anchored, spatially-weighted 2-comp 1D EM in LOG depth. mu0 fixed.

    Returns (mu1, s0, s1, w0, w1, r1) in log space, where r1 are the final
    mode-1 responsibilities (for calibrated-uncertainty fusion). Collapses to
    unimodal (mu1 = mu0, w1 = wmin) when the patch is flat / undersampled, or
    when the fitted second mode is unsupported or too close to the anchor.
    """
    if y.size < 2:
        return mu0, sig_floor, sig_floor, 1.0 - wmin, wmin, np.zeros_like(y)
    W = max(sw.sum(), _TINY)
    ybar = (sw * y).sum() / W
    yvar = (sw * (y - ybar) ** 2).sum() / W
    s_init = max(np.sqrt(yvar), sig_floor)
    if yvar < sig_floor * sig_floor:  # flat patch -> unimodal
        return mu0, s_init, s_init, 1.0 - wmin, wmin, np.zeros_like(y)

    mu1 = y[np.argmax(np.abs(y - mu0))]  # farthest sample seeds mode 1
    s0 = s1 = s_init
    p0 = p1 = 0.5
    r1 = np.zeros_like(y)
    for _ in range(max_iter):
        g0 = p0 * np.exp(-0.5 * ((y - mu0) / s0) ** 2) / s0
        g1 = p1 * np.exp(-0.5 * ((y - mu1) / s1) ** 2) / s1
        den = g0 + g1 + _TINY
        r0, r1 = g0 / den, g1 / den
        n0, n1 = (sw * r0).sum(), (sw * r1).sum()
        p0, p1 = n0 / W, n1 / W
        mu1 = (sw * r1 * y).sum() / max(n1, _TINY)  # mu0 stays fixed
        s0 = max(np.sqrt((sw * r0 * (y - mu0) ** 2).sum() / max(n0, _TINY)), sig_floor)
        s1 = max(np.sqrt((sw * r1 * (y - mu1) ** 2).sum() / max(n1, _TINY)), sig_floor)

    # joint gate: keep the second mode only if supported AND separated
    if p1 < wmin or abs(mu1 - mu0) < sep_min:
        return mu0, s0, s0, 1.0 - wmin, wmin, np.zeros_like(y)
    return mu1, s0, s1, p0, p1, r1


def fit_mixture(
    depth_map,
    uncertainty_map,
    valid_map,
    kps_scaled,
    anchor_depths,
    candidates,
    radius=6,
    wmin=0.05,
    sig_floor=0.05,
    sep_min=0.1,
    em_iters=30,
    uniform_weights=True,
):
    """Fit K=2 log-depth mixtures at keypoints.

    Args:
        depth_map: (H, W) linear depth (the prior, unscaled).
        uncertainty_map: (H, W) calibrated linear-depth variance.
        valid_map: (H, W) bool.
        kps_scaled: (N, 2) keypoints in map pixel coordinates.
        anchor_depths: (N,) committed depth at the keypoints (bilinear prior);
            becomes mode 0.
        candidates: (N,) bool; non-candidates get a degenerate K=1-style stack.

    Returns (modes, weights, sigmas), each (N, 2):
        modes in absolute linear depth, sigmas in log space fusing the EM
        spread with the responsibility-weighted calibrated log-variance.
    """
    h, w = depth_map.shape
    n = len(kps_scaled)
    u_kp = np.clip(np.floor(kps_scaled[:, 0]).astype(np.int64), 0, w - 1)
    v_kp = np.clip(np.floor(kps_scaled[:, 1]).astype(np.int64), 0, h - 1)

    anchor = np.maximum(anchor_depths.astype(np.float64), _TINY)
    # calibrated log-space variance at the keypoints (unimodal fallback sigma)
    cal_var_kp = uncertainty_map[v_kp, u_kp] / anchor**2

    modes = np.repeat(anchor[:, None], 2, axis=1)
    weights = np.repeat(np.array([[1.0 - wmin, wmin]]), n, axis=0)
    sigmas = np.repeat(np.sqrt(cal_var_kp.clip(1e-12, None))[:, None], 2, axis=1)

    dv, du = _disk(radius)
    sigma_s = max(radius / 2.0, _TINY)  # spatial weight bandwidth
    for i in np.flatnonzero(candidates):
        vv, uu = v_kp[i] + dv, u_kp[i] + du
        inb = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
        vvi, uui, dvi, dui = vv[inb], uu[inb], dv[inb], du[inb]
        ok = valid_map[vvi, uui] & (depth_map[vvi, uui] > 0)
        if ok.sum() < 2:
            continue
        vvi, uui, dvi, dui = vvi[ok], uui[ok], dvi[ok], dui[ok]
        dpatch = depth_map[vvi, uui].astype(np.float64)

        mu0 = np.log(anchor[i])
        sw = np.exp(-(dvi * dvi + dui * dui) / (2.0 * sigma_s * sigma_s))
        y = np.log(dpatch)
        mu1, s0, s1, p0, p1, r1 = _weighted_em(y, sw, mu0, sig_floor, wmin, sep_min, em_iters)

        # responsibility-weighted calibrated log-variance per mode
        cal_lv = uncertainty_map[vvi, uui] / dpatch**2
        r0 = 1.0 - r1
        cal0 = (sw * r0 * cal_lv).sum() / max((sw * r0).sum(), _TINY)
        cal1 = (sw * r1 * cal_lv).sum() / max((sw * r1).sum(), _TINY) if r1.sum() > 0 else cal0

        modes[i] = (anchor[i], np.exp(mu1))
        sigmas[i] = (np.sqrt(s0**2 + cal0), np.sqrt(s1**2 + cal1))
        if not uniform_weights:
            weights[i] = (p0, p1)
        elif mu1 != mu0:
            # EM weights are detection-only: with uniform weights, mode
            # selection is left to the multiview residual rather than patch
            # statistics (which mislead at discontinuities, where kps live)
            weights[i] = (0.5, 0.5)

    return modes.astype(np.float64), weights.astype(np.float64), sigmas.astype(np.float64)
