"""Classical landmark-based face warping utilities.

The morph is produced in two stages (see README / model):

  1. Geometric alignment (this module): both faces are warped, via per-triangle
     affine transforms over a Delaunay triangulation, onto a common "mean" shape
     defined by `(1 - alpha) * landmarks_A + alpha * landmarks_B`. After this step
     the eyes/nose/mouth of A and B sit at the *same* pixel coordinates, so a
     straight cross-dissolve no longer ghosts.
  2. Photometric blend + neural refinement (model/diffusion_model.py).

All functions operate on float images in [0, 1], HxWx3, numpy.
"""
import cv2
import numpy as np


def add_boundary_points(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Append the 8 image-border anchor points so the triangulation (and therefore
    the warp) covers the whole frame, not just the convex hull of the face."""
    boundary = np.array([
        [0, 0], [width // 2, 0], [width - 1, 0],
        [width - 1, height // 2], [width - 1, height - 1],
        [width // 2, height - 1], [0, height - 1], [0, height // 2],
    ], dtype=np.float32)
    return np.vstack([landmarks.astype(np.float32), boundary])


def calculate_delaunay_triangles(width: int, height: int, points: np.ndarray):
    """Return Delaunay triangulation as a list of index triples into `points`."""
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangles = []
    for t in subdiv.getTriangleList():
        verts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for vx, vy in verts:
            # match each triangle vertex back to its point index
            d = np.hypot(points[:, 0] - vx, points[:, 1] - vy)
            j = int(np.argmin(d))
            if d[j] < 1.0:
                idx.append(j)
        if len(idx) == 3:
            triangles.append(tuple(idx))
    return triangles


def _warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    t1 = [(p[0] - r1[0], p[1] - r1[1]) for p in t_src]
    t2 = [(p[0] - r2[0], p[1] - r2[1]) for p in t_dst]

    src_crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_crop.size == 0 or r2[2] == 0 or r2[3] == 0:
        return

    M = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
    warped = cv2.warpAffine(src_crop, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2), (1.0, 1.0, 1.0), cv2.LINE_AA, 0)

    region = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = region * (1 - mask) + warped * mask


def warp_to_shape(img, src_landmarks, dst_landmarks, triangles, width, height):
    """Warp `img` so its `src_landmarks` move onto `dst_landmarks`."""
    out = np.zeros((height, width, 3), dtype=np.float32)
    for a, b, c in triangles:
        _warp_triangle(img,
                       out,
                       [src_landmarks[a], src_landmarks[b], src_landmarks[c]],
                       [dst_landmarks[a], dst_landmarks[b], dst_landmarks[c]])
    return out


def warp_to_landmarks(img, src_lm, dst_lm, width, height):
    """Warp a full-frame image from `src_lm` geometry onto `dst_lm` geometry.

    Convenience wrapper that adds the border anchors and builds the triangulation,
    e.g. to move a morph aligned to the mean shape into one identity's posture.
    """
    p_src = add_boundary_points(src_lm[:68], width, height)
    p_dst = add_boundary_points(dst_lm[:68], width, height)
    triangles = calculate_delaunay_triangles(width, height, p_dst)
    return warp_to_shape(img, p_src, p_dst, triangles, width, height)


def hull_mask(landmarks_68, width, height, feather=11):
    """Feathered convex-hull mask of the face (1 inside the face, ~0 outside).

    Uses the 68 facial landmarks (not the border anchors) so the mask covers the
    whole face region - cheeks/forehead included - rather than a sparse set of
    points. Returns HxWx1 float in [0, 1]."""
    mask = np.zeros((height, width), dtype=np.float32)
    hull = cv2.convexHull(np.int32(landmarks_68[:68]))
    cv2.fillConvexPoly(mask, hull, 1.0)
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask[..., None]


def match_color(src, ref, mask):
    """Recolor `src` so its tone matches `ref` inside the face `mask`.

    Reduces the skin-tone cast / desaturation that a cross-dissolve of two
    differently-lit faces produces: by aligning B's per-channel mean and std to
    A's (computed over the hull region) in CIE-Lab, the subsequent blend averages
    two faces that already share tone instead of creating a two-tone face.

    src, ref : HxWx3 float [0, 1].  mask : HxWx1 float in [0, 1].
    Returns the recolored `src` in [0, 1].
    """
    s = cv2.cvtColor((np.clip(src, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor((np.clip(ref, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask[..., 0] > 0.5
    if m.sum() < 16:                       # too small a region to estimate stats
        return src
    for c in range(3):
        s_mu, s_sd = s[m, c].mean(), s[m, c].std() + 1e-5
        r_mu, r_sd = r[m, c].mean(), r[m, c].std() + 1e-5
        s[..., c] = (s[..., c] - s_mu) / s_sd * r_sd + r_mu
    s = np.clip(s, 0, 255).astype(np.uint8)
    return cv2.cvtColor(s, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def poisson_seam_blend(morph, background, landmarks_68, width, height):
    """Gradient-domain (Poisson) composite of the morphed face onto `background`.

    The feathered alpha composite used in the model leaves a faint hull-boundary
    seam (a brightness/colour step where the blended face meets the single-identity
    surround). `cv2.seamlessClone` re-integrates the face over the background in
    the gradient domain, so the boundary disappears while interior detail is kept.

    morph, background : HxWx3 float [0, 1].  Returns HxWx3 float [0, 1].
    """
    hull = cv2.convexHull(np.int32(landmarks_68[:68]))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # erode slightly so seamlessClone has clean background gradients to anchor to
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    # seamlessClone requires the masked region to sit strictly inside the frame;
    # clear a 1px border so a face that fills the frame can't push the ROI out.
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0

    ys, xs = np.where(mask > 0)
    if xs.size == 0:                       # degenerate hull -> nothing to blend
        return np.clip(morph, 0, 1).astype(np.float32)
    # centre on the mask's own bounding box so the cloned patch stays in place
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))

    src = (np.clip(morph, 0, 1) * 255).astype(np.uint8)
    dst = (np.clip(background, 0, 1) * 255).astype(np.uint8)
    blended = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    return blended.astype(np.float32) / 255.0


def shape_distance(lm_a: np.ndarray, lm_b: np.ndarray) -> float:
    """Procrustes-style face-shape score between two 68x2 landmark shapes.

    Each shape is centered and scale-normalized (removing position and face size),
    then we take the mean per-point Euclidean distance. Large values mean the two
    faces differ a lot in posture/shape - the case where morphing to a common mean
    shape deforms. Callers use it to decide whether a pair is blendable.
    """
    def _normalize(p):
        p = p[:68].astype(np.float32)
        p = p - p.mean(axis=0)
        scale = np.sqrt((p ** 2).sum() / len(p))
        return p / (scale + 1e-8)
    a, b = _normalize(lm_a), _normalize(lm_b)
    return float(np.linalg.norm(a - b, axis=1).mean())


# Jaw landmarks (ibug/dlib 0-16) trace the outer face contour. Max pixels (at
# model resolution) the blended contour may sit from BOTH source jawlines before
# we snap it back toward the nearer real contour - prevents an unnatural "average"
# jaw that matches neither face. Set to None to disable clamping.
JAW_INDICES = list(range(17))
CONTOUR_MAX_SHIFT = 10.0


def clamp_contour(mean_lm, lm_a, lm_b, max_shift=CONTOUR_MAX_SHIFT):
    """Pull blended jaw points back when they stray too far from both faces.

    A blended contour point lies between A's and B's jaw. If it ends up farther
    than `max_shift` px from *both* source jawlines (which happens when the two
    jaws are far apart), snap it to within `max_shift` of whichever real contour
    is nearer, so the morph's outline stays a plausible face shape.
    """
    if max_shift is None:
        return mean_lm
    out = mean_lm.astype(np.float32).copy()
    for i in JAW_INDICES:
        da, db = out[i] - lm_a[i], out[i] - lm_b[i]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if min(na, nb) > max_shift:
            if na <= nb:
                out[i] = lm_a[i] + da / (na + 1e-8) * max_shift
            else:
                out[i] = lm_b[i] + db / (nb + 1e-8) * max_shift
    return out


def morph_pair(img_a, img_b, lm_a, lm_b, alpha, width, height):
    """Align A and B onto the mean shape and return the pieces the model needs.

    Returns
    -------
    warped_a, warped_b : HxWx3 float32 in [0, 1], aligned to the mean shape.
    mean_landmarks_68  : the 68 mean facial landmarks (for heatmaps).
    mask               : HxWx1 feathered hull mask at the mean shape.
    """
    lm_a = lm_a.astype(np.float32)
    lm_b = lm_b.astype(np.float32)
    mean_lm = (1.0 - alpha) * lm_a + alpha * lm_b

    # contour correction: keep the morphed jawline from drifting too far from both
    # source faces (avoids a deformed "average" outline that matches neither).
    mean_lm = clamp_contour(mean_lm, lm_a, lm_b)

    # border anchors keep the warp defined over the full frame
    pa = add_boundary_points(lm_a, width, height)
    pb = add_boundary_points(lm_b, width, height)
    pm = add_boundary_points(mean_lm, width, height)

    triangles = calculate_delaunay_triangles(width, height, pm)

    warped_a = warp_to_shape(img_a, pa, pm, triangles, width, height)
    warped_b = warp_to_shape(img_b, pb, pm, triangles, width, height)

    mask = hull_mask(mean_lm, width, height)

    # photometric harmonization: align B's tone to A inside the face so the
    # cross-dissolve doesn't bake in a skin-tone/contrast seam.
    warped_b = match_color(warped_b, warped_a, mask)
    return warped_a, warped_b, mean_lm, mask
