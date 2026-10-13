"""Geometry for downward-looking planar VO, independent of feature extraction."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlanarEstimate:
    transform: object
    status: str
    matches: int
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_displacement_px: float = 0.0
    reprojection_rmse_px: object = None


def estimate_affine(points_a, points_b, config):
    """Estimate previous-to-current full affine transform with RANSAC."""
    points_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    points_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    if points_a.shape != points_b.shape:
        raise ValueError('Correspondence arrays must have the same shape')
    valid = np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
    points_a, points_b = points_a[valid], points_b[valid]
    result = PlanarEstimate(None, 'insufficient_matches', len(points_a))
    if len(points_a) < config['min_matches']:
        return result
    affine, mask = cv2.estimateAffine2D(
        points_a, points_b, method=cv2.RANSAC,
        ransacReprojThreshold=config['ransac_threshold_px'],
        maxIters=2000, confidence=0.99, refineIters=10)
    if affine is None or mask is None or not np.isfinite(affine).all():
        result.status = 'affine_failed'
        return result
    inlier_mask = mask.ravel() != 0
    result.inliers = int(inlier_mask.sum())
    result.inlier_ratio = result.inliers / len(points_a)
    if result.inliers:
        a, b = points_a[inlier_mask], points_b[inlier_mask]
        result.median_displacement_px = float(np.median(np.linalg.norm(b - a, axis=1)))
        residuals = a.dot(affine[:, :2].T) + affine[:, 2] - b
        result.reprojection_rmse_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    if result.inliers < config['min_inliers'] or result.inlier_ratio < config['min_inlier_ratio']:
        result.status = 'insufficient_inliers'
        return result
    if np.linalg.det(affine[:, :2]) <= 1e-8 or np.linalg.cond(affine[:, :2]) > 100:
        result.status = 'degenerate_affine'
        return result
    result.transform = np.eye(3)
    result.transform[:2] = affine
    result.status = 'ok'
    return result


def rectification_matrix(K, normal):
    """Rotate a supplied camera-frame plane normal to +Z, preserving in-plane freedom.

    This requires an external, resolved normal; it does not disambiguate a
    homography decomposition or estimate absolute pitch/roll by itself.
    """
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        raise ValueError('ground_normal_camera must contain three finite nonzero components')
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    target = np.array([0., 0., 1.])
    axis = np.cross(normal, target)
    sine = np.linalg.norm(axis)
    rotation = np.eye(3) if sine < 1e-8 else cv2.Rodrigues(
        axis / sine * np.arctan2(sine, np.dot(normal, target)))[0]
    return K.dot(rotation).dot(np.linalg.inv(K))


def correct_points(points, K=None, distortion=None, rectification=None):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not len(points):
        return points.reshape(-1, 2)
    if K is not None and distortion is not None:
        points = cv2.undistortPoints(points, K, distortion, P=K)
    if rectification is not None:
        homogeneous = np.c_[points.reshape(-1, 2), np.ones(len(points))].dot(rectification.T)
        if np.any(homogeneous[:, 2] <= 1e-6):
            raise ValueError('Rectification sends matched points to or behind the horizon')
        points = (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.float32)
    return points.reshape(-1, 2)


def homography_diagnostics(points_a, points_b, threshold, K=None):
    """Return all decomposition candidates, without claiming an absolute attitude."""
    result = {'homography': None, 'inliers': 0, 'candidates': []}
    if len(points_a) < 4:
        return result
    H, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, threshold)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return result
    result.update(homography=H.tolist(), inliers=int(np.count_nonzero(mask)))
    if K is not None:
        _, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
        for rotation, translation, normal in zip(rotations, translations, normals):
            result['candidates'].append({'R': rotation.tolist(),
                                         'translation_over_plane_distance': translation.ravel().tolist(),
                                         'normal_in_previous_camera': normal.ravel().tolist()})
    return result


def align_and_evaluate(estimated, reference):
    """One global Sim(2) fit for evaluation only. Never feeds back into odometry."""
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    if estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] != 2:
        raise ValueError('Trajectories must have equal shape (N, 2)')
    if len(estimated) < 3 or not np.isfinite(estimated).all() or not np.isfinite(reference).all():
        return None, {'status': 'insufficient_reference'}
    # Image Y points down; East/North is a Cartesian plane viewed from above.
    cartesian = estimated * [1., -1.]
    x_mean, y_mean = cartesian.mean(axis=0), reference.mean(axis=0)
    x, y = cartesian - x_mean, reference - y_mean
    variance = np.sum(x ** 2)
    if variance < 1e-8 or np.sum(y ** 2) < 1e-8:
        return None, {'status': 'stationary_or_degenerate_trajectory'}
    U, singular, Vt = np.linalg.svd(x.T.dot(y))
    signs = np.ones(2)
    signs[-1] = np.sign(np.linalg.det(U.dot(Vt)))
    rotation = U.dot(np.diag(signs)).dot(Vt)
    scale = float(np.sum(singular * signs) / variance)
    if scale <= 1e-8:
        return None, {'status': 'degenerate_alignment'}
    translation = y_mean - scale * x_mean.dot(rotation)
    aligned = scale * cartesian.dot(rotation) + translation
    errors = np.linalg.norm(aligned - reference, axis=1)
    return aligned, {'status': 'ok', 'method': 'single_global_Sim2_fit_on_evaluation_samples',
                     'scale_m_per_px': scale, 'rotation_row_vectors': rotation.tolist(),
                     'image_y_sign_before_alignment': -1,
                     'translation_m': translation.tolist(),
                     'aligned_ate_rmse_m': float(np.sqrt(np.mean(errors ** 2))),
                     'aligned_endpoint_error_m': float(errors[-1]),
                     'warning': 'GPS fits scale/orientation for evaluation; this is not GPS-free metric VO'}
