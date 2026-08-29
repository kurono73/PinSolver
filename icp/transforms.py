import math
from typing import Tuple

import numpy as np

try:
    from mathutils import Matrix
except ImportError:
    Matrix = None


def numpy_to_blender_matrix(matrix: np.ndarray) -> Matrix:
    if Matrix is None:
        raise RuntimeError("mathutils is required for Blender Matrix conversion")
    return Matrix([[float(matrix[r, c]) for c in range(4)] for r in range(4)])


def blender_to_numpy_matrix(matrix: Matrix) -> np.ndarray:
    return np.array([[float(matrix[r][c]) for c in range(4)] for r in range(4)], dtype=np.float64)


def validate_rigid_transform(matrix: np.ndarray, atol: float = 1e-4) -> Tuple[bool, str]:
    if matrix.shape != (4, 4):
        return False, "Transform must be 4x4"
    if not np.isfinite(matrix).all():
        return False, "Transform contains non-finite values"
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        return False, "Invalid homogeneous transform"
    rot = matrix[:3, :3]
    if not np.allclose(rot.T @ rot, np.identity(3), atol=atol * 10.0):
        return False, "Transform rotation is not rigid"
    det = float(np.linalg.det(rot))
    if abs(det - 1.0) > atol * 10.0:
        return False, "Transform determinant is not 1"
    return True, ""


def rigid_transform_from_points(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[bool, np.ndarray, str]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    identity = np.identity(4, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        return False, identity, "Alignment pins must be Nx3 point arrays"
    if len(source) != len(target):
        return False, identity, "Source and Target alignment pin counts must match"
    if len(source) < 3:
        return False, identity, "Pin Initial Alignment requires 3+ valid pairs"
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return False, identity, "Alignment pins contain non-finite values"

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    cov = src_centered.T @ tgt_centered
    try:
        u, _, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return False, identity, "Pin Initial Alignment SVD failed"

    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = tgt_centroid - rot @ src_centroid

    matrix = identity.copy()
    matrix[:3, :3] = rot
    matrix[:3, 3] = trans
    valid, message = validate_rigid_transform(matrix)
    if not valid:
        return False, identity, message
    return True, matrix, ""


def similarity_transform_from_points(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[bool, np.ndarray, str]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    identity = np.identity(4, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        return False, identity, "Alignment pins must be Nx3 point arrays"
    if len(source) != len(target):
        return False, identity, "Source and Target alignment pin counts must match"
    if len(source) < 3:
        return False, identity, "Scale Correction requires 3+ valid pairs"
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return False, identity, "Alignment pins contain non-finite values"

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= 1e-12:
        return False, identity, "Source alignment pins are degenerate"

    cov = (tgt_centered.T @ src_centered) / float(len(source))
    try:
        u, singular_values, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return False, identity, "Scale Correction SVD failed"

    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / src_var)
    if not np.isfinite(scale) or scale <= 1e-12:
        return False, identity, "Scale Correction produced an invalid scale"
    trans = tgt_centroid - scale * (rot @ src_centroid)

    matrix = identity.copy()
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = trans
    if not np.isfinite(matrix).all():
        return False, identity, "Scale Correction produced non-finite transform"
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
        return False, identity, "Invalid homogeneous transform"
    return True, matrix, ""


def compose_source_world_matrix(source_world: Matrix, source_to_target_world: np.ndarray) -> Matrix:
    return numpy_to_blender_matrix(source_to_target_world) @ source_world


def transform_delta_metrics(matrix: np.ndarray) -> Tuple[float, float]:
    translation = float(np.linalg.norm(matrix[:3, 3]))
    rot = np.array(matrix[:3, :3], dtype=np.float64)
    det = float(np.linalg.det(rot))
    if np.isfinite(det) and abs(det) > 1e-12:
        scale = abs(det) ** (1.0 / 3.0)
        rot = rot / scale
    trace = float(np.trace(rot))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    rotation = math.degrees(math.acos(cos_angle))
    return translation, rotation


def matrix_fingerprint(matrix: Matrix) -> Tuple[float, ...]:
    return tuple(round(float(matrix[r][c]), 9) for r in range(4) for c in range(4))
