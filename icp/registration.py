from typing import Any, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .cancel import IcpCancelled, raise_if_cancelled
from .models import IcpResult, IcpSettings
from .transforms import validate_rigid_transform


class _NearestNeighborIndex:
    def __init__(self, target: np.ndarray, use_fast: bool):
        self.target = np.asarray(target, dtype=np.float64)
        self.flann = None
        if use_fast and cv2 is not None and hasattr(cv2, "flann_Index") and len(self.target) >= 1500:
            try:
                self.flann = cv2.flann_Index(np.asarray(self.target, dtype=np.float32), {"algorithm": 1, "trees": 6})
            except Exception:
                self.flann = None

    def query(self, source: np.ndarray, chunk_size: int, cancel_event: Optional[Any]) -> Tuple[np.ndarray, np.ndarray]:
        source = np.asarray(source, dtype=np.float64)
        if self.flann is None:
            return _nearest_neighbors_exact(source, self.target, chunk_size, cancel_event)
        indices = np.empty((len(source),), dtype=np.int64)
        distances = np.empty((len(source),), dtype=np.float64)
        query_chunk = max(128, min(2048, int(chunk_size) * 4))
        try:
            for start in range(0, len(source), query_chunk):
                raise_if_cancelled(cancel_event)
                stop = min(start + query_chunk, len(source))
                idx, dist_sq = self.flann.knnSearch(np.asarray(source[start:stop], dtype=np.float32), 1, params={"checks": 128})
                indices[start:stop] = idx[:, 0]
                distances[start:stop] = np.sqrt(np.maximum(0.0, dist_sq[:, 0].astype(np.float64)))
            return indices, distances
        except IcpCancelled:
            raise
        except Exception:
            self.flann = None
            return _nearest_neighbors_exact(source, self.target, chunk_size, cancel_event)


def _subset_points(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    indices = np.linspace(0, len(points) - 1, max(3, int(count))).astype(np.int64)
    return points[indices]


def _subset_pair(points: np.ndarray, normals: Optional[np.ndarray], count: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) <= count:
        return points, normals
    indices = np.linspace(0, len(points) - 1, max(3, int(count))).astype(np.int64)
    stage_points = points[indices]
    stage_normals = normals[indices] if normals is not None else None
    return stage_points, stage_normals


def _stage_plan(source_count: int, target_count: int, iterations: int, levels: int) -> Tuple[Tuple[int, int, int], ...]:
    level_count = max(1, min(3, int(levels)))
    iterations = max(1, int(iterations))
    if level_count == 1:
        return ((source_count, target_count, iterations),)
    if level_count == 2:
        first = max(3, iterations // 2)
        return (
            (min(source_count, 1200), min(target_count, 2500), first),
            (source_count, target_count, max(1, iterations - first)),
        )
    coarse = max(3, iterations // 4)
    mid = max(3, iterations // 3)
    fine = max(1, iterations - coarse - mid)
    return (
        (min(source_count, 900), min(target_count, 1800), coarse),
        (min(source_count, 2500), min(target_count, 5000), mid),
        (source_count, target_count, fine),
    )


def _nearest_neighbors_exact(source: np.ndarray, target: np.ndarray, chunk_size: int = 128, cancel_event: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.empty((len(source),), dtype=np.int64)
    distances = np.empty((len(source),), dtype=np.float64)
    target_sq = np.sum(target * target, axis=1)
    for start in range(0, len(source), chunk_size):
        raise_if_cancelled(cancel_event)
        stop = min(start + chunk_size, len(source))
        chunk = source[start:stop]
        dist_sq = np.sum(chunk * chunk, axis=1)[:, None] + target_sq[None, :] - 2.0 * (chunk @ target.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        idx = np.argmin(dist_sq, axis=1)
        indices[start:stop] = idx
        distances[start:stop] = np.sqrt(dist_sq[np.arange(stop - start), idx])
    return indices, distances


def _nearest_neighbors_chunked(source: np.ndarray, target: np.ndarray, chunk_size: int = 128, cancel_event: Optional[Any] = None, search_index: Optional[_NearestNeighborIndex] = None, use_fast: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    index = search_index if search_index is not None else _NearestNeighborIndex(target, use_fast)
    return index.query(source, chunk_size, cancel_event)


def _coverage_balanced_mask(matched_targets: np.ndarray, distances: np.ndarray, mask: np.ndarray, strength: float, keep_fraction: Optional[float] = None) -> np.ndarray:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return mask
    eligible = np.flatnonzero(mask)
    if len(eligible) < 24:
        return mask

    points = matched_targets[eligible]
    min_v = np.min(points, axis=0)
    max_v = np.max(points, axis=0)
    extent = max_v - min_v
    if not np.isfinite(extent).all() or float(np.max(extent)) <= 1e-12:
        return mask

    grid_size = int(round(4 + 8 * strength))
    scale = np.where(extent > 1e-12, extent, 1.0)
    cell_coords = np.floor((points - min_v) / scale * grid_size).astype(np.int64)
    cell_coords = np.clip(cell_coords, 0, grid_size - 1)
    cell_ids = cell_coords[:, 0] + grid_size * (cell_coords[:, 1] + grid_size * cell_coords[:, 2])
    if keep_fraction is None:
        keep_fraction = max(0.35, 1.0 - 0.45 * strength)
    keep_fraction = max(0.05, min(1.0, float(keep_fraction)))
    target_keep = max(12, int(np.ceil(len(eligible) * keep_fraction)))
    if target_keep >= len(eligible):
        return mask

    order = np.argsort(distances[eligible], kind='mergesort')
    standard_keep = order[:target_keep]
    standard_mask = np.zeros_like(mask)
    standard_mask[eligible[standard_keep]] = True
    anchor_count = max(3, int(np.ceil(target_keep * (0.34 + 0.20 * (1.0 - strength)))))
    anchor_keep = standard_keep[:min(anchor_count, len(standard_keep))]

    core_cells = cell_ids[standard_keep]
    unique_core_cells = max(1, len(np.unique(core_cells)))
    average_per_cell = target_keep / unique_core_cells
    cap_multiplier = 3.0 - 1.25 * strength
    per_cell_cap = max(3, int(np.ceil(average_per_cell * cap_multiplier)))

    relaxed_fraction = min(1.0, keep_fraction + (1.0 - keep_fraction) * 0.25)
    relaxed_keep = max(target_keep, int(np.ceil(len(eligible) * relaxed_fraction)))
    candidate_order = order[:relaxed_keep]

    kept_local = []
    kept_set = set()
    cell_counts = {}
    for local_index in anchor_keep:
        local_index = int(local_index)
        kept_local.append(local_index)
        kept_set.add(local_index)
        cell_id = int(cell_ids[local_index])
        cell_counts[cell_id] = cell_counts.get(cell_id, 0) + 1

    for local_index in candidate_order:
        local_index = int(local_index)
        if local_index in kept_set:
            continue
        cell_id = int(cell_ids[local_index])
        count = cell_counts.get(cell_id, 0)
        if count >= per_cell_cap:
            continue
        kept_local.append(local_index)
        kept_set.add(local_index)
        cell_counts[cell_id] = count + 1
        if len(kept_local) >= target_keep:
            break

    min_soft_keep = max(6, int(np.ceil(target_keep * 0.75)))
    if len(kept_local) < min_soft_keep:
        return standard_mask

    if len(kept_local) < target_keep:
        for local_index in standard_keep:
            local_index = int(local_index)
            if local_index in kept_set:
                continue
            kept_local.append(local_index)
            kept_set.add(local_index)
            if len(kept_local) >= target_keep:
                break

    balanced = np.zeros_like(mask)
    balanced[eligible[np.array(kept_local, dtype=np.int64)]] = True
    return balanced


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors, dtype=np.float64)
    valid = np.isfinite(lengths) & (lengths > 1e-12)
    out[valid] = vectors[valid] / lengths[valid][:, None]
    return out


def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> Optional[np.ndarray]:
    if normals is None or matrix.shape != (4, 4):
        return None
    basis = matrix[:3, :3]
    try:
        normal_matrix = np.linalg.inv(basis).T
    except np.linalg.LinAlgError:
        return None
    transformed = (normal_matrix @ normals.T).T
    return _normalize_vectors(transformed)


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    if len(source) < 3 or len(target) < 3:
        return None
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    cov = src_centered.T @ tgt_centered
    try:
        u, _, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = tgt_centroid - rot @ src_centroid
    mat = np.identity(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def _similarity_transform(source: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    if len(source) < 3 or len(target) < 3:
        return None
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= 1e-12:
        return None
    cov = (tgt_centered.T @ src_centered) / float(len(source))
    try:
        u, singular_values, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / src_var)
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    trans = tgt_centroid - scale * (rot @ src_centroid)
    mat = np.identity(4, dtype=np.float64)
    mat[:3, :3] = scale * rot
    mat[:3, 3] = trans
    return mat


def _uniform_scale(matrix: np.ndarray) -> Optional[float]:
    if matrix.shape != (4, 4):
        return None
    det = float(np.linalg.det(matrix[:3, :3]))
    if not np.isfinite(det) or det <= 1e-12:
        return None
    return abs(det) ** (1.0 / 3.0)


def _scale_step_within_bounds(total: np.ndarray, delta: np.ndarray, max_deviation: float) -> bool:
    total_scale = _uniform_scale(total)
    delta_scale = _uniform_scale(delta)
    if total_scale is None or delta_scale is None:
        return False
    deviation = max(0.0, float(max_deviation))
    next_scale = total_scale * delta_scale
    return 1.0 - deviation <= next_scale <= 1.0 + deviation


def _validate_icp_transform(matrix: np.ndarray, allow_scale: bool, max_scale_deviation: float = 0.05) -> Tuple[bool, str]:
    if not allow_scale:
        return validate_rigid_transform(matrix)
    if matrix.shape != (4, 4):
        return False, "Transform must be 4x4"
    if not np.isfinite(matrix).all():
        return False, "Transform contains non-finite values"
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4):
        return False, "Invalid homogeneous transform"
    basis = matrix[:3, :3]
    det = float(np.linalg.det(basis))
    if not np.isfinite(det) or det <= 1e-12:
        return False, "Transform scale is invalid"
    scale = abs(det) ** (1.0 / 3.0)
    deviation = max(0.0, float(max_scale_deviation))
    if scale < 1.0 - deviation or scale > 1.0 + deviation:
        return False, "ICP Scale exceeded the allowed correction range"
    rot = basis / scale
    if not np.allclose(rot.T @ rot, np.identity(3), atol=1e-3):
        return False, "Transform is not uniform scale"
    if abs(float(np.linalg.det(rot)) - 1.0) > 1e-3:
        return False, "Transform rotation determinant is not 1"
    return True, ""


def _rodrigues(rotation_vector: np.ndarray) -> Optional[np.ndarray]:
    angle = float(np.linalg.norm(rotation_vector))
    if not np.isfinite(angle):
        return None
    if angle <= 1e-12:
        return np.identity(3, dtype=np.float64)
    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.identity(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _orthogonal_tangents(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    helper = np.zeros_like(normals)
    use_x = np.abs(normals[:, 2]) > 0.8
    helper[use_x, 0] = 1.0
    helper[~use_x, 2] = 1.0
    tangent_u = np.cross(normals, helper)
    tangent_u /= np.maximum(1e-12, np.linalg.norm(tangent_u, axis=1))[:, None]
    tangent_v = np.cross(normals, tangent_u)
    tangent_v /= np.maximum(1e-12, np.linalg.norm(tangent_v, axis=1))[:, None]
    return tangent_u, tangent_v


def _hybrid_point_to_plane_transform(source: np.ndarray, target: np.ndarray, normals: np.ndarray, tangent_weight: float, allow_scale: bool, weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if len(source) < 6 or len(target) < 6 or len(normals) < 6:
        return None
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normal_lengths) & (normal_lengths > 1e-12)
    if np.count_nonzero(valid) < 6:
        return None
    src = source[valid]
    tgt = target[valid]
    nrm = normals[valid] / normal_lengths[valid][:, None]
    point_weights = None
    if weights is not None:
        point_weights = np.asarray(weights, dtype=np.float64)[valid]
        point_weights = np.where(np.isfinite(point_weights), point_weights, 0.0)
        point_weights = np.maximum(0.0, point_weights)
        if float(np.sum(point_weights > 1e-8)) < 6:
            return None
        point_weights = np.sqrt(np.maximum(point_weights, 1e-8))[:, None]
    residual = tgt - src
    normal_row = np.column_stack((np.cross(src, nrm), nrm))
    if allow_scale:
        normal_row = np.column_stack((normal_row, np.sum(nrm * src, axis=1)))
    if point_weights is not None:
        normal_row = point_weights * normal_row
    rows = [normal_row]
    normal_values = np.sum(nrm * residual, axis=1)
    if point_weights is not None:
        normal_values = point_weights[:, 0] * normal_values
    values = [normal_values]

    tangent_weight = max(0.0, float(tangent_weight))
    if tangent_weight > 0.0:
        tangent_u, tangent_v = _orthogonal_tangents(nrm)
        tangent_u_row = np.column_stack((np.cross(src, tangent_u), tangent_u))
        tangent_v_row = np.column_stack((np.cross(src, tangent_v), tangent_v))
        if allow_scale:
            tangent_u_row = np.column_stack((tangent_u_row, np.sum(tangent_u * src, axis=1)))
            tangent_v_row = np.column_stack((tangent_v_row, np.sum(tangent_v * src, axis=1)))
        if point_weights is not None:
            tangent_u_row = point_weights * tangent_u_row
            tangent_v_row = point_weights * tangent_v_row
        rows.append(tangent_weight * tangent_u_row)
        rows.append(tangent_weight * tangent_v_row)
        tangent_u_values = np.sum(tangent_u * residual, axis=1)
        tangent_v_values = np.sum(tangent_v * residual, axis=1)
        if point_weights is not None:
            tangent_u_values = point_weights[:, 0] * tangent_u_values
            tangent_v_values = point_weights[:, 0] * tangent_v_values
        values.append(tangent_weight * tangent_u_values)
        values.append(tangent_weight * tangent_v_values)

    a = np.vstack(rows)
    b = np.concatenate(values)
    try:
        step, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 4 or not np.isfinite(step).all():
        return None

    rotation_vector = step[:3]
    rot_angle = float(np.linalg.norm(rotation_vector))
    if rot_angle > 0.35:
        rotation_vector *= 0.35 / rot_angle
    translation = step[3:6]
    trans_len = float(np.linalg.norm(translation))
    step_limit = max(1e-6, float(np.linalg.norm(np.max(src, axis=0) - np.min(src, axis=0))) * 0.25)
    if trans_len > step_limit:
        translation *= step_limit / trans_len

    rot = _rodrigues(rotation_vector)
    if rot is None:
        return None
    scale = 1.0
    if allow_scale and len(step) >= 7:
        scale = 1.0 + float(step[6])
        if not np.isfinite(scale) or scale <= 1e-6:
            return None
        scale = max(0.95, min(1.05, scale))
    mat = np.identity(4, dtype=np.float64)
    mat[:3, :3] = scale * rot
    mat[:3, 3] = translation
    return mat


def _gicp_like_fine_polish(source: np.ndarray, source_normals: Optional[np.ndarray], target: np.ndarray, target_normals: Optional[np.ndarray], transform: np.ndarray, settings: IcpSettings, cancel_event: Optional[Any] = None) -> Tuple[np.ndarray, float, int, int]:
    passes = max(0, int(getattr(settings, "fine_polish_passes", 0)))
    strength = max(0.0, min(1.0, float(getattr(settings, "fine_polish_strength", 0.0))))
    if passes <= 0 or strength <= 0.0 or source_normals is None or target_normals is None:
        return transform, -1.0, 0, 0
    if len(source) < 6 or len(target) < 6 or len(source_normals) != len(source) or len(target_normals) != len(target):
        return transform, -1.0, 0, 0

    total = transform.copy()
    residual = -1.0
    inlier_count = 0
    completed = 0
    max_dist = settings.max_correspondence_distance
    rejection_scale = max(0.0, float(settings.rejection_scale))
    tangent_weight = min(2.0, max(0.0, float(getattr(settings, "tangent_weight", 0.35)) + 0.35 * strength))
    allow_scale = bool(getattr(settings, "allow_scale", False))
    nearest_chunk_size = max(64, min(2048, int(getattr(settings, "nearest_chunk_size", 128))))
    max_scale_deviation = max(0.0, float(getattr(settings, "max_scale_deviation", 0.05)))

    for _ in range(passes):
        raise_if_cancelled(cancel_event)
        current = (total[:3, :3] @ source.T).T + total[:3, 3]
        current_normals = _transform_normals(source_normals, total)
        if current_normals is None:
            break

        nn_indices, distances = _nearest_neighbors_chunked(current, target, chunk_size=nearest_chunk_size, cancel_event=cancel_event)
        matched_targets = target[nn_indices]
        matched_normals = _normalize_vectors(target_normals[nn_indices])
        normal_alignment = np.abs(np.sum(current_normals * matched_normals, axis=1))

        mask = np.ones((len(current),), dtype=bool)
        if max_dist is not None and max_dist > 0.0:
            mask &= distances <= float(max_dist) * (0.8 + 0.4 * strength)
        if rejection_scale > 0.0 and len(distances) >= 8:
            masked_dist = distances[mask] if np.any(mask) else distances
            median = float(np.median(masked_dist))
            mad = float(np.median(np.abs(masked_dist - median)))
            robust_sigma = max(1e-12, 1.4826 * mad)
            mask &= distances <= median + max(rejection_scale, 3.5) * robust_sigma
        mask &= normal_alignment >= (0.35 + 0.35 * strength)
        inlier_count = int(np.count_nonzero(mask))
        if inlier_count < 6:
            break

        inlier_dist = distances[mask]
        dist_scale = max(1e-12, float(np.percentile(inlier_dist, 70)))
        distance_weight = np.exp(-np.square(distances / dist_scale))
        normal_weight = np.clip((normal_alignment - 0.25) / 0.75, 0.0, 1.0) ** (0.75 + 0.75 * strength)
        weights = np.clip((0.25 + 0.75 * strength) * distance_weight * normal_weight, 0.0, 1.0)
        weights = weights[mask]

        delta = _hybrid_point_to_plane_transform(
            current[mask],
            matched_targets[mask],
            matched_normals[mask],
            tangent_weight,
            allow_scale,
            weights,
        )
        if delta is None:
            break
        if allow_scale and not _scale_step_within_bounds(total, delta, max_scale_deviation):
            delta = _hybrid_point_to_plane_transform(
                current[mask],
                matched_targets[mask],
                matched_normals[mask],
                tangent_weight,
                False,
                weights,
            )
            if delta is None:
                break

        step_translation = float(np.linalg.norm(delta[:3, 3]))
        step_basis = float(np.linalg.norm(delta[:3, :3] - np.identity(3, dtype=np.float64)))
        if step_translation <= max(1e-12, float(settings.tolerance) * 0.1) and step_basis <= 1e-8:
            break

        proposed_total = delta @ total
        updated = (proposed_total[:3, :3] @ source.T).T + proposed_total[:3, 3]
        base_residual = float(np.mean(inlier_dist))
        residual = float(np.mean(np.linalg.norm(updated[mask] - matched_targets[mask], axis=1)))
        if not np.isfinite(residual) or residual > base_residual * 1.03 + 1e-12:
            break
        total = proposed_total
        completed += 1

    return total, residual, completed, inlier_count


def _alignment_quality(source: np.ndarray, target: np.ndarray, transform: np.ndarray, settings: IcpSettings, cancel_event: Optional[Any]) -> Tuple[float, float, float, float]:
    current = (transform[:3, :3] @ source.T).T + transform[:3, 3]
    use_fast = bool(getattr(settings, "use_fast_nearest", True))
    chunk_size = max(64, min(2048, int(getattr(settings, "nearest_chunk_size", 128))))
    index = _NearestNeighborIndex(target, use_fast)
    nn_indices, distances = _nearest_neighbors_chunked(current, target, chunk_size, cancel_event, index, use_fast)
    mask = np.isfinite(distances)
    max_dist = settings.max_correspondence_distance
    if max_dist is not None and max_dist > 0.0:
        mask &= distances <= float(max_dist)
    if np.count_nonzero(mask) >= 8:
        selected = distances[mask]
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        robust_sigma = max(1e-12, 1.4826 * mad)
        rejection_scale = max(2.5, float(settings.rejection_scale))
        mask &= distances <= median + rejection_scale * robust_sigma
    inlier_count = int(np.count_nonzero(mask))
    if inlier_count < 3:
        return 0.0, 0.0, -1.0, 0.0

    inlier_distances = distances[mask]
    overlap_ratio = inlier_count / max(1, len(source))
    p95 = float(np.percentile(inlier_distances, 95))

    matched = target[nn_indices[mask]]
    target_min = np.min(target, axis=0)
    target_extent = np.max(target, axis=0) - target_min
    scale = np.where(target_extent > 1e-12, target_extent, 1.0)
    grid_size = 10
    target_cells = np.floor((target - target_min) / scale * grid_size).astype(np.int64)
    matched_cells = np.floor((matched - target_min) / scale * grid_size).astype(np.int64)
    target_cells = np.clip(target_cells, 0, grid_size - 1)
    matched_cells = np.clip(matched_cells, 0, grid_size - 1)
    target_ids = target_cells[:, 0] + grid_size * (target_cells[:, 1] + grid_size * target_cells[:, 2])
    matched_ids = matched_cells[:, 0] + grid_size * (matched_cells[:, 1] + grid_size * matched_cells[:, 2])
    coverage_ratio = len(np.unique(matched_ids)) / max(1, len(np.unique(target_ids)))

    bbox_diag = float(np.linalg.norm(np.max(target, axis=0) - np.min(target, axis=0)))
    residual_scale = float(max_dist) if max_dist is not None and max_dist > 0.0 else max(1e-8, bbox_diag * 0.02)
    residual_score = float(np.exp(-p95 / max(1e-12, residual_scale)))
    coverage_score = min(1.0, coverage_ratio / 0.35)
    confidence = max(0.0, min(1.0, 0.55 * overlap_ratio + 0.20 * coverage_score + 0.25 * residual_score))
    return confidence, overlap_ratio, p95, coverage_ratio


def _success_result(source: np.ndarray, target: np.ndarray, settings: IcpSettings, transform: np.ndarray, residual: float, iterations: int, inliers: int, message: str, cancel_event: Optional[Any]) -> IcpResult:
    confidence, overlap_ratio, p95, coverage_ratio = _alignment_quality(source, target, transform, settings, cancel_event)
    return IcpResult(True, transform, residual, iterations, inliers, len(target), message, confidence, overlap_ratio, p95, coverage_ratio)


def run_icp(source_points: np.ndarray, target_points: np.ndarray, settings: IcpSettings, target_normals: Optional[np.ndarray] = None, source_normals: Optional[np.ndarray] = None, cancel_event: Optional[Any] = None) -> IcpResult:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    normals = np.asarray(target_normals, dtype=np.float64) if target_normals is not None else None
    src_normals = np.asarray(source_normals, dtype=np.float64) if source_normals is not None else None
    identity = np.identity(4, dtype=np.float64)

    if source.ndim != 2 or source.shape[1] != 3:
        return IcpResult(False, identity, -1.0, 0, 0, len(target), "Source points must be Nx3")
    if target.ndim != 2 or target.shape[1] != 3:
        return IcpResult(False, identity, -1.0, 0, 0, len(target), "Target points must be Nx3")
    if len(source) < 3 or len(target) < 3:
        return IcpResult(False, identity, -1.0, 0, 0, len(target), "ICP requires at least 3 points per object")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return IcpResult(False, identity, -1.0, 0, 0, len(target), "Point cloud contains non-finite values")
    if normals is not None:
        if normals.ndim != 2 or normals.shape != target.shape:
            normals = None
        elif not np.isfinite(normals).all():
            normals = None
    if src_normals is not None:
        if src_normals.ndim != 2 or src_normals.shape != source.shape:
            src_normals = None
        elif not np.isfinite(src_normals).all():
            src_normals = None

    current = source.copy()
    total = identity.copy()
    final_residual = -1.0
    inlier_count = 0

    max_dist = settings.max_correspondence_distance
    rejection_scale = max(0.0, float(settings.rejection_scale))
    trim_fraction = max(0.05, min(1.0, float(getattr(settings, "trim_fraction", 1.0))))
    stages = _stage_plan(len(source), len(target), int(settings.iterations), int(getattr(settings, "pyramid_levels", 1)))
    method = str(getattr(settings, "refinement_method", "AUTO")).upper()
    use_point_to_plane = method in {'AUTO', 'POINT_TO_PLANE'} and normals is not None
    tangent_weight = max(0.0, min(2.0, float(getattr(settings, "tangent_weight", 0.35))))
    coverage_balance = max(0.0, min(1.0, float(getattr(settings, "coverage_balance", 0.50))))
    allow_scale = bool(getattr(settings, "allow_scale", False))
    nearest_chunk_size = max(64, min(2048, int(getattr(settings, "nearest_chunk_size", 128))))
    max_scale_deviation = max(0.0, float(getattr(settings, "max_scale_deviation", 0.05)))
    use_fast_nearest = bool(getattr(settings, "use_fast_nearest", True))
    reciprocal_correspondence = bool(getattr(settings, "reciprocal_correspondence", True))
    exact_final_nearest = bool(getattr(settings, "exact_final_nearest", False))
    completed_iterations = 0

    try:
        for stage_index, (src_count, tgt_count, stage_iterations) in enumerate(stages):
            raise_if_cancelled(cancel_event)
            src_stage, src_normals_stage = _subset_pair(source, src_normals, src_count)
            tgt_stage, tgt_normals_stage = _subset_pair(target, normals, tgt_count)
            stage_transform = total.copy()
            current = (stage_transform[:3, :3] @ src_stage.T).T + stage_transform[:3, 3]
            prev_residual = None
            is_final_stage = stage_index == len(stages) - 1
            stage_use_fast = use_fast_nearest and not (is_final_stage and exact_final_nearest)
            target_index = _NearestNeighborIndex(tgt_stage, stage_use_fast)
            stage_max_dist = max_dist
            if max_dist is not None and max_dist > 0.0 and len(stages) > 1:
                stage_max_dist = float(max_dist) * (2.5 if stage_index == 0 else 1.5 if stage_index == 1 else 1.0)

            for _ in range(max(1, int(stage_iterations))):
                raise_if_cancelled(cancel_event)
                nn_indices, distances = _nearest_neighbors_chunked(current, tgt_stage, chunk_size=nearest_chunk_size, cancel_event=cancel_event, search_index=target_index, use_fast=stage_use_fast)
                matched_targets = tgt_stage[nn_indices]
                mask = np.ones((len(current),), dtype=bool)
                if stage_max_dist is not None and stage_max_dist > 0.0:
                    mask &= distances <= float(stage_max_dist)
                if trim_fraction < 1.0 and len(distances) >= 8:
                    stage_coverage = coverage_balance if is_final_stage else coverage_balance * 0.5
                    if stage_coverage > 0.0 and len(distances) >= 24:
                        mask = _coverage_balanced_mask(matched_targets, distances, mask, stage_coverage, trim_fraction)
                    else:
                        trim_count = max(3, int(np.ceil(len(distances) * trim_fraction)))
                        if trim_count < len(distances):
                            trim_limit = float(np.partition(distances, trim_count - 1)[trim_count - 1])
                            mask &= distances <= trim_limit
                if rejection_scale > 0.0 and len(distances) >= 8:
                    masked_dist = distances[mask] if np.any(mask) else distances
                    median = float(np.median(masked_dist))
                    mad = float(np.median(np.abs(masked_dist - median)))
                    robust_sigma = max(1e-12, 1.4826 * mad)
                    mask &= distances <= median + rejection_scale * robust_sigma
                if is_final_stage and src_normals_stage is not None and tgt_normals_stage is not None:
                    current_normals = _transform_normals(src_normals_stage, total)
                    if current_normals is not None:
                        matched_normals = _normalize_vectors(tgt_normals_stage[nn_indices])
                        normal_alignment = np.abs(np.sum(current_normals * matched_normals, axis=1))
                        normal_mask = mask & (normal_alignment >= 0.20)
                        if np.count_nonzero(normal_mask) >= max(6, int(np.count_nonzero(mask) * 0.35)):
                            mask = normal_mask
                if is_final_stage and reciprocal_correspondence and len(current) >= 12:
                    reverse_index = _NearestNeighborIndex(current, stage_use_fast)
                    reverse_indices, _ = _nearest_neighbors_chunked(matched_targets, current, chunk_size=nearest_chunk_size, cancel_event=cancel_event, search_index=reverse_index, use_fast=stage_use_fast)
                    reciprocal_mask = reverse_indices == np.arange(len(current), dtype=np.int64)
                    mutual_mask = mask & reciprocal_mask
                    if np.count_nonzero(mutual_mask) >= max(6, int(np.count_nonzero(mask) * 0.20)):
                        mask = mutual_mask
                inlier_count = int(np.count_nonzero(mask))
                if inlier_count < 3:
                    return IcpResult(False, total, final_residual, completed_iterations, inlier_count, len(target), "Not enough ICP inliers")

                src_in = current[mask]
                tgt_in = tgt_stage[nn_indices[mask]]
                normal_in = tgt_normals_stage[nn_indices[mask]] if tgt_normals_stage is not None else None
                delta = None
                if use_point_to_plane and is_final_stage and normal_in is not None:
                    delta = _hybrid_point_to_plane_transform(src_in, tgt_in, normal_in, tangent_weight, allow_scale)
                if delta is None:
                    delta = _similarity_transform(src_in, tgt_in) if allow_scale else _rigid_transform(src_in, tgt_in)
                if delta is None:
                    return IcpResult(False, total, final_residual, completed_iterations, inlier_count, len(target), "Rigid transform failed")
                if allow_scale and not _scale_step_within_bounds(total, delta, max_scale_deviation):
                    delta = _rigid_transform(src_in, tgt_in)
                    if delta is None:
                        return IcpResult(False, total, final_residual, completed_iterations, inlier_count, len(target), "Rigid transform failed")

                current = (delta[:3, :3] @ current.T).T + delta[:3, 3]
                total = delta @ total
                residual = float(np.mean(np.linalg.norm(current[mask] - tgt_in, axis=1)))
                final_residual = residual
                completed_iterations += 1

                if prev_residual is not None and abs(prev_residual - residual) <= max(1e-12, float(settings.tolerance)):
                    if is_final_stage:
                        valid, message = _validate_icp_transform(total, allow_scale, max_scale_deviation)
                        if valid:
                            polished, polished_residual, polish_iterations, polish_inliers = _gicp_like_fine_polish(source, src_normals, target, normals, total, settings, cancel_event)
                            if polish_iterations > 0:
                                total = polished
                                residual = polished_residual
                                completed_iterations += polish_iterations
                                inlier_count = polish_inliers
                                valid, message = _validate_icp_transform(total, allow_scale, max_scale_deviation)
                        if valid:
                            return _success_result(source, target, settings, total, residual, completed_iterations, inlier_count, "Converged", cancel_event)
                        return IcpResult(False, total, residual, completed_iterations, inlier_count, len(target), message)
                    break
                prev_residual = residual

        valid, message = _validate_icp_transform(total, allow_scale, max_scale_deviation)
        if valid:
            polished, polished_residual, polish_iterations, polish_inliers = _gicp_like_fine_polish(source, src_normals, target, normals, total, settings, cancel_event)
            if polish_iterations > 0:
                total = polished
                final_residual = polished_residual
                completed_iterations += polish_iterations
                inlier_count = polish_inliers
                valid, message = _validate_icp_transform(total, allow_scale, max_scale_deviation)
        if valid:
            return _success_result(source, target, settings, total, final_residual, completed_iterations, inlier_count, "Max iterations reached", cancel_event)
        return IcpResult(False, total, final_residual, completed_iterations, inlier_count, len(target), message)
    except IcpCancelled:
        return IcpResult(False, total, final_residual, completed_iterations, inlier_count, len(target), "ICP cancelled")
