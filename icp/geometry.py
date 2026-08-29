from typing import Any, Optional, Set, Tuple

import numpy as np
from mathutils import Matrix, Vector

from .cancel import raise_if_cancelled


def _selected_vertex_indices(obj) -> Set[int]:
    if not obj or not getattr(obj, "data", None):
        return set()
    if getattr(obj, "mode", "") == 'EDIT':
        try:
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            return {v.index for v in bm.verts if getattr(v, "select", False)}
        except Exception:
            pass
    return {v.index for v in obj.data.vertices if getattr(v, "select", False)}


def _selected_face_indices(obj) -> Set[int]:
    if not obj or not getattr(obj, "data", None):
        return set()
    if getattr(obj, "mode", "") == 'EDIT':
        try:
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            return {f.index for f in bm.faces if getattr(f, "select", False)}
        except Exception:
            pass
    return {p.index for p in obj.data.polygons if getattr(p, "select", False)}


def _vertex_group_indices(mesh, obj, vertex_group_name: str) -> Tuple[Optional[Set[int]], str]:
    group_name = (vertex_group_name or "").strip()
    if not group_name:
        return None, ""
    vertex_group = obj.vertex_groups.get(group_name) if obj and getattr(obj, "vertex_groups", None) else None
    if vertex_group is None:
        return None, f"Vertex Group '{group_name}' was not found"

    group_index = int(vertex_group.index)
    indices = set()
    for vertex in mesh.vertices:
        for group in getattr(vertex, "groups", []):
            if int(group.group) == group_index and float(group.weight) > 0.0:
                indices.add(vertex.index)
                break
    if not indices:
        return None, f"Vertex Group '{group_name}' has no weighted vertices"
    return indices, ""


def extract_world_triangles(obj, depsgraph, use_evaluated_mesh: bool, selected_faces_only: bool, selected_vertices_only: bool = False, vertex_group_name: str = "", cancel_event: Optional[Any] = None) -> Tuple[Optional[np.ndarray], str]:
    if not obj or obj.type != 'MESH':
        return None, "Object must be a mesh"

    eval_obj = obj.evaluated_get(depsgraph) if use_evaluated_mesh else obj
    mesh = None
    try:
        mesh = eval_obj.to_mesh() if use_evaluated_mesh else obj.data
        if not mesh:
            return None, "Could not read mesh"
        mesh.calc_loop_triangles()

        selected_poly_indices = None
        if selected_faces_only:
            selected_poly_indices = _selected_face_indices(obj)
            if not selected_poly_indices:
                return None, "Selected Faces is enabled, but no faces are selected"

        selected_vertex_indices = None
        if selected_vertices_only:
            selected_vertex_indices = _selected_vertex_indices(obj)
            if not selected_vertex_indices:
                return None, "Selected Vertices is enabled, but no vertices are selected"

        vertex_group_indices, message = _vertex_group_indices(mesh, obj, vertex_group_name)
        if message:
            return None, message

        matrix = eval_obj.matrix_world if use_evaluated_mesh else obj.matrix_world
        triangles = []
        for tri_index, tri in enumerate(mesh.loop_triangles):
            if tri_index % 1024 == 0:
                raise_if_cancelled(cancel_event)
            if selected_poly_indices is not None and tri.polygon_index not in selected_poly_indices:
                continue
            if selected_vertex_indices is not None and not all(i in selected_vertex_indices for i in tri.vertices):
                continue
            if vertex_group_indices is not None and not all(i in vertex_group_indices for i in tri.vertices):
                continue
            verts = [matrix @ mesh.vertices[i].co for i in tri.vertices]
            arr = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
            if not np.isfinite(arr).all():
                continue
            area = 0.5 * np.linalg.norm(np.cross(arr[1] - arr[0], arr[2] - arr[0]))
            if area <= 1e-12:
                continue
            triangles.append(arr)

        if not triangles:
            return None, "Mesh mask contains no usable triangles"
        return np.stack(triangles, axis=0), ""
    finally:
        if use_evaluated_mesh and mesh is not None:
            eval_obj.to_mesh_clear()


def sample_surface_points(triangles: np.ndarray, sample_count: int, seed: int, cancel_event: Optional[Any] = None) -> np.ndarray:
    points, _ = sample_surface_points_with_normals(triangles, sample_count, seed, cancel_event)
    return points


def sample_surface_points_with_normals(triangles: np.ndarray, sample_count: int, seed: int, cancel_event: Optional[Any] = None, focus_points: Optional[np.ndarray] = None, focus_fraction: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    raise_if_cancelled(cancel_event)
    tris = np.asarray(triangles, dtype=np.float64)
    count = max(3, int(sample_count))
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    areas = 0.5 * normal_lengths
    valid = np.isfinite(areas) & (areas > 1e-12)
    tris = tris[valid]
    areas = areas[valid]
    normals = normals[valid] / normal_lengths[valid][:, None]
    if len(tris) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty

    rng = np.random.default_rng(int(seed))
    probs = areas / areas.sum()
    raise_if_cancelled(cancel_event)
    focus = np.asarray(focus_points, dtype=np.float64) if focus_points is not None else np.empty((0, 3), dtype=np.float64)
    if focus.ndim != 2 or focus.shape[1:] != (3,) or not np.isfinite(focus).all():
        focus = np.empty((0, 3), dtype=np.float64)
    focused_count = min(count // 3, int(round(count * max(0.0, min(0.35, float(focus_fraction)))))) if len(focus) else 0
    global_count = count - focused_count
    indices = rng.choice(len(tris), size=global_count, replace=True, p=probs)
    if focused_count > 0:
        centroids = np.mean(tris, axis=1)
        per_focus = max(1, int(np.ceil(focused_count / len(focus))))
        focused_indices = []
        local_pool_size = min(len(tris), max(48, int(np.ceil(len(tris) * 0.025))))
        for point in focus:
            raise_if_cancelled(cancel_event)
            dist_sq = np.sum(np.square(centroids - point[None, :]), axis=1)
            if local_pool_size < len(tris):
                local_pool = np.argpartition(dist_sq, local_pool_size - 1)[:local_pool_size]
            else:
                local_pool = np.arange(len(tris), dtype=np.int64)
            local_areas = areas[local_pool]
            local_probs = local_areas / max(1e-20, float(np.sum(local_areas)))
            focused_indices.extend(rng.choice(local_pool, size=per_focus, replace=True, p=local_probs).tolist())
        indices = np.concatenate((indices, np.asarray(focused_indices[:focused_count], dtype=np.int64)))
    picked = tris[indices]
    picked_normals = normals[indices]

    raise_if_cancelled(cancel_event)
    u = rng.random(count)
    v = rng.random(count)
    sqrt_u = np.sqrt(u)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v
    points = picked[:, 0] * w0[:, None] + picked[:, 1] * w1[:, None] + picked[:, 2] * w2[:, None]
    raise_if_cancelled(cancel_event)
    return points, picked_normals


def _mesh_geometry_signature(mesh) -> Tuple:
    if not mesh:
        return (-1, -1, 0.0, 0.0, 0.0, 0.0)
    vertex_count = len(mesh.vertices)
    loop_count = len(mesh.loops)
    if vertex_count == 0:
        return (0, loop_count, 0.0, 0.0, 0.0, 0.0)

    coords = np.empty(vertex_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    if not np.isfinite(coords).all():
        return (vertex_count, loop_count, float("nan"), 0.0, 0.0, 0.0)
    weights = np.arange(1, len(coords) + 1, dtype=np.float64)
    coord_sum = float(np.sum(coords))
    coord_weighted_sum = float(np.dot(coords, weights))

    if loop_count == 0:
        return (vertex_count, 0, round(coord_sum, 7), round(coord_weighted_sum, 7), 0.0, 0.0)
    loop_vertices = np.empty(loop_count, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    loop_weights = np.arange(1, loop_count + 1, dtype=np.int64)
    loop_sum = int(np.sum(loop_vertices, dtype=np.int64))
    loop_weighted_sum = int(np.dot(loop_vertices.astype(np.int64), loop_weights))
    return (vertex_count, loop_count, round(coord_sum, 7), round(coord_weighted_sum, 7), loop_sum, loop_weighted_sum)


def object_fingerprint(obj, depsgraph, use_evaluated_mesh: bool) -> Tuple:
    eval_obj = obj.evaluated_get(depsgraph) if use_evaluated_mesh else obj
    mesh = None
    try:
        mesh = eval_obj.to_mesh() if use_evaluated_mesh else obj.data
        vertex_count = len(mesh.vertices) if mesh else -1
        poly_count = len(mesh.polygons) if mesh else -1
        geometry_signature = _mesh_geometry_signature(mesh)
        matrix = eval_obj.matrix_world
    finally:
        if use_evaluated_mesh and mesh is not None:
            eval_obj.to_mesh_clear()

    matrix_values = tuple(round(float(matrix[r][c]), 9) for r in range(4) for c in range(4))
    modifiers = tuple((m.name, m.type, bool(m.show_viewport)) for m in getattr(obj, "modifiers", []))
    data_name = obj.data.name_full if getattr(obj, "data", None) else ""
    scene_eval = getattr(depsgraph, "scene_eval", None)
    frame_current = int(getattr(scene_eval, "frame_current", -1))
    return (obj.name_full, data_name, vertex_count, poly_count, geometry_signature, matrix_values, modifiers, frame_current)
