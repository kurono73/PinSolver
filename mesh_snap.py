"""Vertex snapping using consistent evaluated geometry in Edit Mode."""
import bpy
from mathutils.bvhtree import BVHTree


_edit_cache = {}


@bpy.app.handlers.persistent
def invalidate(scene=None, depsgraph=None):
    if depsgraph is None or any(update.is_updated_geometry for update in depsgraph.updates):
        _edit_cache.clear()


def _edit_geometry(obj, depsgraph):
    original = obj.original
    key = (original.as_pointer(), depsgraph.as_pointer())
    cached = _edit_cache.get(key)
    if cached is not None:
        return cached
    evaluated = original.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if mesh is None or not mesh.polygons:
            return None
        vertices = [vertex.co.copy() for vertex in mesh.vertices]
        faces = [tuple(face.vertices) for face in mesh.polygons]
        geometry = (BVHTree.FromPolygons(vertices, faces), vertices, faces)
    finally:
        evaluated.to_mesh_clear()
    if len(_edit_cache) >= 4:
        _edit_cache.clear()
    _edit_cache[key] = geometry
    return geometry


def snap_vertex(obj, depsgraph, matrix, face_index, location, origin=None, direction=None):
    if obj is None or obj.type != 'MESH':
        return location
    try:
        if obj.original.data.is_editmode:
            if origin is None or direction is None:
                return location
            geometry = _edit_geometry(obj, depsgraph)
            if geometry is None:
                return location
            bvh, vertices, faces = geometry
            inverse = matrix.inverted()
            local_direction = inverse.to_3x3() @ direction
            if local_direction.length_squared < 1e-20:
                return location
            # Scene raycasts can report Edit Mode cage indices, not evaluated faces.
            local_hit, _, index, _ = bvh.ray_cast(inverse @ origin, local_direction.normalized())
            if local_hit is None:
                return location
            hit = matrix @ local_hit
            candidates = [matrix @ vertices[i] for i in faces[index]]
        else:
            mesh = obj.evaluated_get(depsgraph).data
            if not 0 <= face_index < len(mesh.polygons):
                return location
            hit = location
            candidates = [matrix @ mesh.vertices[i].co for i in mesh.polygons[face_index].vertices]
        return min(candidates, key=lambda vertex: (vertex-hit).length_squared) if candidates else location
    except (ReferenceError, RuntimeError, ValueError, IndexError):
        return location
