"""Point geometry detection and extraction for ICP sampling."""
import numpy as np


def has_points(context, targets=None):
    for obj in context.view_layer.objects:
        if targets is not None and obj not in targets:
            continue
        if not obj.visible_get():
            continue
        if obj.type == 'POINTCLOUD':
            return True
        if obj.type == 'MESH':
            if not len(obj.data.polygons) and len(obj.data.vertices):
                return True
            if any(m.type == 'NODES' and m.show_viewport for m in obj.modifiers):
                mesh, cloud, geometry = components(obj.evaluated_get(context.evaluated_depsgraph_get()))
                if cloud is not None and len(cloud.points) or mesh is not None and not len(mesh.polygons) and len(mesh.vertices):
                    return True
    return False


def components(obj):
    geometry = None
    mesh = obj.data if obj.type == 'MESH' else None
    cloud = obj.data if obj.type == 'POINTCLOUD' else None
    if hasattr(obj, 'evaluated_geometry'):
        try:
            geometry = obj.evaluated_geometry()
            mesh = geometry.mesh
            cloud = geometry.pointcloud
        except (AttributeError, TypeError, RuntimeError):
            pass
    # GeometrySet owns the component RNA; callers must retain it while reading.
    return mesh, cloud, geometry


def point_coordinates(obj):
    mesh, cloud, geometry = components(obj)
    arrays = []
    if cloud is not None:
        attr = cloud.attributes.get('position')
        if attr is not None:
            data = np.empty(len(attr.data) * 3, dtype=np.float64)
            attr.data.foreach_get('vector', data)
            arrays.append(data.reshape(-1, 3))
    if mesh is not None and not len(mesh.polygons) and len(mesh.vertices):
        data = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get('co', data)
        arrays.append(data.reshape(-1, 3))
    return np.concatenate(arrays) if arrays else np.empty((0, 3), dtype=np.float64)
