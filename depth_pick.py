"""Approximate visible-surface picking; no point geometry extraction or index building."""
import time
from collections import OrderedDict

import bpy
import gpu
import numpy as np
from mathutils import Matrix, Vector

_requests = set()
_images = OrderedDict()
_offscreen = None
_offscreen_size = None
_generation = 0
_rendering = False


@bpy.app.handlers.persistent
def invalidate(scene=None, depsgraph=None):
    global _generation
    if _rendering:
        return
    if depsgraph is None or any(u.is_updated_geometry or u.is_updated_transform for u in depsgraph.updates):
        _generation += 1
        _images.clear()


def release():
    global _offscreen, _offscreen_size
    cancel_all()
    invalidate()
    if _offscreen is not None:
        _offscreen.free()
    _offscreen, _offscreen_size = None, None


def render_depth(context, view, projection, size, targets=None):
    """Render a deterministic depth image without changing the user's camera pose."""
    global _offscreen, _offscreen_size, _rendering
    if bpy.app.background:
        raise RuntimeError('Point-cloud depth picking requires a Blender window and GPU context')
    candidates = [a for w in context.window_manager.windows for a in w.screen.areas if a.type == 'VIEW_3D']
    area = context.area if context.area and context.area.type == 'VIEW_3D' else next(iter(candidates), None)
    if area is None:
        raise RuntimeError('Open a 3D View for point-cloud depth picking')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    space = area.spaces.active
    context.view_layer.update()
    key = (context.scene.as_pointer(), context.view_layer.as_pointer(), context.scene.frame_current,
        _generation, tuple(np.asarray(view).ravel()), tuple(np.asarray(projection).ravel()), tuple(size),
        tuple(o.as_pointer() for o in targets) if targets is not None else None, space.as_pointer())
    if key in _images:
        _images.move_to_end(key)
        return _images[key]
    hidden = []
    old_overlay, old_shading, old_xray = space.overlay.show_overlays, space.shading.type, space.shading.show_xray
    _rendering = True
    try:
        if targets is not None:
            for obj in context.view_layer.objects:
                if obj not in targets and not obj.hide_get(view_layer=context.view_layer):
                    hidden.append(obj)
                    obj.hide_set(True, view_layer=context.view_layer)
        space.overlay.show_overlays = False
        space.shading.type = 'SOLID'
        space.shading.show_xray = False
        context.view_layer.update()
        if _offscreen_size != tuple(size):
            if _offscreen is not None:
                _offscreen.free()
            _offscreen = gpu.types.GPUOffScreen(*size)
            _offscreen_size = tuple(size)
        with context.temp_override(area=area, region=region):
            _offscreen.draw_view3d(context.scene, context.view_layer, space, region,
                Matrix(view), Matrix(projection), draw_background=False)
        with _offscreen.bind():
            result = np.array(gpu.state.active_framebuffer_get().read_depth(0, 0, *size), dtype=np.float32)
        result = result.reshape(size[1], size[0])
    finally:
        for obj in hidden:
            obj.hide_set(False, view_layer=context.view_layer)
        space.overlay.show_overlays, space.shading.type, space.shading.show_xray = old_overlay, old_shading, old_xray
        context.view_layer.update()
        _rendering = False
    _images[key] = result
    if len(_images) > 4:
        _images.popitem(last=False)
    return result


def pick_image(context, view, projection, size, mouse, radius, snap=False, surface=True, targets=None, pixel_filter=None, require_surface=False):
    depth = render_depth(context, view, projection, size, targets)
    x, y = mouse
    x0, y0 = max(0, int(x-radius)), max(0, int(y-radius))
    x1, y1 = min(size[0], int(np.ceil(x+radius))), min(size[1], int(np.ceil(y+radius)))
    if x0 >= x1 or y0 >= y1:
        return None
    mask = None
    if pixel_filter is not None:
        rows, cols = np.indices((y1-y0, x1-x0))
        mask = pixel_filter(np.column_stack((cols.ravel()+x0+.5, rows.ravel()+y0+.5))).reshape(y1-y0, x1-x0)
    point = resolve_depth(depth[y0:y1, x0:x1], (x0,y0,x1-x0,y1-y0), size,
        np.linalg.inv(np.asarray(projection) @ np.asarray(view)), np.asarray(view), mouse, radius, snap, surface, mask, require_surface)
    return Vector(point) if point is not None else None


def pick_view(context, region, rv3d, x, y, radius, snap=False, surface=True, targets=None):
    scale = min(1.0, 2048/max(region.width, region.height))
    size = (max(1, round(region.width*scale)), max(1, round(region.height*scale)))
    return pick_image(context, rv3d.view_matrix, rv3d.window_matrix, size,
        (x*size[0]/region.width, y*size[1]/region.height), radius*min(size[0]/region.width,size[1]/region.height),
        snap, surface, targets)


def centered_sample(points, pixels, ids, foreground, nearest, shape, mouse, radius):
    """Center an isolated displayed splat; do not average merged or clipped blobs."""
    lookup = {int(ids[i]): int(i) for i in foreground}
    seed = int(ids[nearest])
    stack, visited = [seed], {seed}
    height, width = shape
    while stack and len(visited) <= 4096:
        current = stack.pop()
        row, col = divmod(current, width)
        for r, c in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)):
            neighbor = r*width+c
            if 0 <= r < height and 0 <= c < width and neighbor in lookup and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    if len(visited) > 4096:
        return points[nearest]
    indices = np.array([lookup[i] for i in visited])
    xy = pixels[indices]
    if np.any(np.linalg.norm(xy-mouse, axis=1) >= radius-1.5):
        return points[nearest]
    center = xy.mean(axis=0)
    return points[indices[np.argmin(np.linalg.norm(xy-center, axis=1))]]


def surface_samples(points, pixels, ids, foreground, shape, mouse, radius):
    """Use one central sample per separated splat, or a well-covered dense patch."""
    coverage = len(foreground)/max(1.0, np.pi*radius*radius)
    if coverage > .65 or len(foreground) > 8192:
        return points[foreground]
    lookup = {int(ids[i]): int(i) for i in foreground}
    remaining = set(lookup)
    height, width = shape
    centers = []
    while remaining:
        seed = remaining.pop()
        stack, component = [seed], [lookup[seed]]
        while stack:
            row, col = divmod(stack.pop(), width)
            for r, c in ((row-1,col),(row+1,col),(row,col-1),(row,col+1)):
                neighbor = r*width+c
                if 0 <= r < height and 0 <= c < width and neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(lookup[neighbor])
        xy = pixels[component]
        central = component[int(np.argmin(np.linalg.norm(xy-xy.mean(axis=0), axis=1)))]
        centers.append(points[central])
    if len(centers) >= 3:
        return np.asarray(centers)
    # Dense splats can form one connected patch while leaving small pixel gaps.
    if len(foreground) >= 12 and np.min(np.ptp(pixels[foreground], axis=0)) > radius*1.3:
        return points[foreground]
    return np.asarray(centers)


def cancel_all():
    for request in tuple(_requests):
        request.cancel()


def resolve_depth(depth, rect, viewport_size, inverse_projection, view_matrix,
                  mouse, radius, snap=False, surface=True, pixel_mask=None, require_surface=False):
    x0, y0, width, height = rect
    rows, cols = np.indices((height, width))
    pixels = np.column_stack((cols.ravel()+x0+0.5, rows.ravel()+y0+0.5))
    z = np.asarray(depth, dtype=np.float64).ravel()
    distance = np.linalg.norm(pixels-np.asarray(mouse), axis=1)
    valid = np.isfinite(z) & (z > 0) & (z < 1) & (distance <= radius)
    if pixel_mask is not None:
        valid &= np.asarray(pixel_mask, dtype=bool).ravel()
    ids = np.flatnonzero(valid)
    if not len(ids):
        return None
    if not snap and len(ids) > 8192:
        ids = np.unique(np.r_[ids[np.linspace(0, len(ids)-1, 7680, dtype=int)],
            ids[np.argsort(distance[ids])[:256]], ids[np.argsort(z[ids])[:256]]])
    pixels, z, distance = pixels[ids], z[ids], distance[ids]
    clip = np.column_stack((2*pixels[:, 0]/viewport_size[0]-1,
        2*pixels[:, 1]/viewport_size[1]-1, 2*z-1, np.ones(len(z))))
    world = clip @ inverse_projection.T
    valid = np.abs(world[:, 3]) > 1e-12
    points = world[valid, :3]/world[valid, 3:4]
    distance = distance[valid]
    pixels, ids = pixels[valid], ids[valid]
    if not len(points):
        return None
    linear_depth = -(points @ view_matrix[2, :3]+view_matrix[2, 3])
    order = np.argsort(linear_depth)
    reference = clip[valid][int(np.argmin(distance))]
    offsets = np.array([[0,0,0,0], [2/viewport_size[0],0,0,0], [0,2/viewport_size[1],0,0]])
    sample = (reference+offsets) @ inverse_projection.T
    sample = sample[:, :3]/sample[:, 3:4]
    pixel_world = float(np.max(np.linalg.norm(sample[1:]-sample[0], axis=1)))
    tolerance = max(1e-7, pixel_world*max(2, radius*.25), float(np.median(np.abs(linear_depth)))*0.005)
    groups = np.split(order, np.flatnonzero(np.diff(linear_depth[order]) > tolerance)+1)
    foreground = groups[0]
    nearest = foreground[np.argmin(distance[foreground])]
    if snap:
        nearest = int(np.argmin(distance))
        snap_group = next(group for group in groups if np.any(group == nearest))
        return centered_sample(points, pixels, ids, snap_group, nearest, (height,width), np.asarray(mouse), radius)
    if not surface:
        return points[nearest]
    # Use a bounded, spatially distributed sample from the visible foreground.
    ranked = foreground[np.argsort(distance[foreground])]
    selection = np.unique(np.r_[ranked[:128],
        foreground[np.linspace(0, len(foreground)-1, min(384, len(foreground)), dtype=int)]])
    samples = surface_samples(points, pixels, ids, foreground, (height,width), np.asarray(mouse), radius)
    pts = samples if len(samples) <= 512 else points[selection]
    # Sparse support has no reliable normal: use a view-facing virtual patch
    # at its observed depth, never move the hit sideways to a point sample.
    if len(pts) < 3:
        if require_surface:
            return None
        center = pts.mean(axis=0)
        normal = np.asarray(view_matrix[2, :3], dtype=float)
        normal /= np.linalg.norm(normal)
        axes = np.vstack((view_matrix[0, :3], view_matrix[1, :3], normal))
    else:
        center, axes = fit_surface(pts, tolerance)
        if axes is None:
            if require_surface:
                return None
            axes = np.asarray(view_matrix[:3, :3], dtype=float)
    xy = np.asarray(mouse)/np.asarray(viewport_size)*2-1
    endpoints = np.array([[xy[0], xy[1], -1, 1], [xy[0], xy[1], 1, 1]]) @ inverse_projection.T
    if np.any(np.abs(endpoints[:, 3]) < 1e-12):
        return None
    endpoints = endpoints[:, :3]/endpoints[:, 3:4]
    direction = endpoints[1]-endpoints[0]
    direction /= max(np.linalg.norm(direction), 1e-12)
    denominator = float(axes[2] @ direction)
    if abs(denominator) < .005:
        return None
    t = float((center-endpoints[0]) @ axes[2]/denominator)
    hit = endpoints[0]+direction*t
    hit_depth = -(hit @ view_matrix[2, :3]+view_matrix[2, 3])
    margin = max(tolerance*2, pixel_world*radius*4)
    if t < 0 or hit_depth < linear_depth[foreground].min()-margin or hit_depth > linear_depth[foreground].max()+margin:
        return None
    return hit


def fit_surface(pts, tolerance):
    for _ in range(2):
        center = pts.mean(axis=0)
        _, singular, axes = np.linalg.svd(pts-center, full_matrices=False)
        residual = np.abs((pts-center) @ axes[2])
        keep = residual <= max(tolerance*.1, float(np.median(residual))*3, 1e-8)
        if keep.all() or keep.sum() < 3:
            break
        pts = pts[keep]
    center = pts.mean(axis=0)
    _, singular, axes = np.linalg.svd(pts-center, full_matrices=False)
    if singular[1] < 1e-12 or singular[2] > singular[1]*.35:
        return center, None
    return center, axes


class Request:
    def __init__(self, region, scene, x, y, radius, snap, surface):
        self.region_id = region.as_pointer()
        self.scene_id = scene.as_pointer()
        self.mouse = (x, y)
        self.radius = radius
        self.snap, self.surface = snap, surface
        self.done, self.point, self.error = False, None, None
        self.started = time.monotonic()
        self.space, self.show_extras = None, None
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D' and any(r.as_pointer() == self.region_id for r in area.regions):
                    self.space = area.spaces.active
                    self.show_extras = self.space.overlay.show_extras
                    # Camera/light helper drawings can write foreground depth.
                    self.space.overlay.show_extras = False
                    area.tag_redraw()
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self._read, (), 'WINDOW', 'POST_VIEW')
        _requests.add(self)

    def _read(self):
        context = bpy.context
        if _rendering or self.done or context.region is None or context.region.as_pointer() != self.region_id:
            return
        if context.scene.as_pointer() != self.scene_id:
            self.error, self.done = 'View scene changed during depth pick', True
            return
        try:
            region, rv3d = context.region, context.region_data
            vx, vy, width, height = gpu.state.viewport_get()
            # Read framebuffer pixels in its own resolution, not logical UI pixels.
            sx, sy = width/region.width, height/region.height
            x, y = self.mouse[0]*sx, self.mouse[1]*sy
            radius = self.radius*min(sx, sy)
            x0, y0 = max(0, int(x-radius)), max(0, int(y-radius))
            x1, y1 = min(width, int(np.ceil(x+radius))), min(height, int(np.ceil(y+radius)))
            if x1 > x0 and y1 > y0:
                data = gpu.state.active_framebuffer_get().read_depth(vx+x0, vy+y0, x1-x0, y1-y0)
                self.point = resolve_depth(np.asarray(data), (x0, y0, x1-x0, y1-y0),
                    (width, height), np.asarray(rv3d.perspective_matrix.inverted()),
                    np.asarray(rv3d.view_matrix), (x, y), radius, self.snap, self.surface)
        except Exception as exc:
            self.error = f'Viewport depth unavailable: {exc}'
        self.done = True

    def cancel(self):
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
            self.handle = None
        _requests.discard(self)
        if self.space is not None:
            try:
                self.space.overlay.show_extras = self.show_extras
            except ReferenceError:
                pass
            self.space = None
        if not self.done:
            self.error, self.done = 'Depth pick cancelled', True

    @property
    def timed_out(self):
        return time.monotonic()-self.started > 2.0
