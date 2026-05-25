# PinSolver

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import blf
from bpy.props import (StringProperty, FloatVectorProperty, CollectionProperty, 
                       IntProperty, BoolProperty, EnumProperty, PointerProperty, FloatProperty)
from bpy.types import PropertyGroup, Operator, Panel, UIList
from mathutils import Vector, Matrix
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d, location_3d_to_region_2d
import numpy as np
import colorsys
import traceback
import math
from typing import Tuple, Optional, Any, List, Dict, Set

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ==========================================
# Configuration / Constants
# ==========================================
class PinSolverConfig:
    PIN_SELECTION_THRESHOLD_PX: float = 30.0
    REPROJ_ERROR_GOOD_THRESHOLD_PX: float = 5.0 
    MIN_PINS_FOR_SOLVE: int = 3
    MIN_PINS_FOR_CALIBRATION: int = 6 
    RAYCAST_MAX_RETRIES: int = 10
    RAYCAST_START_OFFSET: float = 0.1
    PICK_DELAY_FRAMES: int = 3
    DEFAULT_ADD_PIN_OFFSET: float = 5.0

class UIStrings:
    OVERLAY_EDIT_TITLE = "PinSolver - Pin Editor Mode"
    OVERLAY_EDIT_SUB_LAYOUT = "Drag 2D(●)/3D(■) [Shift: Force 2D, Ctrl: Force 3D, Alt: Snap] | A: Add | X: Del | ESC: Exit"
    OVERLAY_EDIT_SUB_MM = "Drag 3D(■) [Ctrl: Force 3D, Alt: Snap] | A: Add | X: Del | ESC: Exit"
    OVERLAY_TWEAK_ADD = "PinSolver - Add Pins (A)"
    OVERLAY_TWEAK_PAN = "PinSolver - Tweak (1 Pin: Pan)"
    OVERLAY_TWEAK_ORBIT = "PinSolver - Tweak (2 Pins: Orbit)"
    OVERLAY_TWEAK_FULL = "PinSolver - Tweak ({count} Pins: Full Solve)"
    OVERLAY_TWEAK_SUB = "Drag 3D Pin(■) to Move [{target}] | A: Add | X: Del | ESC: Exit"

# ==========================================
# 0. Utilities & Pure OpenCV Native Math
# ==========================================
def redraw_all_3d_views(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def get_3d_region_context(context: bpy.types.Context, event=None, cross_window: bool = False) -> Tuple[Any, Any, float, float]:
    if cross_window and event:
        mouse_x, mouse_y = event.mouse_x, event.mouse_y
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D' and (area.x <= mouse_x <= area.x + area.width) and (area.y <= mouse_y <= area.y + area.height):
                    for region in area.regions:
                        if region.type == 'WINDOW' and (region.x <= mouse_x <= region.x + region.width) and (region.y <= mouse_y <= region.y + region.height):
                            for space in area.spaces:
                                if space.type == 'VIEW_3D':
                                    return region, space.region_3d, mouse_x - region.x, mouse_y - region.y
        return None, None, 0.0, 0.0

    region = None
    rv3d = None
    if context.area and context.area.type == 'VIEW_3D':
        for r in context.area.regions:
            if r.type == 'WINDOW':
                region = r
                break
        if hasattr(context.area.spaces.active, "region_3d"):
            rv3d = context.area.spaces.active.region_3d

    if region and rv3d and event:
        return region, rv3d, event.mouse_x - region.x, event.mouse_y - region.y
        
    return region, rv3d, 0.0, 0.0

def _get_active_camera_view(context: bpy.types.Context) -> Tuple[Any, Any]:
    region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
    if region and rv3d and rv3d.view_perspective == 'CAMERA':
        return region, rv3d
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and space.region_3d.view_perspective == 'CAMERA':
                        for r in area.regions:
                            if r.type == 'WINDOW':
                                return r, space.region_3d
    return None, None

def get_camera_frame_bounds(context: bpy.types.Context, region=None, rv3d=None) -> Optional[Tuple[float, float, float, float]]:
    scene = context.scene
    camera = scene.camera
    if not region or not rv3d: 
        region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
    if not region or not rv3d or not camera or rv3d.view_perspective != 'CAMERA': 
        return None
        
    frame_local = camera.data.view_frame(scene=scene)
    depsgraph = context.evaluated_depsgraph_get()
    eval_cam = camera.evaluated_get(depsgraph)
    cam_mat = eval_cam.matrix_world
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for pt in frame_local:
        pt_world = cam_mat @ pt
        pt_2d = location_3d_to_region_2d(region, rv3d, pt_world)
        if not pt_2d: return None
        min_x = min(min_x, pt_2d.x)
        max_x = max(max_x, pt_2d.x)
        min_y = min(min_y, pt_2d.y)
        max_y = max(max_y, pt_2d.y)
        
    if (max_x - min_x) < 1e-4 or (max_y - min_y) < 1e-4: return None
    return min_x, min_y, max_x, max_y

def to_cv_pixel(u: float, v: float, res_x: float, res_y: float) -> Tuple[float, float]:
    return u * res_x, (1.0 - v) * res_y

def get_cv_camera_params(context: bpy.types.Context, cam_data: Any) -> Tuple[np.ndarray, np.ndarray, float, float]:
    clip = cam_data.target_clip
    render = context.scene.render
    pct = render.resolution_percentage / 100.0
    res_x = max(1.0, render.resolution_x * pct)
    res_y = max(1.0, render.resolution_y * pct)
    
    if clip:
        clipcam = clip.tracking.camera
        focal = clipcam.focal_length_pixels
        optcent = clipcam.principal if bpy.app.version < (3, 5, 0) else clipcam.principal_point_pixels
            
        camintr = np.array([
            [focal, 0, optcent[0]],
            [0, focal, clip.size[1] - optcent[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        k1 = clipcam.k1 if clipcam.distortion_model == 'POLYNOMIAL' else 0.0
        k2 = clipcam.k2 if clipcam.distortion_model == 'POLYNOMIAL' else 0.0
        k3 = clipcam.k3 if clipcam.distortion_model == 'POLYNOMIAL' else 0.0
        distcoef = np.array([k1, k2, 0, 0, k3], dtype=np.float64)
        return camintr, distcoef, float(clip.size[0]), float(clip.size[1])
    else:
        max_res = max(res_x, res_y)
        cam_ref = context.scene.camera.data
        
        if cam_ref.sensor_fit == 'VERTICAL' or (cam_ref.sensor_fit == 'AUTO' and res_x < res_y):
            sw = cam_ref.sensor_height * (res_x / res_y)
            sh = cam_ref.sensor_height
        else:
            sw = cam_ref.sensor_width
            sh = cam_ref.sensor_width * (res_y / res_x)
            
        fx = (cam_ref.lens / max(1e-4, sw)) * res_x
        fy = (cam_ref.lens / max(1e-4, sh)) * res_y
        
        cx = res_x / 2.0 + (cam_ref.shift_x * max_res)
        cy = res_y / 2.0 - (cam_ref.shift_y * max_res)
        
        camintr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return camintr, np.zeros(5, dtype=np.float64), res_x, res_y

def sync_scene_camera_from_clip(context: bpy.types.Context, cam_data: Any) -> bool:
    clip = cam_data.target_clip
    camera_obj = context.scene.camera
    if not clip or not camera_obj or not camera_obj.data:
        return False
        
    trk_cam = clip.tracking.camera
    cam_ref = camera_obj.data
    res_x = max(1.0, float(clip.size[0]))
    res_y = max(1.0, float(clip.size[1]))
    max_res = max(res_x, res_y)
    optcent = trk_cam.principal if bpy.app.version < (3, 5, 0) else trk_cam.principal_point_pixels
    
    try:
        cam_ref.sensor_width = trk_cam.sensor_width
        cam_ref.lens = trk_cam.focal_length
        cam_ref.shift_x = (float(optcent[0]) - res_x / 2.0) / max_res
        cam_ref.shift_y = (float(optcent[1]) - res_y / 2.0) / max_res
        return True
    except Exception:
        return False

def _get_undistorted_2d_coords_cached(p2d: Vector, bounds: Tuple, camintr: np.ndarray, distcoef: np.ndarray, res_x: float, res_y: float) -> Optional[Vector]:
    if not bounds or p2d is None: return None
    bounds_width = max(1e-4, bounds[2] - bounds[0])
    bounds_height = max(1e-4, bounds[3] - bounds[1])
    
    px, py = to_cv_pixel(p2d.x, p2d.y, res_x, res_y)
    
    has_distortion = np.linalg.norm(distcoef) > 1e-8
    if HAS_OPENCV and has_distortion:
        pt_2d = np.array([[[px, py]]], dtype=np.float64)
        undist = cv2.undistortPoints(pt_2d, camintr, distcoef, P=camintr)
        px, py = undist[0][0][0], undist[0][0][1]
        
    u_undist = px / res_x
    v_undist = 1.0 - (py / res_y)
    return Vector((bounds[0] + u_undist * bounds_width, bounds[1] + v_undist * bounds_height))

def mouse_to_distorted_uv(context: bpy.types.Context, region_x: float, region_y: float, region=None, rv3d=None) -> Tuple[float, float]:
    if not region or not rv3d: 
        region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
    bounds = get_camera_frame_bounds(context, region, rv3d)
    if not bounds: return 0.5, 0.5
    
    bounds_width = max(1e-4, bounds[2] - bounds[0])
    bounds_height = max(1e-4, bounds[3] - bounds[1])
    u_undist = (region_x - bounds[0]) / bounds_width
    v_undist = (region_y - bounds[1]) / bounds_height
    
    cam_data = context.scene.camera.pinsolver_data
    camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
    
    px_u = u_undist * res_x
    py_u = (1.0 - v_undist) * res_y
    
    has_distortion = np.linalg.norm(distcoef) > 1e-8
    if HAS_OPENCV and has_distortion:
        X_c = (px_u - camintr[0,2]) / camintr[0,0]
        Y_c = (py_u - camintr[1,2]) / camintr[1,1]
        pts_3d = np.array([[[X_c, Y_c, 1.0]]], dtype=np.float64)
        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camintr, distcoef)
        px_d, py_d = pts_2d[0][0][0], pts_2d[0][0][1]
    else:
        px_d, py_d = px_u, py_u
        
    return px_d / res_x, 1.0 - (py_d / res_y)

def get_pin_pixel_coords(context: bpy.types.Context, p2d: Vector, bounds=None, region=None, rv3d=None) -> Tuple[float, float]:
    if p2d is None: return None
    if bounds is None:
        if not region or not rv3d: region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
        bounds = get_camera_frame_bounds(context, region, rv3d)
        
    if bounds:
        bounds_width = bounds[2] - bounds[0]
        bounds_height = bounds[3] - bounds[1]
        return (bounds[0] + p2d.x * bounds_width, bounds[1] + p2d.y * bounds_height)
    return (p2d.x * (region.width if region else 100), p2d.y * (region.height if region else 100))

def get_track_marker_co(track: Any, marker: Any, cancel_offset: bool = True) -> Vector:
    co = Vector((marker.co[0], marker.co[1]))
    if cancel_offset:
        try:
            # Parent-offset trackers should solve from the parent point, not the shifted display point.
            offset = getattr(track, "offset", None)
            if offset is not None:
                co += Vector((offset[0], offset[1]))
        except Exception:
            pass
    return co

def get_pins(cam_data: Any, target_data: Any) -> Tuple[Any, int]:
    if not cam_data or not target_data: return [], 0
    if cam_data.ui_mode == 'LAYOUT':
        return target_data.layout_pins, target_data.layout_pin_idx
    else:
        return target_data.mm_pins, target_data.mm_pin_idx

def set_pin_idx(cam_data: Any, target_data: Any, val: int):
    if not cam_data or not target_data: return
    if cam_data.ui_mode == 'LAYOUT':
        target_data.layout_pin_idx = val
    else:
        target_data.mm_pin_idx = val

def get_current_pin_pos_2d(context: bpy.types.Context, cam_data: Any, pin: Any) -> Optional[Vector]:
    if cam_data.ui_mode == 'MATCHMOVE' and pin.is_track_linked:
        clip = cam_data.target_clip
        if clip:
            try:
                idx = int(cam_data.tracking_object_idx)
                if idx < len(clip.tracking.objects):
                    tracks = clip.tracking.objects[idx].tracks
                    track = tracks.get(pin.track_name)
                    if track:
                        scn_f = context.scene.frame_current
                        c_f = scn_f - clip.frame_start + clip.frame_offset + 1
                        
                        frames_to_check = [c_f, c_f - 1, scn_f, scn_f + clip.frame_offset]
                        marker = None
                        for f in frames_to_check:
                            m = track.markers.find_frame(f)
                            if m: 
                                marker = m
                                break
                                
                        if not marker:
                            for m in track.markers:
                                if m.frame in frames_to_check:
                                    marker = m
                                    break
                                    
                        if marker and not getattr(marker, 'mute', False):
                            return get_track_marker_co(track, marker)
            except: pass
        return None 
    return Vector(pin.pos_2d)

def get_closest_pin_item(context: bpy.types.Context, cam_data: Any, target_data: Any, mouse_pos_vec: Vector, region=None, rv3d=None, prefer_type: str = 'NONE') -> Tuple[int, str]:
    if not region or not rv3d: 
        region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
    closest_idx = -1
    closest_type = 'NONE'
    min_dist = PinSolverConfig.PIN_SELECTION_THRESHOLD_PX
    
    bounds = get_camera_frame_bounds(context, region, rv3d)
    camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
    
    pins, active_idx = get_pins(cam_data, target_data)
    num_pins = len(pins)
    eval_order = [active_idx] + [i for i in range(num_pins) if i != active_idx] if 0 <= active_idx < num_pins else list(range(num_pins))

    for i in eval_order:
        pin = pins[i]
        
        is_active = pin.use_initial if cam_data.ui_mode == 'MATCHMOVE' else True
        if not is_active or not pin.has_valid_3d: continue
        
        p2d = get_current_pin_pos_2d(context, cam_data, pin)
        if prefer_type in {'NONE', '2D'} and p2d is not None:
            if cam_data.use_distortion_overlay:
                coord_2d_pin = _get_undistorted_2d_coords_cached(p2d, bounds, camintr, distcoef, res_x, res_y)
            else:
                px = get_pin_pixel_coords(context, p2d, bounds=bounds)
                coord_2d_pin = Vector(px) if px else None
                
            if coord_2d_pin:
                dist_2d = (mouse_pos_vec - coord_2d_pin).length
                if dist_2d < min_dist:
                    min_dist = dist_2d
                    closest_idx = i
                    closest_type = '2D'
                    
        if prefer_type in {'NONE', '3D'}:
            coord_3d_pin = location_3d_to_region_2d(region, rv3d, Vector(pin.pos_3d))
            if coord_3d_pin:
                dist_3d = (mouse_pos_vec - coord_3d_pin).length
                if dist_3d < min_dist:
                    min_dist = dist_3d
                    closest_idx = i
                    closest_type = '3D'
                
    return closest_idx, closest_type

def get_closest_pin_index(context: bpy.types.Context, cam_data: Any, target_data: Any, mouse_pos_vec: Vector, 
                          mode_filter: Optional[str] = None, is_tweak: bool = False,
                          region=None, rv3d=None, prefer_type: str = 'NONE') -> int:
    idx, _ = get_closest_pin_item(context, cam_data, target_data, mouse_pos_vec, region, rv3d, prefer_type)
    pins, _ = get_pins(cam_data, target_data)
    if idx != -1:
        if mode_filter == 'TWEAK' and not pins[idx].use_tweak: 
            return -1
    return idx

def get_active_target_data(context: bpy.types.Context) -> Tuple[Optional[Any], Optional[Any], Optional[bpy.types.Object]]:
    camera = context.scene.camera
    if not camera or not hasattr(camera, "pinsolver_data"): return None, None, None
    cam_data = camera.pinsolver_data
    if cam_data.solve_mode in {'CAMERA', 'PARENT'}: 
        return cam_data, cam_data, camera
    else: 
        if not cam_data.target_objects: return cam_data, None, None
        idx = cam_data.active_target_index
        if idx < 0 or idx >= len(cam_data.target_objects): return cam_data, None, None
        target_obj = cam_data.target_objects[idx].obj
        if not target_obj: return cam_data, None, None
        return cam_data, target_obj.pinsolver_data, target_obj

def update_distortion_model(self, context: bpy.types.Context) -> None:
    if self.target_clip and (self.calib_k1 or self.calib_k2 or self.calib_k3):
        self.target_clip.tracking.camera.distortion_model = 'POLYNOMIAL'

def update_reproj_errors(context: bpy.types.Context, cam_data: Any, target_data: Any, force_update: bool = False) -> None:
    pins, _ = get_pins(cam_data, target_data)
    
    if cam_data.ui_mode == 'MATCHMOVE' and not force_update:
        return

    if cam_data.is_tweak_mode:
        if cam_data.ui_mode == 'LAYOUT': target_data.layout_avg_error = -1.0
        else: target_data.mm_avg_error = -1.0
        for pin in pins: pin.reproj_error = -1.0
        return

    if cam_data.ui_mode == 'LAYOUT':
        cam_region, cam_rv3d = _get_active_camera_view(context)
        if not cam_region or not cam_rv3d: return
        bounds = get_camera_frame_bounds(context, cam_region, cam_rv3d)
        if not bounds: return
        
        camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
        total_error = 0.0
        valid_count = 0
        
        bw = max(1e-4, bounds[2] - bounds[0])
        bh = max(1e-4, bounds[3] - bounds[1])
        
        for pin in pins:
            is_active = True
            if not cam_data.is_edit_mode and not pin.use_initial: is_active = False
            p2d = get_current_pin_pos_2d(context, cam_data, pin)
            
            if not is_active or p2d is None or not pin.has_valid_3d:
                pin.reproj_error = -1.0
                continue
                
            c2d_3dpin = location_3d_to_region_2d(cam_region, cam_rv3d, Vector(pin.pos_3d))
            if cam_data.use_distortion_overlay:
                c2d_2dpin_display = _get_undistorted_2d_coords_cached(p2d, bounds, camintr, distcoef, res_x, res_y)
            else:
                px = get_pin_pixel_coords(context, p2d, bounds=bounds)
                c2d_2dpin_display = Vector(px) if px else None
            
            if c2d_3dpin and c2d_2dpin_display:
                err_vp_x = abs(c2d_3dpin.x - c2d_2dpin_display.x)
                err_vp_y = abs(c2d_3dpin.y - c2d_2dpin_display.y)
                err_img_x = (err_vp_x / bw) * res_x
                err_img_y = (err_vp_y / bh) * res_y
                err = np.sqrt(err_img_x**2 + err_img_y**2)
                
                pin.reproj_error = float(err)
                total_error += float(err)
                valid_count += 1
            else:
                pin.reproj_error = -1.0
                
        target_data.layout_avg_error = (total_error / valid_count) if valid_count > 0 else -1.0

    else:
        if not force_update: return
        context.view_layer.update()
        
        depsgraph = context.evaluated_depsgraph_get()
        camera = context.scene.camera
        if not camera: return
        eval_cam = camera.evaluated_get(depsgraph)
        cam_mat = eval_cam.matrix_world
        
        camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
        
        total_error = 0.0
        valid_count = 0
        
        if HAS_OPENCV:
            cam_loc, cam_rot, _ = cam_mat.decompose()
            cam_mat_unscaled = Matrix.LocRotScale(cam_loc, cam_rot, Vector((1.0, 1.0, 1.0)))
            
            R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            R_world2bcam = np.array(cam_mat_unscaled.to_3x3().transposed(), dtype=np.float64)
            R_world2cv = R_bcam2cv @ R_world2bcam
            
            rvec, _ = cv2.Rodrigues(R_world2cv)
            tvec = -R_world2cv @ np.array(cam_mat_unscaled.translation, dtype=np.float64)
            
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

            obj_pts = []
            img_pts = []
            valid_pins_eval = []
            
            for pin in pins:
                if not pin.use_initial: 
                    pin.reproj_error = -1.0
                    continue 
                    
                p2d = get_current_pin_pos_2d(context, cam_data, pin)
                if p2d is None or not pin.has_valid_3d:
                    pin.reproj_error = -1.0
                    continue
                
                obj_pts.append(list(pin.pos_3d))
                px, py = to_cv_pixel(p2d.x, p2d.y, res_x, res_y)
                img_pts.append([px, py])
                valid_pins_eval.append(pin)
                
            if obj_pts:
                proj_pts, _ = cv2.projectPoints(np.array(obj_pts, dtype=np.float32), rvec, tvec, camintr, distcoef)
                for i, pin in enumerate(valid_pins_eval):
                    err_x = proj_pts[i][0][0] - img_pts[i][0]
                    err_y = proj_pts[i][0][1] - img_pts[i][1]
                    err = float(np.sqrt(err_x**2 + err_y**2))
                    pin.reproj_error = err
                    total_error += err
                    valid_count += 1
                    
        target_data.mm_avg_error = (total_error / valid_count) if valid_count > 0 else -1.0

def schedule_error_update():
    def delayed_update():
        ctx = bpy.context
        cam_data, target_data, _ = get_active_target_data(ctx)
        if cam_data and target_data:
            update_reproj_errors(ctx, cam_data, target_data, force_update=True)
            redraw_all_3d_views(ctx)
        return None
    bpy.app.timers.register(delayed_update, first_interval=0.05)

def safe_ray_cast(context: bpy.types.Context, origin: Vector, direction: Vector, snap_to_vertex: bool = False) -> Tuple[bool, Optional[Vector]]:
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    cur_origin = origin + direction * PinSolverConfig.RAYCAST_START_OFFSET 
    
    for _ in range(PinSolverConfig.RAYCAST_MAX_RETRIES):
        hit, loc, normal, index, obj, matrix = scene.ray_cast(depsgraph, cur_origin, direction)
        if not hit: return False, None
        if obj and obj.type in {'CAMERA', 'LIGHT', 'SPEAKER'}:
            cur_origin = loc + direction * 0.01
            continue
            
        if snap_to_vertex and obj and obj.type == 'MESH':
            try:
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.data
                if mesh and hasattr(mesh, "polygons") and index < len(mesh.polygons):
                    face = mesh.polygons[index]
                    closest_v_loc = None
                    min_dist = float('inf')
                    for v_idx in face.vertices:
                        v_world = matrix @ mesh.vertices[v_idx].co
                        dist = (v_world - loc).length
                        if dist < min_dist:
                            min_dist = dist
                            closest_v_loc = v_world
                    if closest_v_loc:
                        loc = closest_v_loc
            except Exception:
                pass
                
        return True, loc
    return False, None

def get_track_objects(self, context):
    items = []
    if self.target_clip:
        for i, ob in enumerate(self.target_clip.tracking.objects):
            items.append((str(i), ob.name, ""))
    if not items:
        items.append(("0", "Camera", ""))
    return items

# ==========================================
# 1. Data Structures & Settings
# ==========================================
class PinSolverSettings(PropertyGroup):
    pin_radius: IntProperty(name="Pin Radius", default=6, min=1, max=20, description="Radius of the drawn pins (2D and 3D) in pixels")
    text_size: IntProperty(name="Text Size", default=14, min=8, max=32, description="Font size for the pin name labels")
    line_width: FloatProperty(name="Line Width", default=1.5, min=0.1, max=10.0, description="Thickness of the line connecting 2D and 3D pins")
    line_opacity: FloatProperty(name="Line Opacity", default=0.7, min=0.0, max=1.0, description="Opacity of the line connecting 2D and 3D pins")
    overlay_opacity: FloatProperty(name="Overlay Opacity", default=1.0, min=0.0, max=1.0, description="Overall opacity of the on-screen overlays (pins, lines, texts)")
    add_pin_offset: FloatProperty(name="Add Pin Offset", default=PinSolverConfig.DEFAULT_ADD_PIN_OFFSET, min=0.1, max=100.0, description="Distance in meters from the camera when a new pin is manually added")
    show_name_solve: BoolProperty(name="Show Pin Names (Solve)", default=True, description="Display pin names in normal solve mode")
    show_name_tweak: BoolProperty(name="Show Pin Names (Tweak)", default=False, description="Display pin names during interactive tweak mode")
    show_error_3d: BoolProperty(name="Show Error on 3D Pin", default=False, description="Display the reprojection error value next to the 3D pin")
    show_weight_3d: BoolProperty(name="Show Weight on 3D Pin", default=False, description="Display the weight value next to the 3D pin")
    
    text_use_custom_color: BoolProperty(name="Custom Text Color", default=False, description="Use a uniform color for all text labels instead of matching the pin color")
    text_color: FloatVectorProperty(name="Text Color", subtype='COLOR', size=4, default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, description="Custom color for the pin labels")
    text_use_outline: BoolProperty(name="Text Outline", default=False, description="Draw an outline around text labels for better visibility")
    text_outline_color: FloatVectorProperty(name="Outline Color", subtype='COLOR', size=4, default=(0.0, 0.0, 0.0, 1.0), min=0.0, max=1.0, description="Color of the text outline")
    
    lock_camera_z: BoolProperty(name="Lock Camera Z (1-Pin)", default=True, description="1-Pin Solve: Keep the camera's absolute Z height unchanged")

class PinSolverPin(PropertyGroup):
    name: StringProperty(name="Name", default="Pin", description="Custom name for this pin")
    pos_3d: FloatVectorProperty(name="3D", size=3, default=(0.0, 0.0, 0.0), description="3D World Coordinate of the pin")
    has_valid_3d: BoolProperty(default=True, description="Indicates if this pin has been properly raycasted or placed in 3D space")
    pos_2d: FloatVectorProperty(name="2D", size=2, default=(0.5, 0.5), description="2D Viewport/Camera Coordinate (U,V) of the pin")
    color: FloatVectorProperty(name="Color", subtype='COLOR', size=4, default=(1.0, 0.2, 0.2, 1.0), min=0.0, max=1.0, description="Color code of the pin")
    use_initial: BoolProperty(name="Solve", default=True, description="Toggle pin participation and visibility")
    use_tweak: BoolProperty(name="Tweak", default=True, description="Include this pin in the Interactive Tweak calculation")
    weight: FloatProperty(name="Weight", default=0.0, min=0.0, max=1.0, subtype='FACTOR', description="Binding strength. 0.0 = Normal, 1.0 = Max Constraint")
    reproj_error: FloatProperty(name="Error", default=0.0, description="Current reprojection error distance in pixels")
    track_name: StringProperty(name="Track Name", default="")
    is_track_linked: BoolProperty(default=False)

class PinSolverTargetItem(PropertyGroup):
    obj: PointerProperty(type=bpy.types.Object, name="Object", description="Select the object to be aligned")

# ==========================================
# Draw Handler Management
# ==========================================
_draw_handle = None

def update_show_overlays(self, context):
    global _draw_handle
    if self.show_overlays:
        if _draw_handle is None:
            _draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_overlay, (), 'WINDOW', 'POST_PIXEL')
    else:
        if _draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
            _draw_handle = None

class PinSolverData(PropertyGroup):
    picking_state: EnumProperty(items=[('NONE', "", ""), ('PICK_2D', "", ""), ('PICK_3D', "", "")], default='NONE')
    picking_index: IntProperty(default=-1)
    last_error: StringProperty(default="")
    
    layout_avg_error: FloatProperty(name="Layout Error", default=-1.0)
    mm_avg_error: FloatProperty(name="Sequence Error", default=-1.0)
    
    show_overlays: BoolProperty(
        name="Show Pins", 
        default=False, 
        options={'SKIP_SAVE'}, 
        update=update_show_overlays, 
        description="Toggle the visibility of all pins and lines in the 3D Viewport"
    )
    use_distortion_overlay: BoolProperty(name="Undistort 2D Pins", default=True, description="Un-distorts 2D pins in the viewport to match the linear 3D pins perfectly")
    is_tweak_mode: BoolProperty(default=False)
    is_edit_mode: BoolProperty(default=False)
    
    use_planar_solve: BoolProperty(name="Planar Mode", default=False, description="Use IPPE algorithm for perfectly flat surfaces (requires 4+ pins)")
    
    ui_mode: EnumProperty(
        name="Workflow Mode",
        items=[('LAYOUT', "Layout", "Single frame static alignment"), 
               ('MATCHMOVE', "Matchmove", "Animate target based on tracker data")],
        default='LAYOUT'
    )
    
    layout_pins: CollectionProperty(type=PinSolverPin)
    layout_pin_idx: IntProperty(default=0, description="Active Layout Pin - Select a pin to view or edit its coordinates")
    mm_pins: CollectionProperty(type=PinSolverPin)
    mm_pin_idx: IntProperty(default=0, description="Active Matchmove Track - Select a tracked pin to view or edit its coordinates")
    
    reference_frame: IntProperty(name="Reference Frame", default=1, description="The frame where 3D Pins are perfectly aligned to the scene")
    bake_target: EnumProperty(
        name="Bake Range",
        items=[('SCENE', "Scene Range", "Bake using the scene's start and end frames"), 
               ('MARKERS', "Timeline Markers", "Bake only on frames that contain timeline markers")],
        default='SCENE'
    )
    sequence_motion_source: EnumProperty(
        name="Location Source",
        items=[
            ('PNP', "Solve Location", "Use PnP-derived camera location; optional filtering can repair and smooth the location path"),
            ('EXISTING', "Existing Location", "Keep the existing animated location and run a rotation-only solve from pins"),
            ('GUIDED', "Guided Location", "Use the existing animated camera location as a guide while still solving location from pins"),
            ('MARKER_REFS', "Timeline Markers", "Use timeline marker camera poses as fixed references while solving the scene range")
        ],
        default='PNP'
    )
    sequence_stabilize_mode: EnumProperty(
        name="Location Filter",
        items=[
            ('OFF', "Off", "Use direct PnP location and rotation without curve repair"),
            ('AUTO', "Smooth", "Use guided PnP, repair location outliers, smooth location, then refit rotation"),
            ('STRONG', "Hyper Smooth", "Use stronger location repair and smoothing for difficult tracker layouts")
        ],
        default='OFF'
    )
    sequence_location_filter_strength: FloatProperty(
        name="Location Strength",
        default=1.0,
        min=0.0,
        max=2.0,
        soft_min=0.0,
        soft_max=2.0,
        subtype='FACTOR',
        description="Adjust the strength of the selected location filter mode"
    )
    sequence_guided_location_strength: FloatProperty(
        name="Guide Strength",
        default=1.0,
        min=0.0,
        max=2.0,
        soft_min=0.0,
        soft_max=2.0,
        subtype='FACTOR',
        description="Adjust how strongly Guided Location follows the existing camera location curve"
    )
    sequence_roll_smoothing: EnumProperty(
        name="Roll Smoothing",
        items=[
            ('OFF', "Off", "Keep solved roll as-is"),
            ('SMOOTH', "Smooth", "Smooth camera roll while keeping the solved viewing direction"),
            ('HYPER', "Hyper Smooth", "Apply stronger camera roll smoothing for one-sided tracker layouts")
        ],
        default='OFF'
    )
    sequence_roll_smoothing_strength: FloatProperty(
        name="Roll Strength",
        default=1.0,
        min=0.0,
        max=2.0,
        soft_min=0.0,
        soft_max=2.0,
        subtype='FACTOR',
        description="Adjust the strength of the selected roll smoothing mode"
    )
    
    solve_mode: EnumProperty(
        name="Solve Target",
        items=[
            ('CAMERA', "Camera", "Align the Camera itself"), 
            ('PARENT', "Parent", "Align the Parent object of the Camera"), 
            ('OBJECT', "Object", "Align a selected target Object to the Camera")
        ],
        description="Select which object will be transformed by the solver"
    )
    target_objects: CollectionProperty(type=PinSolverTargetItem)
    active_target_index: IntProperty(default=0)
    
    show_pin_details: BoolProperty(name="Manual Coordinates", default=False, description="Expand to manually edit 2D and 3D coordinates") 
    show_sequence_options: BoolProperty(name="Options", default=True, description="Expand sequence solver options")
    show_calibration: BoolProperty(name="Lens Calibration", default=False, description="Expand Lens Calibration settings") 
    show_settings: BoolProperty(name="PinSolver Settings", default=False, description="Expand to change appearance and behaviors") 
    
    pin_disp_focal: BoolProperty(name="Pin Focal Length", default=False, description="Keep this setting visible even when the panel is collapsed")
    pin_disp_center: BoolProperty(name="Pin Optical Center", default=False, description="Keep this setting visible even when the panel is collapsed")
    pin_disp_dist: BoolProperty(name="Pin Distortion", default=False, description="Keep this setting visible even when the panel is collapsed")
    
    target_clip: PointerProperty(type=bpy.types.MovieClip, name="Target Clip", description="Select a Movie Clip to sync with trackers and lens parameters")
    tracking_object_idx: EnumProperty(name="Track Layer", items=get_track_objects, description="Select the tracking object layer from the Movie Clip")
    
    calib_focal_length: BoolProperty(name="Focal Length", default=False, description="Allow the solver to estimate and modify the Focal Length")
    calib_optical_center: BoolProperty(name="Optical Center", default=False, description="Allow the solver to estimate and modify the Optical Center (Camera Shift)")
    calib_k1: BoolProperty(name="K1", default=False, update=update_distortion_model, description="Allow the solver to estimate K1 (Radial Distortion)")
    calib_k2: BoolProperty(name="K2", default=False, update=update_distortion_model, description="Allow the solver to estimate K2 (Radial Distortion)")
    calib_k3: BoolProperty(name="K3", default=False, update=update_distortion_model, description="Allow the solver to estimate K3 (Radial Distortion)")
    dummy_k: FloatProperty(default=0.0)
    
    calib_animation_mode: EnumProperty(
        name="Lens Animation",
        items=[('STATIC', "Static Lens", "Calculate single optimal lens for all frames without keyframing"), 
               ('ZOOM', "Zooming Lens", "Lock Base Params and calculate dynamic Focal Length per frame")],
        default='STATIC'
    )
    calib_static_method: EnumProperty(
        name="Base Params",
        items=[('AVERAGE', "Average All Frames", "Calculate average intrinsics across bake range"), 
               ('MEDIAN', "Median All Frames", "Calculate median intrinsics across bake range")],
        default='MEDIAN'
    )
    use_dynamic_zoom: BoolProperty(name="Zooming", default=False, description="Keyframe this to toggle between Static lens and Dynamic Zoom per frame")

# ==========================================
# 2. Core Solver Logic
# ==========================================
def _solve_single_pin(context: bpy.types.Context, cam_mat_unscaled: Matrix, pin: Any, p2d: Vector, region: Any, rv3d: Any) -> Tuple[bool, Optional[Matrix], str]:
    C_old = cam_mat_unscaled.translation
    P = Vector(pin.pos_3d)

    cam_data = context.scene.camera.pinsolver_data
    camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
    settings = context.scene.pinsolver_settings

    if cam_data.ui_mode == 'LAYOUT':
        F_vec = cam_mat_unscaled.to_3x3() @ Vector((0, 0, -1))
        bounds = get_camera_frame_bounds(context, region, rv3d)
        coord_2d_pin = _get_undistorted_2d_coords_cached(p2d, bounds, camintr, distcoef, res_x, res_y)
        if not coord_2d_pin: return False, None, "Undistort failed"

        origin = region_2d_to_origin_3d(region, rv3d, (coord_2d_pin.x, coord_2d_pin.y))
        direction = region_2d_to_vector_3d(region, rv3d, (coord_2d_pin.x, coord_2d_pin.y))
        if not origin or not direction: return False, None, "Ray generation failed"
        
    else:
        px, py = to_cv_pixel(p2d.x, p2d.y, res_x, res_y)
        has_distortion = np.linalg.norm(distcoef) > 1e-8
        if HAS_OPENCV and has_distortion:
            pt_2d = np.array([[[px, py]]], dtype=np.float64)
            undist = cv2.undistortPoints(pt_2d, camintr, distcoef, P=camintr)
            px_u, py_u = undist[0][0][0], undist[0][0][1]
        else:
            px_u, py_u = px, py

        fx, fy = camintr[0,0], camintr[1,1]
        cx, cy = camintr[0,2], camintr[1,2]
        
        X_c = (px_u - cx) / fx
        Y_c = (py_u - cy) / fy
        Z_c = 1.0
        D_c = Vector((X_c, Y_c, Z_c)).normalized()
        
        R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        R_world2bcam = cam_mat_unscaled.to_3x3().transposed()
        R_world2cv = R_bcam2cv @ R_world2bcam
        R_cv2world = R_world2cv.transposed()
        
        direction = R_cv2world @ D_c
        F_vec = R_cv2world @ Vector((0, 0, 1))
        origin = C_old

    if settings.lock_camera_z and not cam_data.is_tweak_mode and abs(direction.z) > 1e-6:
        t = (P.z - C_old.z) / direction.z
        if t <= 0.0: t = abs(t)
        if t < 1e-4: t = 1.0
        C_new = P - t * direction
        delta_C = C_new - C_old
    else:
        denom = direction.dot(F_vec)
        if abs(denom) < 1e-6: return False, None, "Invalid angle"
        t = (P - origin).dot(F_vec) / denom
        
        if t <= 0.0: t = abs(t)
        if t < 1e-4: t = 1.0
        
        I = origin + t * direction
        delta_C = P - I
    
    relative_limit = max(100.0, (P - C_old).length * 100.0)
    if delta_C.length > relative_limit: return False, None, "Distance too large"
    
    new_mat = cam_mat_unscaled.copy()
    new_mat.translation += delta_C
    return True, new_mat, ""

def _prep_multi_pin_data(valid_pins: List[Any], valid_p2ds: List[Vector], res_x: float, res_y: float, cam_mat_unscaled: Matrix, camintr: np.ndarray, distcoef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R_world2bcam = np.array(cam_mat_unscaled.to_3x3().transposed(), dtype=np.float64)
    R_world2cv = R_bcam2cv @ R_world2bcam
    rvec_guess, _ = cv2.Rodrigues(R_world2cv)
    tvec_guess = -R_world2cv @ np.array(cam_mat_unscaled.translation, dtype=np.float64)
    
    rvec_guess = np.array(rvec_guess, dtype=np.float64).reshape(3, 1)
    tvec_guess = np.array(tvec_guess, dtype=np.float64).reshape(3, 1)

    solve_obj_pts = []
    solve_img_pts = []
    
    for p, p2d in zip(valid_pins, valid_p2ds):
        pixel_x, pixel_y = to_cv_pixel(p2d.x, p2d.y, res_x, res_y)
        w_int = 1 + int(p.weight * 99)
            
        for _ in range(w_int):
            solve_obj_pts.append(list(p.pos_3d))
            solve_img_pts.append([pixel_x, pixel_y])

    if len(valid_pins) == 2:
        cam_right = cam_mat_unscaled.to_3x3() @ Vector((1, 0, 0))
        cam_up = cam_mat_unscaled.to_3x3() @ Vector((0, 1, 0))
        P1 = Vector(valid_pins[0].pos_3d)
        dist = max(0.1, (P1 - cam_mat_unscaled.translation).length * 0.2)
        P3_3d = P1 + cam_right * dist
        P4_3d = P1 + cam_up * dist
        
        virt_3d = np.array([list(P3_3d), list(P4_3d)], dtype=np.float64)
        virt_2d, _ = cv2.projectPoints(virt_3d, rvec_guess, tvec_guess, camintr, distcoef)
        solve_obj_pts.extend([list(P3_3d), list(P4_3d)])
        solve_img_pts.extend([list(virt_2d[0][0]), list(virt_2d[1][0])])

    obj_pts = np.ascontiguousarray(solve_obj_pts, dtype=np.float32)
    img_pts = np.ascontiguousarray(solve_img_pts, dtype=np.float32)
    
    return obj_pts, img_pts, rvec_guess, tvec_guess

def _calibrate_lens(cam_data: Any, valid_p2ds: List[Vector], valid_pins: List[Any], res_x: float, res_y: float, camintr: np.ndarray, distcoef: np.ndarray, target_mode: str, camera_obj: bpy.types.Object, apply_to_blender: bool = True, zoom_base_intrinsics: np.ndarray = None, zoom_base_distcoef: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str]:
    if len(valid_pins) < PinSolverConfig.MIN_PINS_FOR_CALIBRATION:
        return camintr, distcoef, None, None, False, f"Calib Skipped: Need {PinSolverConfig.MIN_PINS_FOR_CALIBRATION}+ points"

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO
    
    if zoom_base_intrinsics is not None and zoom_base_distcoef is not None:
        camintr = zoom_base_intrinsics.copy()
        distcoef_guess = zoom_base_distcoef.copy()
        if not cam_data.calib_focal_length: flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        if not cam_data.calib_optical_center: flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    else:
        if not cam_data.calib_focal_length: flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        if not cam_data.calib_optical_center: flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if not cam_data.calib_k1: flags |= cv2.CALIB_FIX_K1
        if not cam_data.calib_k2: flags |= cv2.CALIB_FIX_K2
        if not cam_data.calib_k3: flags |= cv2.CALIB_FIX_K3
        distcoef_guess = distcoef.copy() if target_mode == 'tweak' else np.zeros(5, dtype=np.float64)

    calib_obj = np.array([list(p.pos_3d) for p in valid_pins], dtype=np.float32)
    calib_img = np.array([[p2d.x * res_x, (1.0 - p2d.y) * res_y] for p2d in valid_p2ds], dtype=np.float32)
    
    try:
        ret, mtx, dist_calib, rvecs, tvecs = cv2.calibrateCamera(
            [calib_obj], [calib_img], (int(res_x), int(res_y)), camintr.copy(), distcoef_guess, flags=flags)
        
        camintr_new = mtx
        distcoef_new = dist_calib.flatten()
        rvec_calib = np.array(rvecs[0], dtype=np.float64).reshape(3, 1)
        tvec_calib = np.array(tvecs[0], dtype=np.float64).reshape(3, 1)
        
        if apply_to_blender:
            cam_data_ref = camera_obj.data
            if cam_data.target_clip:
                trk_cam = cam_data.target_clip.tracking.camera
                if cam_data.calib_focal_length or zoom_base_intrinsics is not None: 
                    trk_cam.focal_length_pixels = float(camintr_new[0, 0])
                if cam_data.calib_optical_center:
                    cx_cv = float(camintr_new[0, 2])
                    cy_cv = float(camintr_new[1, 2])
                    px = res_x / 2.0 - cx_cv
                    py = res_y / 2.0 - cy_cv
                    if bpy.app.version < (3, 5, 0): trk_cam.principal = [px, py]
                    else: trk_cam.principal_point_pixels = [px, py]
                if cam_data.calib_k1 and zoom_base_intrinsics is None: trk_cam.k1 = float(distcoef_new[0])
                if cam_data.calib_k2 and zoom_base_intrinsics is None: trk_cam.k2 = float(distcoef_new[1])
                if cam_data.calib_k3 and zoom_base_intrinsics is None: trk_cam.k3 = float(distcoef_new[4])
                if cam_data.calib_focal_length or zoom_base_intrinsics is not None: 
                    cam_data_ref.lens = cam_data.target_clip.tracking.camera.focal_length
            else:
                max_res = max(res_x, res_y)
                sw = cam_data_ref.sensor_height * (res_x / res_y) if cam_data_ref.sensor_fit == 'VERTICAL' else cam_data_ref.sensor_width
                if cam_data.calib_focal_length or zoom_base_intrinsics is not None:
                    cam_data_ref.lens = float(camintr_new[0, 0]) * max(1e-4, sw) / res_x
                if cam_data.calib_optical_center:
                    cx = float(camintr_new[0, 2])
                    cy = float(camintr_new[1, 2])
                    cam_data_ref.shift_x = (cx - res_x / 2.0) / max_res
                    cam_data_ref.shift_y = (res_y / 2.0 - cy) / max_res
                
        return camintr_new, distcoef_new, rvec_calib, tvec_calib, True, ""
        
    except cv2.error as e:
        return camintr, distcoef, None, None, False, f"Calib Error: {str(e).splitlines()[0][:40]}"
    except Exception as e:
        traceback.print_exc()
        return camintr, distcoef, None, None, False, f"Unexpected Calib Error: {type(e).__name__}"

def _estimate_pose(obj_pts: np.ndarray, img_pts: np.ndarray, camintr: np.ndarray, distcoef: np.ndarray, rvec_guess: np.ndarray, tvec_guess: np.ndarray, use_guess: bool, fallback_rvec: np.ndarray = None, fallback_tvec: np.ndarray = None, use_planar: bool = False, allow_global_fallback: bool = True) -> Tuple[bool, Optional[Matrix], str]:
    pnp_success = False
    rvec_out = np.copy(rvec_guess)
    tvec_out = np.copy(tvec_guess)
    error_msg = ""
    
    if use_guess:
        try:
            pnp_success, rvec_out, tvec_out = cv2.solvePnP(
                obj_pts, img_pts, camintr, distcoef, 
                rvec=rvec_out, tvec=tvec_out, 
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if pnp_success and hasattr(cv2, 'solvePnPRefineLM'):
                rvec_out, tvec_out = cv2.solvePnPRefineLM(obj_pts, img_pts, camintr, distcoef, rvec_out, tvec_out)
        except cv2.error as e:
            error_msg = f"Guided PnP Error: {str(e).splitlines()[0][:40]}"
            pnp_success = False
        except Exception as e:
            error_msg = f"Unexpected Guided Error: {type(e).__name__}"
            pnp_success = False

    if not pnp_success and allow_global_fallback:
        if use_planar and hasattr(cv2, 'SOLVEPNP_IPPE') and len(obj_pts) >= 4:
            try:
                pnp_success, rvec_out, tvec_out = cv2.solvePnP(
                    obj_pts, img_pts, camintr, distcoef, flags=cv2.SOLVEPNP_IPPE)
            except cv2.error as e:
                error_msg = f"Planar PnP Error: {str(e).splitlines()[0][:40]}"
            except Exception as e:
                error_msg = f"Unexpected Planar Error: {type(e).__name__}"
                
        if not use_guess or not pnp_success:
            try:
                pnp_success, rvec_out, tvec_out = cv2.solvePnP(
                    obj_pts, img_pts, camintr, distcoef, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error as e:
                error_msg = f"PnP Error: {str(e).splitlines()[0][:40]}"
            except Exception as e:
                error_msg = f"Unexpected PnP Error: {type(e).__name__}"

    if not pnp_success:
        if use_planar and len(obj_pts) < 4:
            error_msg = "Planar Mode requires 4+ pins"
            
        if fallback_rvec is not None and fallback_tvec is not None:
            rvec_out = fallback_rvec
            tvec_out = fallback_tvec
        else:
            return False, None, error_msg or "PnP Failed"

    rvec_out = np.asarray(rvec_out, dtype=np.float64).reshape(3, 1)
    tvec_out = np.asarray(tvec_out, dtype=np.float64).reshape(3, 1)

    R_inv = cv2.Rodrigues(rvec_out)[0].T
    cam_pos = (-np.dot(R_inv, tvec_out)).flatten()
    
    if not np.isfinite(cam_pos).all() or np.linalg.norm(cam_pos) > 1e6: 
        return False, None, "Invalid Matrix Result"
        
    R_blender = np.dot(R_inv, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64))
    mat_loc = Matrix.Translation(Vector((cam_pos[0], cam_pos[1], cam_pos[2])))
    return True, mat_loc @ Matrix(R_blender).to_4x4(), ""

def get_camera_unscaled_matrix(camera_obj: bpy.types.Object, depsgraph: Any = None) -> Matrix:
    eval_cam = camera_obj.evaluated_get(depsgraph) if depsgraph else camera_obj
    cam_loc, cam_rot, _ = eval_cam.matrix_world.decompose()
    return Matrix.LocRotScale(cam_loc, cam_rot, Vector((1.0, 1.0, 1.0)))

def interpolate_pose_matrices(a: Matrix, b: Matrix, factor: float) -> Matrix:
    factor = max(0.0, min(1.0, float(factor)))
    loc_a, rot_a, scale_a = a.decompose()
    loc_b, rot_b, _ = b.decompose()
    return Matrix.LocRotScale(loc_a.lerp(loc_b, factor), rot_a.slerp(rot_b, factor), scale_a)

def extrapolate_location_pose(previous_pose: Matrix, previous_frame: int, current_pose: Matrix, current_frame: int, target_frame: int, older_pose: Matrix = None, older_frame: int = None, use_acceleration: bool = False) -> Matrix:
    if previous_pose is None or current_pose is None or previous_frame == current_frame:
        return current_pose.copy() if current_pose else None
        
    prev_loc, _, _ = previous_pose.decompose()
    curr_loc, curr_rot, curr_scale = current_pose.decompose()
    factor = (target_frame - current_frame) / max(1.0, abs(current_frame - previous_frame))
    current_step = curr_loc - prev_loc
    predicted_loc = curr_loc + current_step * factor
    
    if use_acceleration and older_pose is not None and older_frame is not None and previous_frame != older_frame:
        older_loc, _, _ = older_pose.decompose()
        previous_step = prev_loc - older_loc
        accel_adjust = (current_step - previous_step) * factor * 0.5
        max_adjust = max(0.0, current_step.length * 0.75)
        if accel_adjust.length > max_adjust > 1e-8:
            accel_adjust = accel_adjust.normalized() * max_adjust
        predicted_loc += accel_adjust
        
    return Matrix.LocRotScale(predicted_loc, curr_rot, curr_scale)

def get_pose_delta(a: Matrix, b: Matrix) -> Tuple[float, float]:
    loc_a, rot_a, _ = a.decompose()
    loc_b, rot_b, _ = b.decompose()
    return (loc_b - loc_a).length, rot_a.rotation_difference(rot_b).angle

def get_roll_basis(forward: Vector) -> Tuple[Vector, Vector]:
    fwd = forward.normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(fwd.dot(world_up)) > 0.98:
        world_up = Vector((0.0, 1.0, 0.0))
    right = fwd.cross(world_up).normalized()
    up = right.cross(fwd).normalized()
    return right, up

def pose_roll_angle(pose: Matrix) -> Tuple[float, Vector, Vector, Vector]:
    rot_mat = pose.to_3x3()
    forward = (rot_mat @ Vector((0.0, 0.0, -1.0))).normalized()
    up = (rot_mat @ Vector((0.0, 1.0, 0.0))).normalized()
    base_right, base_up = get_roll_basis(forward)
    angle = float(np.arctan2(up.dot(base_right), up.dot(base_up)))
    return angle, forward, base_right, base_up

def matrix_with_roll(pose: Matrix, roll_angle: float) -> Matrix:
    loc, _, scale = pose.decompose()
    _, forward, base_right, base_up = pose_roll_angle(pose)
    up = (math.cos(roll_angle) * base_up + math.sin(roll_angle) * base_right).normalized()
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    back = -forward
    rot = Matrix((
        (right.x, up.x, back.x),
        (right.y, up.y, back.y),
        (right.z, up.z, back.z),
    ))
    return Matrix.LocRotScale(loc, rot.to_quaternion(), scale)

def repair_roll_curve(unwrapped: np.ndarray, mode: str, reference_index: int, strength_factor: float) -> np.ndarray:
    if len(unwrapped) < 5:
        return np.array(unwrapped, dtype=np.float64, copy=True)
        
    strength_factor = max(0.0, min(2.0, float(strength_factor)))
    repaired = np.array(unwrapped, dtype=np.float64, copy=True)
    radius = 4 if mode == 'HYPER' else 3
    min_threshold = math.radians(6.0 if mode == 'HYPER' else 10.0)
    strength = min(0.95, (0.75 if mode == 'HYPER' else 0.55) * strength_factor)
    
    for _ in range(2):
        previous = repaired.copy()
        for i in range(len(previous)):
            if i == reference_index:
                continue
                
            lo = max(0, i - radius)
            hi = min(len(previous), i + radius + 1)
            samples = [previous[j] for j in range(lo, hi) if j != i]
            if len(samples) < 2:
                continue
                
            sample_arr = np.array(samples, dtype=np.float64)
            median = float(np.median(sample_arr))
            mad = float(np.median(np.abs(sample_arr - median)))
            threshold = max(min_threshold, mad * 1.4826 * 3.0)
            residual = previous[i] - median
            if abs(residual) > threshold:
                repaired[i] = previous[i] * (1.0 - strength) + median * strength
                
    return repaired

def stabilize_roll_pose_path(solved_pose_by_frame: Dict[int, Matrix], mode: str, reference_frame: int, strength_factor: float = 1.0, fixed_frames: Optional[Set[int]] = None) -> Dict[int, Matrix]:
    if mode == 'OFF' or len(solved_pose_by_frame) < 3:
        return {}
        
    fixed_frames = fixed_frames or {reference_frame}
    strength_factor = max(0.0, min(2.0, float(strength_factor)))
    frames = sorted(solved_pose_by_frame.keys())
    rolls = []
    for f in frames:
        angle, _, _, _ = pose_roll_angle(solved_pose_by_frame[f])
        rolls.append(angle)
        
    unwrapped = np.unwrap(np.array(rolls, dtype=np.float64))
    reference_index = frames.index(reference_frame) if reference_frame in frames else -1
    repaired = repair_roll_curve(unwrapped, mode, reference_index, strength_factor)
    if mode == 'HYPER':
        radius = 6
        blend = min(0.98, 0.85 * strength_factor)
    else:
        radius = 3
        blend = min(0.98, 0.55 * strength_factor)
        
    smoothed = []
    for i in range(len(frames)):
        weighted = 0.0
        weight_sum = 0.0
        for j in range(max(0, i - radius), min(len(frames), i + radius + 1)):
            dist = abs(j - i)
            weight = (radius + 1 - dist) / (radius + 1)
            weighted += repaired[j] * weight
            weight_sum += weight
        smoothed.append(weighted / max(weight_sum, 1e-8))
        
    if reference_index >= 0:
        ref_delta = smoothed[reference_index] - unwrapped[reference_index]
        smoothed = [roll - ref_delta for roll in smoothed]
        
    result = {}
    for f, raw_roll, repaired_roll, smooth_roll in zip(frames, unwrapped, repaired, smoothed):
        if f in fixed_frames:
            continue
        final_roll = repaired_roll * (1.0 - blend) + smooth_roll * blend
        result[f] = matrix_with_roll(solved_pose_by_frame[f], final_roll)
    return result

def stabilize_location_pose_path(solved_pose_by_frame: Dict[int, Matrix], condition_by_frame: Dict[int, float], pin_count_by_frame: Dict[int, int], mode: str, strength_factor: float = 1.0, hard_frames: Optional[Set[int]] = None, soft_anchor_pose_by_frame: Optional[Dict[int, Matrix]] = None) -> Dict[int, Matrix]:
    if mode == 'OFF' or len(solved_pose_by_frame) < 3:
        return {}
        
    strength_factor = max(0.0, min(2.0, float(strength_factor)))
    if strength_factor <= 1e-6:
        return {}
        
    hard_frames = hard_frames or set()
    soft_anchor_pose_by_frame = soft_anchor_pose_by_frame or {}
    frames = sorted(solved_pose_by_frame.keys())
    loc_by_frame = {}
    rot_by_frame = {}
    scale_by_frame = {}
    for f in frames:
        loc, rot, scale = solved_pose_by_frame[f].decompose()
        loc_by_frame[f] = loc
        rot_by_frame[f] = rot
        scale_by_frame[f] = scale
    soft_anchor_loc_by_frame = {
        f: pose.decompose()[0]
        for f, pose in soft_anchor_pose_by_frame.items()
        if f in loc_by_frame and f not in hard_frames and pose is not None
    }
        
    step_lengths = []
    for a, b in zip(frames[:-1], frames[1:]):
        dt = max(1.0, float(b - a))
        step_lengths.append((loc_by_frame[b] - loc_by_frame[a]).length / dt)
    step_med = float(np.median(step_lengths)) if step_lengths else 0.0
    step_med = max(step_med, 1e-6)
    
    residuals = {}
    for i in range(1, len(frames) - 1):
        f = frames[i]
        prev_f = frames[i - 1]
        next_f = frames[i + 1]
        factor = (f - prev_f) / max(1.0, next_f - prev_f)
        expected = loc_by_frame[prev_f].lerp(loc_by_frame[next_f], factor)
        residuals[f] = (loc_by_frame[f] - expected).length
        
    if residuals:
        vals = np.array(list(residuals.values()), dtype=np.float64)
        resid_med = float(np.median(vals))
        resid_mad = float(np.median(np.abs(vals - resid_med)))
    else:
        resid_med = 0.0
        resid_mad = 0.0
        
    if mode == 'STRONG':
        resid_limit = max(step_med * 3.0, resid_med + resid_mad * 4.0)
        low_cond_limit = max(step_med * 1.8, resid_med + resid_mad * 2.5)
        jump_limit = step_med * 5.0
        radius = 5
        base_strength = 0.75
    else:
        resid_limit = max(step_med * 5.0, resid_med + resid_mad * 6.0)
        low_cond_limit = max(step_med * 3.0, resid_med + resid_mad * 4.0)
        jump_limit = step_med * 8.0
        radius = 3
        base_strength = 0.45
        
    outliers = set()
    for i in range(1, len(frames) - 1):
        f = frames[i]
        if f in hard_frames:
            continue
        prev_f = frames[i - 1]
        next_f = frames[i + 1]
        condition = condition_by_frame.get(f, 1.0)
        pin_count = pin_count_by_frame.get(f, 0)
        low_condition = condition >= 0.45 or pin_count < 5
        residual = residuals.get(f, 0.0)
        if residual > resid_limit or (low_condition and residual > low_cond_limit):
            outliers.add(f)
            continue
            
        prev_step = (loc_by_frame[f] - loc_by_frame[prev_f]).length / max(1.0, f - prev_f)
        next_step = (loc_by_frame[next_f] - loc_by_frame[f]).length / max(1.0, next_f - f)
        across_step = (loc_by_frame[next_f] - loc_by_frame[prev_f]).length / max(1.0, next_f - prev_f)
        if low_condition and prev_step > jump_limit and next_step > jump_limit and across_step < max(prev_step, next_step) * 0.35:
            outliers.add(f)
    
    good_frames = [f for f in frames if f not in outliers]
    repaired_loc = dict(loc_by_frame)
    for f in sorted(outliers):
        prev_good = [gf for gf in good_frames if gf < f]
        next_good = [gf for gf in good_frames if gf > f]
        if prev_good and next_good:
            a = max(prev_good)
            b = min(next_good)
            factor = (f - a) / max(1.0, b - a)
            repaired_loc[f] = repaired_loc[a].lerp(repaired_loc[b], factor)
        elif prev_good:
            repaired_loc[f] = repaired_loc[max(prev_good)].copy()
        elif next_good:
            repaired_loc[f] = repaired_loc[min(next_good)].copy()
    
    stabilized = {}
    for i, f in enumerate(frames):
        weighted = Vector((0.0, 0.0, 0.0))
        weight_sum = 0.0
        for j in range(max(0, i - radius), min(len(frames), i + radius + 1)):
            nf = frames[j]
            dist = abs(j - i)
            weight = (radius + 1 - dist) / (radius + 1)
            weighted += repaired_loc[nf] * weight
            weight_sum += weight
        if weight_sum <= 1e-8:
            continue
        avg_loc = weighted / weight_sum
        if f in hard_frames:
            final_loc = loc_by_frame[f]
        else:
            condition = condition_by_frame.get(f, 0.0)
            strength = min(0.98, (base_strength + condition * (0.2 if mode == 'STRONG' else 0.15)) * strength_factor)
            final_loc = repaired_loc[f].lerp(avg_loc, min(1.0, strength_factor)) if f in outliers else repaired_loc[f].lerp(avg_loc, strength)
            if f in soft_anchor_loc_by_frame:
                anchor_strength = min(0.90, (0.70 if mode == 'STRONG' else 0.55) * strength_factor)
                final_loc = final_loc.lerp(soft_anchor_loc_by_frame[f], anchor_strength)
        stabilized[f] = Matrix.LocRotScale(final_loc, rot_by_frame[f], scale_by_frame[f])
        
    return stabilized

def align_marker_reference_segments(solved_pose_by_frame: Dict[int, Matrix], marker_reference_pose_by_frame: Dict[int, Matrix], anchor_frames: Set[int]) -> Dict[int, Matrix]:
    anchors = sorted(f for f in anchor_frames if f in solved_pose_by_frame and f in marker_reference_pose_by_frame)
    if not anchors:
        return {}
        
    aligned = {}
    solved_frames = sorted(solved_pose_by_frame.keys())
    anchor_offsets = {}
    for f in anchors:
        solved_loc, _, _ = solved_pose_by_frame[f].decompose()
        marker_loc, _, _ = marker_reference_pose_by_frame[f].decompose()
        anchor_offsets[f] = marker_loc - solved_loc
        aligned[f] = marker_reference_pose_by_frame[f].copy()
        
    if len(anchors) < 2:
        return aligned
        
    for left_f, right_f in zip(anchors[:-1], anchors[1:]):
        span = max(1.0, float(right_f - left_f))
        for f in sorted(frame for frame in solved_pose_by_frame.keys() if left_f < frame < right_f):
            factor = (f - left_f) / span
            smooth_factor = factor * factor * (3.0 - 2.0 * factor)
            base_pose = solved_pose_by_frame[f]
            base_loc, base_rot, base_scale = base_pose.decompose()
            offset = anchor_offsets[left_f].lerp(anchor_offsets[right_f], smooth_factor)
            final_loc = base_loc + offset
            aligned[f] = Matrix.LocRotScale(final_loc, base_rot, base_scale)
            
    first_anchor = anchors[0]
    last_anchor = anchors[-1]
    for f in solved_frames:
        if f in aligned:
            continue
        base_pose = solved_pose_by_frame[f]
        base_loc, base_rot, base_scale = base_pose.decompose()
        if f < first_anchor:
            aligned[f] = Matrix.LocRotScale(base_loc + anchor_offsets[first_anchor], base_rot, base_scale)
        elif f > last_anchor:
            aligned[f] = Matrix.LocRotScale(base_loc + anchor_offsets[last_anchor], base_rot, base_scale)
    return aligned

def reinforce_fixed_reference_transitions(solved_pose_by_frame: Dict[int, Matrix], fixed_frames: Set[int], mode: str, strength_factor: float = 1.0) -> Dict[int, Matrix]:
    fixed_frames = {f for f in fixed_frames if f in solved_pose_by_frame}
    if len(solved_pose_by_frame) < 3 or not fixed_frames:
        return {}
        
    frames = sorted(solved_pose_by_frame.keys())
    frame_set = set(frames)
    fixed_sorted = sorted(fixed_frames)
    strength_factor = max(0.0, min(2.0, float(strength_factor)))
    radius = 12 if mode == 'STRONG' else 7
    blend = min(1.0, (0.96 if mode == 'STRONG' else 0.86) * strength_factor)
    if blend <= 1e-6:
        return {}
        
    reinforced = {}
    
    def hermite(p0: Vector, p1: Vector, m0: Vector, m1: Vector, t: float) -> Vector:
        t = max(0.0, min(1.0, float(t)))
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
    
    def collect_side(anchor_f: int, anchor_index: int, direction: int) -> List[int]:
        side_indices = []
        idx = anchor_index + direction
        while 0 <= idx < len(frames):
            f = frames[idx]
            if f in fixed_frames:
                break
            if abs(f - anchor_f) > radius:
                break
            side_indices.append(idx)
            idx += direction
        return side_indices
    
    def apply_side(anchor_f: int, side_indices: List[int], boundary_f: int, anchor_tangent_per_frame: Vector, boundary_tangent_scale: float):
        if not side_indices:
            return
        anchor_loc, _, _ = solved_pose_by_frame[anchor_f].decompose()
        boundary_loc, _, _ = solved_pose_by_frame[boundary_f].decompose()
        span = max(1.0, abs(boundary_f - anchor_f))
        direction = 1 if boundary_f > anchor_f else -1
        if direction > 0:
            p0 = anchor_loc
            p1 = boundary_loc
            m0 = anchor_tangent_per_frame * span
            m1 = (boundary_loc - anchor_loc) * boundary_tangent_scale
        else:
            p0 = boundary_loc
            p1 = anchor_loc
            m0 = (anchor_loc - boundary_loc) * boundary_tangent_scale
            m1 = anchor_tangent_per_frame * span
        for i in side_indices:
            f = frames[i]
            if f not in frame_set or f in fixed_frames:
                continue
            if direction > 0:
                factor = (f - anchor_f) / span
            else:
                factor = (f - boundary_f) / span
            target_loc = hermite(p0, p1, m0, m1, factor)
            base_loc, base_rot, base_scale = solved_pose_by_frame[f].decompose()
            distance_from_anchor = abs(f - anchor_f) / span
            distance_weight = 1.0 - min(1.0, distance_from_anchor)
            local_blend = min(1.0, blend * (0.82 + 0.18 * distance_weight))
            final_loc = target_loc if local_blend >= 0.999 else base_loc.lerp(target_loc, local_blend)
            reinforced[f] = Matrix.LocRotScale(final_loc, base_rot, base_scale)
    
    for anchor_f in fixed_sorted:
        anchor_index = frames.index(anchor_f)
        left_indices = collect_side(anchor_f, anchor_index, -1)
        right_indices = collect_side(anchor_f, anchor_index, 1)
        if not left_indices and not right_indices:
            continue
            
        left_boundary_f = frames[left_indices[-1]] if left_indices else None
        right_boundary_f = frames[right_indices[-1]] if right_indices else None
        anchor_loc, _, _ = solved_pose_by_frame[anchor_f].decompose()
        
        if left_boundary_f is not None and right_boundary_f is not None:
            left_loc, _, _ = solved_pose_by_frame[left_boundary_f].decompose()
            right_loc, _, _ = solved_pose_by_frame[right_boundary_f].decompose()
            anchor_tangent = (right_loc - left_loc) / max(1.0, float(right_boundary_f - left_boundary_f))
        elif left_boundary_f is not None:
            left_loc, _, _ = solved_pose_by_frame[left_boundary_f].decompose()
            anchor_tangent = (anchor_loc - left_loc) / max(1.0, float(anchor_f - left_boundary_f))
        else:
            right_loc, _, _ = solved_pose_by_frame[right_boundary_f].decompose()
            anchor_tangent = (right_loc - anchor_loc) / max(1.0, float(right_boundary_f - anchor_f))
        anchor_tangent *= 0.18 if mode == 'STRONG' else 0.28
        boundary_tangent_scale = 0.35 if mode == 'STRONG' else 0.50
        
        if left_boundary_f is not None:
            apply_side(anchor_f, left_indices, left_boundary_f, anchor_tangent, boundary_tangent_scale)
        if right_boundary_f is not None:
            apply_side(anchor_f, right_indices, right_boundary_f, anchor_tangent, boundary_tangent_scale)
            
    return reinforced

def smooth_guided_location_pose(pose: Matrix, guide_location: Vector, strength_factor: float) -> Matrix:
    if pose is None or guide_location is None:
        return pose
    strength_factor = max(0.0, min(2.0, float(strength_factor)))
    if strength_factor <= 1e-6:
        return pose
    loc, rot, scale = pose.decompose()
    blend = min(0.85, 0.42 * strength_factor)
    return Matrix.LocRotScale(loc.lerp(guide_location, blend), rot, scale)

def refine_rotation_for_fixed_location(context: bpy.types.Context, cam_data: Any, target_data: Any, base_pose: Matrix) -> Matrix:
    if base_pose is None:
        return base_pose
        
    pins, _ = get_pins(cam_data, target_data)
    world_dirs = []
    camera_rays = []
    weights = []
    camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
    loc, rot, scale = base_pose.decompose()
    
    for pin in pins:
        if not pin.use_initial or not pin.has_valid_3d:
            continue
        p2d = get_current_pin_pos_2d(context, cam_data, pin)
        if p2d is None:
            continue
            
        world_vec = Vector(pin.pos_3d) - loc
        if world_vec.length < 1e-6:
            continue
        world_dirs.append(np.array(world_vec.normalized(), dtype=np.float64))
        
        px, py = to_cv_pixel(p2d.x, p2d.y, res_x, res_y)
        has_distortion = HAS_OPENCV and np.linalg.norm(distcoef) > 1e-8
        if has_distortion:
            pt = np.array([[[px, py]]], dtype=np.float64)
            undist = cv2.undistortPoints(pt, camintr, distcoef)
            x = float(undist[0][0][0])
            y = float(undist[0][0][1])
        else:
            x = (px - camintr[0, 2]) / max(1e-8, camintr[0, 0])
            y = (py - camintr[1, 2]) / max(1e-8, camintr[1, 1])
        ray = Vector((x, y, 1.0)).normalized()
        camera_rays.append(np.array(ray, dtype=np.float64))
        weights.append(1.0 + float(getattr(pin, "weight", 0.0)) * 4.0)
        
    if len(world_dirs) < 2:
        return base_pose
        
    cam_mat_unscaled = Matrix.LocRotScale(loc, rot, Vector((1.0, 1.0, 1.0)))
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R_base_world2bcam = np.array(cam_mat_unscaled.to_3x3().transposed(), dtype=np.float64)
    R_base_world2cv = R_bcam2cv @ R_base_world2bcam
    
    H = np.zeros((3, 3), dtype=np.float64)
    for src_world, dst_ray, weight in zip(world_dirs, camera_rays, weights):
        H += weight * np.outer(dst_ray, src_world)
        
    try:
        U, _, Vt = np.linalg.svd(H)
        R_world2cv = U @ Vt
        if np.linalg.det(R_world2cv) < 0.0:
            U[:, -1] *= -1.0
            R_world2cv = U @ Vt
    except Exception:
        return base_pose
        
    def angular_error(R):
        total = 0.0
        wsum = 0.0
        for src_world, dst_ray, weight in zip(world_dirs, camera_rays, weights):
            pred = R @ src_world
            dot = float(np.clip(np.dot(pred, dst_ray), -1.0, 1.0))
            total += np.arccos(dot) * weight
            wsum += weight
        return total / max(wsum, 1e-8)
    
    base_err = angular_error(R_base_world2cv)
    cand_err = angular_error(R_world2cv)
    R_delta = R_world2cv @ R_base_world2cv.T
    delta_angle = float(np.arccos(np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)))
    
    if cand_err > base_err + np.deg2rad(0.25):
        return base_pose
    if delta_angle > np.deg2rad(45.0) and cand_err > base_err * 0.7:
        return base_pose
    
    R_cv2world = R_world2cv.T
    R_blender = np.dot(R_cv2world, R_bcam2cv)
    return Matrix.Translation(loc) @ Matrix(R_blender).to_4x4()

def get_solve_condition_score(context: bpy.types.Context, cam_data: Any, target_data: Any) -> Tuple[float, int]:
    pins, _ = get_pins(cam_data, target_data)
    pts = []
    for pin in pins:
        if not pin.use_initial or not pin.has_valid_3d:
            continue
        p2d = get_current_pin_pos_2d(context, cam_data, pin)
        if p2d is not None:
            pts.append((float(p2d.x), float(p2d.y)))
            
    count = len(pts)
    if count < 3:
        return 1.0, count
        
    arr = np.array(pts, dtype=np.float64)
    span = np.maximum(np.max(arr, axis=0) - np.min(arr, axis=0), 0.0)
    bbox_area = float(span[0] * span[1])
    coverage_bad = 1.0 - min(1.0, bbox_area / 0.12)
    
    centroid = np.mean(arr, axis=0)
    center_dist = float(np.linalg.norm(centroid - np.array([0.5, 0.5], dtype=np.float64)))
    one_sided_bad = min(1.0, max(0.0, (center_dist - 0.18) / 0.35))
    
    centered = arr - centroid
    line_bad = 0.0
    try:
        cov = np.cov(centered.T)
        eig_vals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        if eig_vals[1] > 1e-8:
            ratio = eig_vals[0] / eig_vals[1]
            line_bad = min(1.0, max(0.0, (0.12 - ratio) / 0.12))
        else:
            line_bad = 1.0
    except Exception:
        line_bad = 1.0
    
    count_bad = max(0.0, (5.0 - count) / 2.0)
    return max(coverage_bad, one_sided_bad, line_bad, min(1.0, count_bad)), count

def restore_sequence_pin_positions(context: bpy.types.Context, cam_data: Any, target_obj: bpy.types.Object, pins: Any, ref_3d_pos: Dict[str, Vector], ref_local_pos: Dict[str, Vector]) -> None:
    if cam_data.solve_mode == 'OBJECT':
        depsgraph = context.evaluated_depsgraph_get()
        tgt_mat = target_obj.evaluated_get(depsgraph).matrix_world if target_obj else Matrix.Identity(4)
        for p in pins:
            p.pos_3d = tgt_mat @ ref_local_pos[p.name]
    else:
        for p in pins:
            p.pos_3d = ref_3d_pos[p.name]

def replace_pose_location(pose: Matrix, location: Vector) -> Matrix:
    _, rot, scale = pose.decompose()
    return Matrix.LocRotScale(location, rot, scale)

def pose_reprojection_error(context: bpy.types.Context, cam_data: Any, target_data: Any, pose: Matrix) -> float:
    if pose is None or not HAS_OPENCV:
        return float("inf")
        
    pins, _ = get_pins(cam_data, target_data)
    obj_pts = []
    img_pts = []
    camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
    for pin in pins:
        if not pin.use_initial or not pin.has_valid_3d:
            continue
        p2d = get_current_pin_pos_2d(context, cam_data, pin)
        if p2d is None:
            continue
        obj_pts.append(list(Vector(pin.pos_3d)))
        img_pts.append(to_cv_pixel(p2d.x, p2d.y, res_x, res_y))
        
    if not obj_pts:
        return float("inf")
        
    try:
        R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        R_world2bcam = np.array(pose.to_3x3().transposed(), dtype=np.float64)
        R_world2cv = R_bcam2cv @ R_world2bcam
        rvec, _ = cv2.Rodrigues(R_world2cv)
        tvec = -R_world2cv @ np.array(pose.translation, dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(np.array(obj_pts, dtype=np.float64), rvec, tvec.reshape(3, 1), camintr, distcoef)
        errors = []
        for proj, img in zip(proj_pts.reshape(-1, 2), img_pts):
            errors.append(float(np.linalg.norm(proj - np.array(img, dtype=np.float64))))
        return float(np.mean(errors)) if errors else float("inf")
    except Exception:
        return float("inf")

def get_active_pin_world_points(context: bpy.types.Context, cam_data: Any, target_data: Any) -> List[Vector]:
    pins, _ = get_pins(cam_data, target_data)
    points = []
    for pin in pins:
        if not pin.use_initial or not pin.has_valid_3d:
            continue
        if get_current_pin_pos_2d(context, cam_data, pin) is None:
            continue
        points.append(Vector(pin.pos_3d))
    return points

def get_pin_plane_hint(points: List[Vector]) -> Tuple[Optional[Vector], Optional[Vector], bool]:
    if len(points) < 3:
        return None, None, False
        
    centroid = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    arr = np.array([list(p - centroid) for p in points], dtype=np.float64)
    try:
        _, s, vh = np.linalg.svd(arr, full_matrices=False)
        if len(s) < 3:
            return centroid, None, False
        normal = Vector(vh[-1]).normalized()
        planar = float(s[-1]) <= max(1e-5, float(s[0]) * 0.025)
        return centroid, normal, planar
    except Exception:
        return centroid, None, False

def pose_forward_vector(pose: Matrix) -> Vector:
    return (pose.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()

def view_direction_to_points(pose: Matrix, centroid: Vector) -> Optional[Vector]:
    loc = pose.translation
    vec = centroid - loc
    if vec.length < 1e-8:
        return None
    return vec.normalized()

def pose_flip_penalty(pose: Matrix, reference_pose: Matrix, points: List[Vector], centroid: Vector, plane_normal: Optional[Vector], is_planar: bool, low_condition: bool) -> float:
    if pose is None or reference_pose is None or not points:
        return 0.0
        
    penalty = 0.0
    forward = pose_forward_vector(pose)
    ref_forward = pose_forward_vector(reference_pose)
    loc = pose.translation
    
    behind_count = 0
    for p in points:
        if (p - loc).dot(forward) <= 1e-5:
            behind_count += 1
    if behind_count:
        penalty += 1000.0 + behind_count * 250.0
        
    forward_angle = ref_forward.angle(forward, 0.0)
    forward_limit = np.deg2rad(18.0 if (low_condition or is_planar) else 35.0)
    if forward_angle > forward_limit:
        penalty += (forward_angle - forward_limit) * (70.0 if (low_condition or is_planar) else 20.0)
        
    ref_view = view_direction_to_points(reference_pose, centroid)
    cand_view = view_direction_to_points(pose, centroid)
    if ref_view is not None and cand_view is not None:
        view_angle = ref_view.angle(cand_view, 0.0)
        view_limit = np.deg2rad(24.0 if (low_condition or is_planar) else 45.0)
        if view_angle > view_limit:
            penalty += (view_angle - view_limit) * (65.0 if (low_condition or is_planar) else 18.0)
            
    if is_planar and plane_normal is not None:
        ref_side = (reference_pose.translation - centroid).dot(plane_normal)
        cand_side = (pose.translation - centroid).dot(plane_normal)
        if abs(ref_side) > 1e-6 and abs(cand_side) > 1e-6 and ref_side * cand_side < 0.0:
            penalty += 1500.0
        if abs(ref_side) > 1e-6 and abs(cand_side) > 1e-6:
            dist_ratio = abs(cand_side) / max(abs(ref_side), 1e-6)
            if dist_ratio > 1.8:
                penalty += (dist_ratio - 1.8) * 180.0
            elif dist_ratio < 0.55:
                penalty += (0.55 - dist_ratio) * 240.0
            
    return penalty if (low_condition or is_planar) else penalty * 0.25

def choose_temporal_pose_candidate(context: bpy.types.Context, cam_data: Any, target_data: Any, reference_pose: Matrix, candidates: List[Matrix], prediction_pose: Matrix = None, mode: str = 'AUTO', strength_factor: float = 1.0, condition_score: float = 0.0, pin_count: int = 0, location_guide_pose: Matrix = None, guide_strength: float = 0.0) -> Optional[Matrix]:
    points = get_active_pin_world_points(context, cam_data, target_data)
    centroid, plane_normal, is_planar = get_pin_plane_hint(points)
    low_condition = condition_score >= 0.45 or pin_count < 5
    scored = []
    guide_strength = max(0.0, min(5.0, float(guide_strength)))
    for pose in candidates:
        if pose is None:
            continue
        reproj = pose_reprojection_error(context, cam_data, target_data, pose)
        if not np.isfinite(reproj):
            continue
        loc_err, rot_err = get_pose_delta(reference_pose, pose) if reference_pose else (0.0, 0.0)
        pred_loc_err, pred_rot_err = get_pose_delta(prediction_pose, pose) if prediction_pose else (loc_err, rot_err)
        guide_loc_err, _ = get_pose_delta(location_guide_pose, pose) if location_guide_pose else (0.0, 0.0)
        flip_penalty = pose_flip_penalty(pose, reference_pose, points, centroid, plane_normal, is_planar, low_condition) if centroid is not None else 0.0
        scored.append((reproj, loc_err, rot_err, pred_loc_err, pred_rot_err, flip_penalty, guide_loc_err, pose))
        
    if not scored:
        return None
        
    best_reproj = min(item[0] for item in scored)
    strength_factor = max(0.0, min(2.0, float(strength_factor)))
    temporal_margin = (0.30 if mode == 'STRONG' else 0.20) * max(0.5, strength_factor)
    if low_condition or is_planar:
        temporal_margin *= 1.75 if mode == 'STRONG' else 1.35
    tolerance = max(1.5 if (low_condition or is_planar) else 1.0, best_reproj * temporal_margin)
    near_best = [item for item in scored if item[0] <= best_reproj + tolerance]
    near_best.sort(key=lambda item: (item[5], item[6] * guide_strength + item[3] + item[4] * 10.0 + item[1] * 0.25, item[0]))
    return near_best[0][7].copy()

def solve_camera_pose(context: bpy.types.Context, cam_data: Any, target_data: Any, camera_obj: bpy.types.Object, target_mode: str = 'initial', skip_calib: bool = False, initial_pose_matrix: Optional[Matrix] = None, allow_global_fallback: bool = True) -> Tuple[bool, Optional[Matrix]]:
    try:
        if not camera_obj or not camera_obj.data: return False, None
            
        pins, active_idx = get_pins(cam_data, target_data)
        valid_pins = []
        valid_p2ds = []
        
        for p in pins:
            is_active = p.use_initial if cam_data.ui_mode == 'MATCHMOVE' else getattr(p, f"use_{target_mode}")
            if is_active and p.has_valid_3d:
                p2d = get_current_pin_pos_2d(context, cam_data, p)
                if p2d is not None:
                    valid_pins.append(p)
                    valid_p2ds.append(p2d)

        num_pins = len(valid_pins)
        if num_pins == 0: return False, None
            
        if initial_pose_matrix is not None:
            cam_loc, cam_rot, _ = initial_pose_matrix.decompose()
            cam_mat_unscaled = Matrix.LocRotScale(cam_loc, cam_rot, Vector((1.0, 1.0, 1.0)))
        else:
            depsgraph = context.evaluated_depsgraph_get()
            eval_cam = camera_obj.evaluated_get(depsgraph)
            cam_loc, cam_rot, _ = eval_cam.matrix_world.decompose()
            cam_mat_unscaled = Matrix.LocRotScale(cam_loc, cam_rot, Vector((1.0, 1.0, 1.0)))
            
        target_data.last_error = ""
        camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
        region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)

        if num_pins == 1:
            pin = valid_pins[0]
            p2d = valid_p2ds[0]
            success, mat, err = _solve_single_pin(context, cam_mat_unscaled, pin, p2d, region, rv3d)
            if not success: target_data.last_error = err
            return success, mat

        if not HAS_OPENCV:
            target_data.last_error = "OpenCV required for 2+ pins"
            return False, None

        obj_pts, img_pts, rvec_guess, tvec_guess = _prep_multi_pin_data(valid_pins, valid_p2ds, res_x, res_y, cam_mat_unscaled, camintr, distcoef)
        if len(obj_pts) < PinSolverConfig.MIN_PINS_FOR_SOLVE and not cam_data.use_planar_solve:
            target_data.last_error = f"Need {PinSolverConfig.MIN_PINS_FOR_SOLVE}+ valid points"
            return False, None

        needs_calib = False if skip_calib else any([cam_data.calib_focal_length, cam_data.calib_optical_center, cam_data.calib_k1, cam_data.calib_k2, cam_data.calib_k3])
        use_sequence_guess = initial_pose_matrix is not None
        use_guess = (target_mode == 'tweak') or use_sequence_guess
        fallback_rvec, fallback_tvec = None, None
        
        if needs_calib:
            camintr_new, distcoef_new, rvec_calib, tvec_calib, calib_success, err_msg = _calibrate_lens(
                cam_data, valid_p2ds, valid_pins, res_x, res_y, camintr, distcoef, target_mode, camera_obj, apply_to_blender=True
            )
            if calib_success:
                camintr = camintr_new
                distcoef = distcoef_new
                if target_mode == 'initial':
                    use_guess = True
                    rvec_guess = rvec_calib
                    tvec_guess = tvec_calib
                    fallback_rvec = rvec_calib
                    fallback_tvec = tvec_calib
            elif err_msg:
                target_data.last_error = err_msg

        success, mat, err = _estimate_pose(obj_pts, img_pts, camintr, distcoef, rvec_guess, tvec_guess, use_guess, fallback_rvec, fallback_tvec, use_planar=cam_data.use_planar_solve, allow_global_fallback=allow_global_fallback)
        if not success and not target_data.last_error:
            target_data.last_error = err
        return success, mat
        
    except Exception as e:
        traceback.print_exc()
        target_data.last_error = f"Err: {type(e).__name__} {str(e)[:20]}"
        return False, None

def apply_solve_result(context: bpy.types.Context, cam_data: Any, target_data: Any, camera_obj: bpy.types.Object, target_obj: bpy.types.Object, result_matrix: Matrix) -> bool:
    try:
        depsgraph = context.evaluated_depsgraph_get()
        eval_cam = camera_obj.evaluated_get(depsgraph)
        
        cam_loc, cam_rot, cam_scale = eval_cam.matrix_world.decompose()
        M_cam_old_unscaled = Matrix.LocRotScale(cam_loc, cam_rot, Vector((1.0, 1.0, 1.0)))
        
        if cam_data.solve_mode == 'OBJECT':
            eval_target = target_obj.evaluated_get(depsgraph)
            try: delta_M = M_cam_old_unscaled @ result_matrix.inverted()
            except ValueError: return False
            
            orig_scale = target_obj.matrix_world.to_scale()
            new_mat = delta_M @ eval_target.matrix_world
            loc, rot, _ = new_mat.decompose()
            target_obj.matrix_world = Matrix.LocRotScale(loc, rot, orig_scale)
            pins, _ = get_pins(cam_data, target_data)
            for pin in pins: pin.pos_3d = delta_M @ Vector(pin.pos_3d)
            return True
            
        elif cam_data.solve_mode == 'PARENT':
            parent_obj = camera_obj.parent
            if not parent_obj: return False
            eval_parent = parent_obj.evaluated_get(depsgraph)
            
            M_parent_world = eval_parent.matrix_world.copy()
            M_cam_world = eval_cam.matrix_world.copy()
            
            try: M_cam_local = M_parent_world.inverted() @ M_cam_world
            except ValueError: return False
            
            target_loc, target_rot, _ = result_matrix.decompose()
            M_cam_world_target = Matrix.LocRotScale(target_loc, target_rot, cam_scale)
            
            try: M_parent_world_target = M_cam_world_target @ M_cam_local.inverted()
            except ValueError: return False
            
            orig_parent_scale = parent_obj.matrix_world.to_scale()
            loc, rot, _ = M_parent_world_target.decompose()
            parent_obj.matrix_world = Matrix.LocRotScale(loc, rot, orig_parent_scale)
            return True
            
        else: # CAMERA
            loc, rot, _ = result_matrix.decompose()
            camera_obj.matrix_world = Matrix.LocRotScale(loc, rot, cam_scale)
            return True
            
    except Exception as e:
        target_data.last_error = f"Application Error"
        return False

# ==========================================
# 3. Operators
# ==========================================
class PinSolverBaseOperator(Operator):
    @classmethod
    def poll(cls, context): return context.scene.camera and context.scene.camera.type == 'CAMERA'

class PinSolverPickMixin:
    def invoke_common(self, context, target_state):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        cam_data.show_overlays = True
        pins, active_idx = get_pins(cam_data, target_data)
        idx = self.target_index if self.target_index != -1 else active_idx
        if idx < 0 or idx >= len(pins): return {'CANCELLED'}
        self.target_index = idx
        target_data.picking_state = target_state
        target_data.picking_index = idx
        
        self.pick_countdown = 0
        self.pick_x = 0
        self.pick_y = 0
        self._timer = context.window_manager.event_timer_add(0.02, window=context.window)
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
        
    def _clean(self, context):
        if hasattr(self, '_timer') and self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

class PINSOLVER_OT_toggle_show_pins(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_toggle_show_pins"
    bl_label = "Toggle Show Pins"
    bl_description = "Toggle visibility of all pins and overlays in the 3D viewport"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        cam_data = context.scene.camera.pinsolver_data
        cam_data.show_overlays = not cam_data.show_overlays
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_set_solve_target(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_set_solve_target"
    bl_label = "Set Solve Target"
    bl_description = "Change the target object to be aligned by the solver"
    bl_options = {'REGISTER', 'UNDO'}
    target: StringProperty()
    def execute(self, context):
        cam_data = context.scene.camera.pinsolver_data
        cam_data.solve_mode = self.target
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_toggle_pin_mode(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_toggle_pin_mode"
    bl_label = "Toggle Pin Property"
    bl_description = "Toggle the participation of this pin in calculations"
    bl_options = {'REGISTER', 'UNDO'}
    mode: StringProperty()
    index: IntProperty(default=-1)
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        pins, active_idx = get_pins(cam_data, target_data)
        idx = self.index if self.index != -1 else active_idx
        if 0 <= idx < len(pins):
            pin = pins[idx]
            if self.mode == 'SOLVE': pin.use_initial = not pin.use_initial
            elif self.mode == 'TWEAK': pin.use_tweak = not pin.use_tweak
            update_reproj_errors(context, cam_data, target_data, force_update=True)
            redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_add_target(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_add_target"
    bl_label = "Add Target"
    bl_description = "Add a new Object target slot to align"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        cam_data = context.scene.camera.pinsolver_data
        cam_data.target_objects.add()
        cam_data.active_target_index = len(cam_data.target_objects) - 1
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_remove_target(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_remove_target"
    bl_label = "Remove Target"
    bl_description = "Remove the selected Object target slot"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        cam_data = context.scene.camera.pinsolver_data
        idx = cam_data.active_target_index
        if 0 <= idx < len(cam_data.target_objects):
            cam_data.target_objects.remove(idx)
            cam_data.active_target_index = max(0, idx - 1)
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_add_pin(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_add_pin"
    bl_label = "Add Pin"
    bl_description = "Add a new pin to the list"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        cam_data.show_overlays = True
        pins, active_idx = get_pins(cam_data, target_data)
        idx = len(pins)
        new_pin = pins.add()
        new_pin.name = f"Pin {idx + 1}"
        new_pin.color = (*colorsys.hsv_to_rgb((idx * 0.618) % 1.0, 0.85, 1.0), 1.0)
        new_pin.pos_2d = (0.5, 0.5)
        new_pin.has_valid_3d = True
        new_pin.reproj_error = -1.0 
        
        cam = context.scene.camera
        if cam:
            offset_dist = context.scene.pinsolver_settings.add_pin_offset
            offset = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -offset_dist))
            new_pin.pos_3d = cam.matrix_world.translation + offset
            
        set_pin_idx(cam_data, target_data, idx)
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_remove_pin(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_remove_pin"
    bl_label = "Remove Pin"
    bl_description = "Remove the selected pin from the list"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty(default=-1)
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        pins, active_idx = get_pins(cam_data, target_data)
        idx = self.index if self.index != -1 else active_idx
        if 0 <= idx < len(pins): pins.remove(idx)
        set_pin_idx(cam_data, target_data, max(0, min(active_idx, len(pins) - 1)))
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_clear_pins(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_clear_pins"
    bl_label = "Clear All Pins"
    bl_description = "Remove all pins from the list (WARNING: Cannot be undone easily)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
        
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        pins, _ = get_pins(cam_data, target_data)
        pins.clear()
        set_pin_idx(cam_data, target_data, 0)
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_solve(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_solve"
    bl_label = "Solve Alignment"
    bl_description = "Calculate and apply the camera/object pose based on active pins"
    bl_options = {'REGISTER', 'UNDO'}
    
    target_mode: StringProperty(default='initial')
    
    def execute(self, context):
        cam_data, target_data, target_obj = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}

        _, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
        if rv3d and rv3d.view_perspective != 'CAMERA':
            rv3d.view_perspective = 'CAMERA'
        cam_data.show_overlays = True
        if cam_data.target_clip:
            sync_scene_camera_from_clip(context, cam_data)
            context.view_layer.update()

        success, result = solve_camera_pose(context, cam_data, target_data, context.scene.camera, target_mode=self.target_mode)
        if success and result:
            if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, result):
                schedule_error_update() 
        return {'FINISHED'}

class PINSOLVER_OT_sync_clip(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_sync_clip"
    bl_label = "Sync Lens"
    bl_description = "Manually push Blender camera lens parameters into the target Movie Clip tracking camera"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        cam_data = context.scene.camera.pinsolver_data
        if not cam_data.target_clip: return {'CANCELLED'}
        clip = cam_data.target_clip
        trk_cam = clip.tracking.camera
        cam_ref = context.scene.camera.data
        
        if trk_cam.k1 != 0.0 or trk_cam.k2 != 0.0 or trk_cam.k3 != 0.0:
            trk_cam.distortion_model = 'POLYNOMIAL'
            
        trk_cam.sensor_width = cam_ref.sensor_width
        trk_cam.focal_length = cam_ref.lens
        
        try:
            res_x = float(clip.size[0])
            res_y = float(clip.size[1])
            max_res = max(res_x, res_y)
            
            px = res_x / 2.0 + (cam_ref.shift_x * max_res)
            py = res_y / 2.0 + (cam_ref.shift_y * max_res)
            
            if bpy.app.version < (3, 5, 0): 
                trk_cam.principal = [px, py]
            else: 
                trk_cam.principal_point_pixels = [px, py]
        except Exception as e:
            self.report({'WARNING'}, f"Could not sync optical center: {e}")
            
        return {'FINISHED'}

class PINSOLVER_OT_pick_2d(PinSolverBaseOperator, PinSolverPickMixin):
    bl_idname = "view3d.pinsolver_pick_2d"
    bl_label = "Pick 2D"
    bl_description = "Pick a new 2D (U,V) position for the selected pin by clicking in the Camera View"
    bl_options = {'REGISTER', 'UNDO'}
    target_index: IntProperty(default=-1)
    
    def modal(self, context, event):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data or target_data.picking_state != 'PICK_2D' or target_data.picking_index != self.target_index:
            self._clean(context)
            return {'CANCELLED'}
            
        if event.type == 'TIMER':
            if hasattr(self, 'pick_countdown') and self.pick_countdown > 0:
                self.pick_countdown -= 1
                if self.pick_countdown == 0:
                    self._do_pick(context, cam_data, target_data)
                    target_data.picking_state = 'NONE'
                    update_reproj_errors(context, cam_data, target_data, force_update=True)
                    redraw_all_3d_views(context)
                    self._clean(context)
                    return {'FINISHED'}
            return {'PASS_THROUGH'}

        if event.type in {'LEFTMOUSE', 'MOUSE_LMB_2X'} and event.value == 'PRESS':
            self.pick_x = event.mouse_x
            self.pick_y = event.mouse_y
            self.pick_countdown = PinSolverConfig.PICK_DELAY_FRAMES
            return {'RUNNING_MODAL'}
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            target_data.picking_state = 'NONE'
            redraw_all_3d_views(context)
            self._clean(context)
            return {'CANCELLED'}
            
        return {'PASS_THROUGH'}

    def _do_pick(self, context, cam_data, target_data):
        class DummyEvent:
            mouse_x = self.pick_x
            mouse_y = self.pick_y
        region, rv3d, rx, ry = get_3d_region_context(context, DummyEvent(), cross_window=True)
        
        if not region or not rv3d: return
        pins, _ = get_pins(cam_data, target_data)
        pin = pins[self.target_index]
        
        if cam_data.use_distortion_overlay:
            u_dist, v_dist = mouse_to_distorted_uv(context, rx, ry, region, rv3d)
            pin.pos_2d = (u_dist, v_dist)
        else:
            bounds = get_camera_frame_bounds(context, region, rv3d)
            if bounds:
                bw = max(1e-4, bounds[2] - bounds[0])
                bh = max(1e-4, bounds[3] - bounds[1])
                pin.pos_2d = ((rx - bounds[0]) / bw, (ry - bounds[1]) / bh)
        
    def invoke(self, context, event):
        return self.invoke_common(context, 'PICK_2D')

class PINSOLVER_OT_pick_3d(PinSolverBaseOperator, PinSolverPickMixin):
    bl_idname = "view3d.pinsolver_pick_3d"
    bl_label = "Pick 3D"
    bl_description = "Pick a new 3D world position for the selected pin by clicking on a mesh surface (Hold Alt to snap to vertex)"
    bl_options = {'REGISTER', 'UNDO'}
    target_index: IntProperty(default=-1)
    
    def modal(self, context, event):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data or target_data.picking_state != 'PICK_3D' or target_data.picking_index != self.target_index:
            self._clean(context)
            return {'CANCELLED'}
            
        if event.type == 'TIMER':
            if hasattr(self, 'pick_countdown') and self.pick_countdown > 0:
                self.pick_countdown -= 1
                if self.pick_countdown == 0:
                    self._do_pick(context, cam_data, target_data)
                    target_data.picking_state = 'NONE'
                    update_reproj_errors(context, cam_data, target_data, force_update=True)
                    redraw_all_3d_views(context)
                    self._clean(context)
                    return {'FINISHED'}
            return {'PASS_THROUGH'}

        if event.type in {'LEFTMOUSE', 'MOUSE_LMB_2X'} and event.value == 'PRESS':
            self.pick_x = event.mouse_x
            self.pick_y = event.mouse_y
            self.is_alt_pressed = event.alt
            self.pick_countdown = PinSolverConfig.PICK_DELAY_FRAMES
            return {'RUNNING_MODAL'}
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            target_data.picking_state = 'NONE'
            redraw_all_3d_views(context)
            self._clean(context)
            return {'CANCELLED'}
            
        return {'PASS_THROUGH'}

    def _do_pick(self, context, cam_data, target_data):
        class DummyEvent:
            mouse_x = self.pick_x
            mouse_y = self.pick_y
        region, rv3d, rx, ry = get_3d_region_context(context, DummyEvent(), cross_window=True)
        
        if not region or not rv3d: return
        origin = region_2d_to_origin_3d(region, rv3d, (rx, ry))
        direction = region_2d_to_vector_3d(region, rv3d, (rx, ry))
        hit, loc = safe_ray_cast(context, origin, direction, snap_to_vertex=getattr(self, 'is_alt_pressed', False))
        
        if hit:
            pins, _ = get_pins(cam_data, target_data)
            pins[self.target_index].pos_3d = loc
            pins[self.target_index].has_valid_3d = True 
        
    def invoke(self, context, event):
        return self.invoke_common(context, 'PICK_3D')

class PINSOLVER_OT_auto_raycast_single(Operator):
    bl_idname = "view3d.pinsolver_auto_raycast_single"
    bl_label = "Raycast 3D"
    bl_description = "Automatically shoot a laser from the camera through this 2D Pin to find its 3D position"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        
        cam_region, cam_rv3d = _get_active_camera_view(context)

        if not cam_region or not cam_rv3d:
            self.report({'WARNING'}, "Requires an active Camera View viewport")
            return {'CANCELLED'}
            
        bounds = get_camera_frame_bounds(context, cam_region, cam_rv3d)
        if not bounds: return {'CANCELLED'}
        
        pins, active_idx = get_pins(cam_data, target_data)
        if active_idx < 0 or active_idx >= len(pins): return {'CANCELLED'}
        pin = pins[active_idx]
        
        if cam_data.ui_mode == 'MATCHMOVE' and not pin.is_track_linked:
            return {'CANCELLED'}
            
        p2d = get_current_pin_pos_2d(context, cam_data, pin)
        if p2d is None: return {'CANCELLED'}
        
        px = get_pin_pixel_coords(context, p2d, bounds=bounds)
        if px is None: return {'CANCELLED'}
        
        origin = region_2d_to_origin_3d(cam_region, cam_rv3d, px)
        direction = region_2d_to_vector_3d(cam_region, cam_rv3d, px)
        if not origin or not direction: return {'CANCELLED'}
        
        hit, loc = safe_ray_cast(context, origin, direction)
        if hit:
            pin.pos_3d = loc
            pin.has_valid_3d = True 
            update_reproj_errors(context, cam_data, target_data, force_update=True)
            redraw_all_3d_views(context)
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Raycast missed geometry")
            return {'CANCELLED'}

class PINSOLVER_OT_edit_pins(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_edit_pins"
    bl_label = "Interactive Pin Editor"
    bl_description = "Interactively adjust pin positions (drag to move, A to add, X to delete) without solving"
    bl_options = {'REGISTER', 'UNDO'}
    dragging_idx: IntProperty(default=-1)
    dragging_type: StringProperty(default='NONE')

    def modal(self, context, event):
        cam_data, target_data, _ = get_active_target_data(context)
        if not cam_data or not target_data or not cam_data.is_edit_mode: return {'FINISHED'}

        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            self.dragging_idx = -1
            self.dragging_type = 'NONE'
            return {'PASS_THROUGH'}

        region, rv3d, rx, ry = get_3d_region_context(context, event, cross_window=True)
        if not region: return {'PASS_THROUGH'}
        mouse_vec = Vector((rx, ry))
        pins, active_idx = get_pins(cam_data, target_data)

        if event.value == 'PRESS' and self.dragging_idx == -1:
            if event.type == 'A' and cam_data.ui_mode == 'LAYOUT':
                origin = region_2d_to_origin_3d(region, rv3d, (rx, ry))
                direction = region_2d_to_vector_3d(region, rv3d, (rx, ry))
                hit, loc = safe_ray_cast(context, origin, direction, snap_to_vertex=event.alt)
                if hit:
                    idx = len(pins)
                    new_pin = pins.add()
                    new_pin.name = f"Pin {idx + 1}"
                    new_pin.pos_3d = loc
                    new_pin.has_valid_3d = True 
                    new_pin.reproj_error = -1.0 
                    
                    if cam_data.use_distortion_overlay:
                        u_dist, v_dist = mouse_to_distorted_uv(context, rx, ry, region, rv3d)
                        new_pin.pos_2d = (u_dist, v_dist)
                    else:
                        bounds = get_camera_frame_bounds(context, region, rv3d)
                        if bounds:
                            bw = max(1e-4, bounds[2] - bounds[0])
                            bh = max(1e-4, bounds[3] - bounds[1])
                            new_pin.pos_2d = ((rx - bounds[0]) / bw, (ry - bounds[1]) / bh)
                            
                    new_pin.color = (*colorsys.hsv_to_rgb((idx * 0.618) % 1.0, 0.85, 1.0), 1.0)
                    set_pin_idx(cam_data, target_data, idx)
                    update_reproj_errors(context, cam_data, target_data, force_update=True)
                    redraw_all_3d_views(context)
                    bpy.ops.ed.undo_push(message="PinSolver: Add Pin")
                return {'RUNNING_MODAL'}
            elif event.type == 'X' and cam_data.ui_mode == 'LAYOUT':
                hover_idx, _ = get_closest_pin_item(context, cam_data, target_data, mouse_vec, region, rv3d)
                if hover_idx != -1:
                    pins.remove(hover_idx)
                    set_pin_idx(cam_data, target_data, max(0, min(active_idx, len(pins) - 1)))
                    update_reproj_errors(context, cam_data, target_data, force_update=True)
                    redraw_all_3d_views(context)
                    bpy.ops.ed.undo_push(message="PinSolver: Delete Pin")
                return {'RUNNING_MODAL'}

        if event.type in {'LEFTMOUSE', 'MOUSE_LMB_2X'}:
            if event.value == 'PRESS':
                prefer = 'NONE'
                if event.shift: prefer = '2D'
                elif event.ctrl: prefer = '3D'

                closest_idx, closest_type = get_closest_pin_item(context, cam_data, target_data, mouse_vec, region, rv3d, prefer)
                if closest_idx != -1:
                    if cam_data.ui_mode == 'MATCHMOVE' and closest_type == '2D':
                        return {'PASS_THROUGH'}
                        
                    self.dragging_idx = closest_idx
                    self.dragging_type = closest_type
                    set_pin_idx(cam_data, target_data, closest_idx)
                    redraw_all_3d_views(context)
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            elif event.value == 'RELEASE':
                if self.dragging_idx != -1:
                    self.dragging_idx = -1
                    self.dragging_type = 'NONE'
                    bpy.ops.ed.undo_push(message="PinSolver: Move Pin")
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
                
        elif event.type == 'MOUSEMOVE' and self.dragging_idx != -1:
            if self.dragging_idx < len(pins):
                pin = pins[self.dragging_idx]
                if self.dragging_type == '2D' and cam_data.ui_mode == 'LAYOUT':
                    if cam_data.use_distortion_overlay:
                        u_dist, v_dist = mouse_to_distorted_uv(context, rx, ry, region, rv3d)
                        pin.pos_2d = (u_dist, v_dist)
                    else:
                        bounds = get_camera_frame_bounds(context, region, rv3d)
                        if bounds:
                            bw = max(1e-4, bounds[2] - bounds[0])
                            bh = max(1e-4, bounds[3] - bounds[1])
                            pin.pos_2d = ((rx - bounds[0]) / bw, (ry - bounds[1]) / bh)
                elif self.dragging_type == '3D':
                    origin = region_2d_to_origin_3d(region, rv3d, (rx, ry))
                    direction = region_2d_to_vector_3d(region, rv3d, (rx, ry))
                    hit, loc = safe_ray_cast(context, origin, direction, snap_to_vertex=event.alt)
                    if hit: 
                        pin.pos_3d = loc
                        pin.has_valid_3d = True 
                
                update_reproj_errors(context, cam_data, target_data, force_update=True)
                redraw_all_3d_views(context)
            return {'RUNNING_MODAL'}
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            cam_data.is_edit_mode = False; redraw_all_3d_views(context); return {'FINISHED'}
            
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        cam_data, target_data, _ = get_active_target_data(context)
        if not cam_data or not target_data: return {'CANCELLED'}
        
        if cam_data.is_edit_mode:
            cam_data.is_edit_mode = False
            redraw_all_3d_views(context)
            return {'FINISHED'}
            
        cam_data.is_edit_mode = True
        cam_data.is_tweak_mode = False 
        cam_data.show_overlays = True 
        self.dragging_idx = -1
        self.dragging_type = 'NONE'
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        context.window_manager.modal_handler_add(self)
        redraw_all_3d_views(context)
        return {'RUNNING_MODAL'}

class PINSOLVER_OT_tweak(PinSolverBaseOperator):
    bl_idname = "view3d.pinsolver_tweak"
    bl_label = "Start Tweak"
    bl_description = "Interactively tweak alignment by dragging 3D pins, recalculating pose in real-time"
    bl_options = {'REGISTER', 'UNDO'}
    dragging_idx: IntProperty(default=-1)

    def _sync_other_pins_2d(self, context, cam_data, target_data, region, rv3d, ignore_idx=-1):
        if cam_data.ui_mode != 'LAYOUT': return
        bounds = get_camera_frame_bounds(context, region, rv3d)
        if not bounds: return
        bw = max(1e-4, bounds[2] - bounds[0])
        bh = max(1e-4, bounds[3] - bounds[1])
        camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)
        
        pins, _ = get_pins(cam_data, target_data)
        for i, pin in enumerate(pins):
            if i == ignore_idx: continue
            if not pin.use_tweak or not pin.has_valid_3d: continue
            
            c2d_lin = location_3d_to_region_2d(region, rv3d, Vector(pin.pos_3d))
            if c2d_lin:
                u_lin = (c2d_lin.x - bounds[0]) / bw
                v_lin = (c2d_lin.y - bounds[1]) / bh
                px_u = u_lin * res_x
                py_u = (1.0 - v_lin) * res_y
                
                has_distortion = np.linalg.norm(distcoef) > 1e-8
                if HAS_OPENCV and has_distortion:
                    X_c = (px_u - camintr[0,2]) / camintr[0,0]
                    Y_c = (py_u - camintr[1,2]) / camintr[1,1]
                    pts_3d = np.array([[[X_c, Y_c, 1.0]]], dtype=np.float64)
                    pts_2d, _ = cv2.projectPoints(pts_3d, np.zeros((3,1), dtype=np.float64), np.zeros((3,1), dtype=np.float64), camintr, distcoef)
                    px_d, py_d = pts_2d[0][0][0], pts_2d[0][0][1]
                else:
                    px_d, py_d = px_u, py_u
                    
                pin.pos_2d = (px_d / res_x, 1.0 - (py_d / res_y))

    def _trigger_solve(self, context, cam_data, target_data, target_obj, is_dragging=False):
        mode = 'tweak' if is_dragging else 'initial'
        success, result = solve_camera_pose(context, cam_data, target_data, context.scene.camera, target_mode=mode)
        if success and result:
            if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, result):
                context.view_layer.update()
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        context.area.tag_redraw()

    def modal(self, context, event):
        cam_data, target_data, target_obj = get_active_target_data(context)
        if not cam_data or not target_data or not cam_data.is_tweak_mode: 
            self._clean(context)
            return {'FINISHED'}

        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            self.dragging_idx = -1
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            if hasattr(self, '_needs_initial_sync') and self._needs_initial_sync:
                self._needs_initial_sync = False
                region, rv3d, _, _ = get_3d_region_context(context, event, cross_window=False)
                if region and rv3d:
                    self._sync_other_pins_2d(context, cam_data, target_data, region, rv3d)
                    update_reproj_errors(context, cam_data, target_data, force_update=True)
                    redraw_all_3d_views(context)
            return {'PASS_THROUGH'}

        region, rv3d, rx, ry = get_3d_region_context(context, event, cross_window=False)
        if not region: return {'PASS_THROUGH'}
        mouse_vec = Vector((rx, ry))
        pins, active_idx = get_pins(cam_data, target_data)

        if event.value == 'PRESS' and self.dragging_idx == -1:
            if event.type == 'A' and cam_data.ui_mode == 'LAYOUT':
                origin = region_2d_to_origin_3d(region, rv3d, (rx, ry))
                direction = region_2d_to_vector_3d(region, rv3d, (rx, ry))
                hit, loc = safe_ray_cast(context, origin, direction)
                if hit:
                    self._sync_other_pins_2d(context, cam_data, target_data, region, rv3d)
                    idx = len(pins)
                    new_pin = pins.add()
                    new_pin.name = f"Pin {idx + 1}"
                    new_pin.use_initial = False
                    new_pin.pos_3d = loc
                    new_pin.has_valid_3d = True
                    new_pin.reproj_error = -1.0
                    
                    if cam_data.use_distortion_overlay:
                        u_dist, v_dist = mouse_to_distorted_uv(context, rx, ry, region, rv3d)
                        new_pin.pos_2d = (u_dist, v_dist)
                    else:
                        bounds = get_camera_frame_bounds(context, region, rv3d)
                        if bounds:
                            bw = max(1e-4, bounds[2] - bounds[0])
                            bh = max(1e-4, bounds[3] - bounds[1])
                            new_pin.pos_2d = ((rx - bounds[0]) / bw, (ry - bounds[1]) / bh)
                            
                    new_pin.color = (*colorsys.hsv_to_rgb((idx * 0.618) % 1.0, 0.85, 1.0), 1.0)
                    set_pin_idx(cam_data, target_data, idx)
                    self._trigger_solve(context, cam_data, target_data, target_obj, is_dragging=False)
                    bpy.ops.ed.undo_push(message="PinSolver: Add Pin")
                return {'RUNNING_MODAL'}
            elif event.type in {'X', 'I', 'T'}:
                hover_idx = get_closest_pin_index(context, cam_data, target_data, mouse_vec, is_tweak=True, region=region, rv3d=rv3d)
                if hover_idx != -1:
                    if event.type == 'X' and cam_data.ui_mode == 'LAYOUT': pins.remove(hover_idx)
                    elif event.type == 'I': pins[hover_idx].use_initial = not pins[hover_idx].use_initial
                    elif event.type == 'T' and cam_data.ui_mode == 'LAYOUT': pins[hover_idx].use_tweak = not pins[hover_idx].use_tweak
                    self._trigger_solve(context, cam_data, target_data, target_obj, is_dragging=False)
                    bpy.ops.ed.undo_push(message="PinSolver: Tweak Pin")
                return {'RUNNING_MODAL'}

        if event.type in {'LEFTMOUSE', 'MOUSE_LMB_2X'}:
            if event.value == 'PRESS':
                closest_idx = get_closest_pin_index(context, cam_data, target_data, mouse_vec, mode_filter='TWEAK', is_tweak=True, region=region, rv3d=rv3d)
                if closest_idx != -1:
                    self.dragging_idx = closest_idx
                    set_pin_idx(cam_data, target_data, closest_idx)
                    self._sync_other_pins_2d(context, cam_data, target_data, region, rv3d, ignore_idx=closest_idx)
                    context.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            elif event.value == 'RELEASE':
                if self.dragging_idx != -1:
                    self.dragging_idx = -1
                    bpy.ops.ed.undo_push(message="PinSolver: Move Pin")
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
                
        elif event.type == 'MOUSEMOVE' and self.dragging_idx != -1:
            if self.dragging_idx < len(pins):
                pin = pins[self.dragging_idx]
                if cam_data.ui_mode == 'LAYOUT':
                    if cam_data.use_distortion_overlay:
                        u_dist, v_dist = mouse_to_distorted_uv(context, rx, ry, region, rv3d)
                        pin.pos_2d = (u_dist, v_dist)
                    else:
                        bounds = get_camera_frame_bounds(context, region, rv3d)
                        if bounds:
                            bw = max(1e-4, bounds[2] - bounds[0])
                            bh = max(1e-4, bounds[3] - bounds[1])
                            pin.pos_2d = ((rx - bounds[0]) / bw, (ry - bounds[1]) / bh)
                self._trigger_solve(context, cam_data, target_data, target_obj, is_dragging=True)
            return {'RUNNING_MODAL'}
            
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            cam_data.is_tweak_mode = False
            schedule_error_update()
            self._clean(context)
            return {'FINISHED'}
            
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        cam_data, target_data, _ = get_active_target_data(context)
        if not cam_data or not target_data: return {'CANCELLED'}
        
        if cam_data.is_tweak_mode:
            cam_data.is_tweak_mode = False
            schedule_error_update()
            return {'FINISHED'}
            
        if cam_data.solve_mode == 'PARENT' and not context.scene.camera.parent: return {'CANCELLED'}
        
        region, rv3d, _, _ = get_3d_region_context(context, cross_window=False)
        
        self._needs_initial_sync = False
        if rv3d and rv3d.view_perspective != 'CAMERA':
            rv3d.view_perspective = 'CAMERA'
            self._needs_initial_sync = True
        else:
            if region and rv3d: self._sync_other_pins_2d(context, cam_data, target_data, region, rv3d)
        
        cam_data.is_tweak_mode = True
        cam_data.is_edit_mode = False 
        cam_data.show_overlays = True 
        self.dragging_idx = -1
        
        self._timer = context.window_manager.event_timer_add(0.02, window=context.window)
        
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        context.window_manager.modal_handler_add(self)
        redraw_all_3d_views(context)
        return {'RUNNING_MODAL'}
        
    def _clean(self, context):
        if hasattr(self, '_timer') and self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

# ==========================================
# 3.5. Matchmove Engine Operators
# ==========================================
class PINSOLVER_OT_send_to_layout(Operator):
    bl_idname = "view3d.pinsolver_send_to_layout"
    bl_label = "Send to Layout"
    bl_description = "Clear current Layout pins and send visible trackers to Layout mode for manual tweaking"
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        
        target_data.layout_pins.clear()
        
        added = 0
        for p in target_data.mm_pins:
            if not p.use_initial or not p.has_valid_3d: continue 
            p2d = get_current_pin_pos_2d(context, cam_data, p)
            if p2d is None: continue 
            
            new_p = target_data.layout_pins.add()
            new_p.name = p.name.replace("Trk: ", "Pin_")
            new_p.pos_2d = p2d
            new_p.pos_3d = p.pos_3d
            new_p.color = p.color
            new_p.weight = p.weight
            new_p.use_initial = True
            new_p.use_tweak = True
            new_p.reproj_error = -1.0
            added += 1
            
        cam_data.ui_mode = 'LAYOUT'
        target_data.layout_pin_idx = 0
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        
        self.report({'INFO'}, f"Sent {added} pins to Layout Mode for tweaking.")
        redraw_all_3d_views(context)
        return {'FINISHED'}

class PINSOLVER_OT_sync_trackers(Operator):
    bl_idname = "view3d.pinsolver_sync_trackers"
    bl_label = "Sync 2D Trackers"
    bl_description = "Import or update 2D Pins from the selected Tracking Layer in the Movie Clip Editor"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data or not cam_data.target_clip:
            self.report({'WARNING'}, "Target Clip is not set")
            return {'CANCELLED'}
            
        clip = cam_data.target_clip
        try:
            idx = int(cam_data.tracking_object_idx)
            tracks = clip.tracking.objects[idx].tracks
        except:
            self.report({'WARNING'}, "Invalid Track Layer")
            return {'CANCELLED'}
            
        if not tracks:
            self.report({'WARNING'}, "No tracks found in the active layer")
            return {'CANCELLED'}
            
        cam_data.show_overlays = True
        
        pins, _ = get_pins(cam_data, target_data)
        alive_track_names = {t.name for t in tracks}
        
        removed_count = 0
        for i in range(len(pins) - 1, -1, -1):
            if pins[i].track_name not in alive_track_names:
                pins.remove(i)
                removed_count += 1
                
        existing_pins = {p.track_name: p for p in pins if p.is_track_linked}
        added_count, updated_count = 0, 0
        
        for t in tracks:
            if t.name in existing_pins:
                updated_count += 1
            else:
                new_pin = pins.add()
                new_pin.name = f"Trk: {t.name}"
                new_pin.track_name = t.name
                new_pin.is_track_linked = True
                new_pin.use_initial = True
                new_pin.has_valid_3d = False
                new_pin.reproj_error = -1.0
                
                if len(t.markers) > 0:
                    marker_co = get_track_marker_co(t, t.markers[0])
                    new_pin.pos_2d = (marker_co.x, marker_co.y)
                
                p_idx = len(pins) - 1
                new_pin.color = (*colorsys.hsv_to_rgb((p_idx * 0.618) % 1.0, 0.85, 1.0), 1.0)
                added_count += 1
                
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        redraw_all_3d_views(context)
        self.report({'INFO'}, f"Sync: {added_count} new, {updated_count} existing, {removed_count} removed")
        return {'FINISHED'}

class PINSOLVER_OT_auto_raycast(Operator):
    bl_idname = "view3d.pinsolver_auto_raycast"
    bl_label = "Auto Raycast 3D Pins"
    bl_description = "Automatically shoot lasers from the camera through linked 2D Pins to find their 3D positions"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        cam_data, target_data, _ = get_active_target_data(context)
        if not target_data: return {'CANCELLED'}
        
        cam_region, cam_rv3d = _get_active_camera_view(context)
        if not cam_region or not cam_rv3d:
            self.report({'WARNING'}, "Requires an active Camera View viewport")
            return {'CANCELLED'}
            
        bounds = get_camera_frame_bounds(context, cam_region, cam_rv3d)
        if not bounds: return {'CANCELLED'}
        
        pins, _ = get_pins(cam_data, target_data)
        hit_count = 0
        for pin in pins:
            if not pin.is_track_linked: continue
            
            p2d = get_current_pin_pos_2d(context, cam_data, pin)
            if p2d is None: continue
            
            px = get_pin_pixel_coords(context, p2d, bounds=bounds)
            if px is None: continue
            
            origin = region_2d_to_origin_3d(cam_region, cam_rv3d, px)
            direction = region_2d_to_vector_3d(cam_region, cam_rv3d, px)
            if not origin or not direction: continue
            
            hit, loc = safe_ray_cast(context, origin, direction)
            if hit:
                pin.pos_3d = loc
                pin.has_valid_3d = True 
                hit_count += 1
                
        update_reproj_errors(context, cam_data, target_data, force_update=True)
        redraw_all_3d_views(context)
        self.report({'INFO'}, f"Auto-Raycasted {hit_count} pins")
        return {'FINISHED'}

class PINSOLVER_OT_set_reference_frame(Operator):
    bl_idname = "view3d.pinsolver_set_reference_frame"
    bl_label = "Set Reference Frame"
    bl_description = "Lock the current frame as the 'Ground Truth' for 3D Pin coordinates during matchmove"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        cam_data = context.scene.camera.pinsolver_data
        cam_data.reference_frame = context.scene.frame_current
        self.report({'INFO'}, f"Reference Frame set to {cam_data.reference_frame}")
        return {'FINISHED'}

class PINSOLVER_OT_bake_animation(Operator):
    bl_idname = "view3d.pinsolver_bake_animation"
    bl_label = "Sequence Solver"
    bl_description = "Solve PnP for each targeted frame and bake keyframes"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        cam_data, target_data, target_obj = get_active_target_data(context)
        if not target_data or not cam_data.target_clip: return {'CANCELLED'}
        
        clip = cam_data.target_clip
        try:
            idx = int(cam_data.tracking_object_idx)
            tracks = clip.tracking.objects[idx].tracks
        except:
            self.report({'WARNING'}, "Invalid Track Layer")
            return {'CANCELLED'}
                
        if not tracks: 
            self.report({'WARNING'}, "No tracks found in the active tracking layer")
            return {'CANCELLED'}
        
        bake_frames = []
        if cam_data.bake_target == 'MARKERS':
            mrks = context.scene.timeline_markers
            if not mrks:
                self.report({'WARNING'}, "No Timeline Markers found")
                return {'CANCELLED'}
            bake_frames = sorted(list(set([m.frame for m in mrks])))
        else:
            bake_frames = list(range(context.scene.frame_start, context.scene.frame_end + 1))
            
        if not bake_frames: return {'CANCELLED'}
        orig_frame = context.scene.frame_current
        motion_source = getattr(cam_data, "sequence_motion_source", 'PNP')
        if motion_source == 'MARKER_REFS' and cam_data.bake_target != 'SCENE':
            self.report({'WARNING'}, "Timeline Markers source requires Scene Range")
            return {'CANCELLED'}
        use_existing_location = motion_source == 'EXISTING'
        use_guided_location = motion_source == 'GUIDED'
        use_marker_references = motion_source == 'MARKER_REFS'
        marker_reference_frames = set()
        marker_reference_pose_by_frame = {}
        if use_marker_references:
            marker_reference_frames = {
                m.frame for m in context.scene.timeline_markers
                if context.scene.frame_start <= m.frame <= context.scene.frame_end
            }
            if not marker_reference_frames:
                self.report({'WARNING'}, "Timeline Markers source requires markers inside the scene range")
                return {'CANCELLED'}
            marker_reference_frames.add(cam_data.reference_frame)
            for f in sorted(marker_reference_frames):
                context.scene.frame_set(f)
                context.view_layer.update()
                depsgraph = context.evaluated_depsgraph_get()
                marker_reference_pose_by_frame[f] = get_camera_unscaled_matrix(context.scene.camera, depsgraph)
        existing_location_by_frame = {}
        if use_existing_location or use_guided_location:
            for f in bake_frames:
                context.scene.frame_set(f)
                context.view_layer.update()
                depsgraph = context.evaluated_depsgraph_get()
                existing_location_by_frame[f] = context.scene.camera.evaluated_get(depsgraph).matrix_world.translation.copy()
        
        pins, _ = get_pins(cam_data, target_data)
        valid_pin_count = sum(1 for p in pins if p.use_initial and p.has_valid_3d)
        min_sequence_pins = 2 if use_existing_location else 4
        if valid_pin_count < min_sequence_pins:
            if use_existing_location:
                self.report({'WARNING'}, "Existing Location requires 2+ valid pins for rotation solve")
            else:
                self.report({'WARNING'}, "Sequence PnP requires 4+ valid pins")
            return {'CANCELLED'}
            
        ref_3d_pos = {p.name: Vector(p.pos_3d) for p in pins}
        
        needs_calib = any([cam_data.calib_focal_length, cam_data.calib_optical_center, cam_data.calib_k1, cam_data.calib_k2, cam_data.calib_k3])
        
        cam_ref = context.scene.camera.data
        trk_cam = clip.tracking.camera
        sync_scene_camera_from_clip(context, cam_data)
        context.view_layer.update()
        
        # ----------------------------------------------------
        # Pass 1 - Ground Truth Calibration
        # ----------------------------------------------------
        if needs_calib and cam_data.calib_animation_mode == 'STATIC' and cam_data.calib_static_method != 'CURRENT':
            static_data = []
            for f in bake_frames:
                context.scene.frame_set(f)
                success, _ = solve_camera_pose(context, cam_data, target_data, context.scene.camera, target_mode='initial', skip_calib=False)
                if success:
                    static_data.append((cam_ref.lens, cam_ref.shift_x, cam_ref.shift_y, trk_cam.k1, trk_cam.k2, trk_cam.k3))
            
            if static_data:
                np_data = np.array(static_data, dtype=np.float64)
                if cam_data.calib_static_method == 'MEDIAN':
                    s_vals = np.median(np_data, axis=0)
                else:
                    s_vals = np.mean(np_data, axis=0)
                base_lens, base_sx, base_sy, base_k1, base_k2, base_k3 = s_vals
            else:
                base_lens, base_sx, base_sy = cam_ref.lens, cam_ref.shift_x, cam_ref.shift_y
                base_k1, base_k2, base_k3 = trk_cam.k1, trk_cam.k2, trk_cam.k3
                
        else: # ZOOM or CURRENT
            context.scene.frame_set(cam_data.reference_frame)
            success, _ = solve_camera_pose(context, cam_data, target_data, context.scene.camera, target_mode='initial', skip_calib=not needs_calib)
            base_lens = cam_ref.lens
            base_sx = cam_ref.shift_x
            base_sy = cam_ref.shift_y
            base_k1, base_k2, base_k3 = trk_cam.k1, trk_cam.k2, trk_cam.k3
        
        if cam_data.calib_animation_mode == 'STATIC':
            cam_ref.lens = base_lens
            cam_ref.shift_x = base_sx
            cam_ref.shift_y = base_sy
            trk_cam.k1, trk_cam.k2, trk_cam.k3 = base_k1, base_k2, base_k3

        res_x = max(1.0, context.scene.render.resolution_x * (context.scene.render.resolution_percentage / 100.0))
        res_y = max(1.0, context.scene.render.resolution_y * (context.scene.render.resolution_percentage / 100.0))
        
        context.scene.frame_set(cam_data.reference_frame)
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        target_obj_eval = target_obj.evaluated_get(depsgraph) if target_obj else None
        target_matrix_inv = target_obj_eval.matrix_world.inverted() if target_obj_eval else Matrix.Identity(4)
        ref_local_pos = {p.name: (target_matrix_inv @ Vector(p.pos_3d)) for p in pins}
        ref_result = get_camera_unscaled_matrix(context.scene.camera, depsgraph)
        ref_key_obj = target_obj if cam_data.solve_mode == 'OBJECT' else context.scene.camera
        if cam_data.solve_mode == 'PARENT' and context.scene.camera.parent:
            ref_key_obj = context.scene.camera.parent
        ref_key_matrix = ref_key_obj.matrix_world.copy() if ref_key_obj else None
        
        # ----------------------------------------------------
        # Pass 2 - Chain Solving
        # ----------------------------------------------------
        success_count = 0
        frames_forward = [f for f in bake_frames if f >= cam_data.reference_frame]
        frames_backward = sorted([f for f in bake_frames if f < cam_data.reference_frame], reverse=True)
        chain_pose = ref_result.copy()
        chain_frame = cam_data.reference_frame
        chain_prev_pose = None
        chain_prev_frame = None
        chain_prev2_pose = None
        chain_prev2_frame = None
        solved_pose_by_frame = {}
        condition_by_frame = {}
        pin_count_by_frame = {}
        location_filter_strength = max(0.0, min(2.0, float(getattr(cam_data, "sequence_location_filter_strength", 1.0))))
        guided_location_strength = max(0.0, min(2.0, float(getattr(cam_data, "sequence_guided_location_strength", 1.0))))
        effective_stabilize_mode = getattr(cam_data, "sequence_stabilize_mode", 'OFF')
        if location_filter_strength <= 1e-6:
            effective_stabilize_mode = 'OFF'
        
        def keyframe_current_target(f):
            obj_to_key = target_obj if cam_data.solve_mode == 'OBJECT' else context.scene.camera
            if cam_data.solve_mode == 'PARENT' and context.scene.camera.parent:
                obj_to_key = context.scene.camera.parent
            if not obj_to_key:
                return False
            if f == cam_data.reference_frame and ref_key_matrix is not None:
                obj_to_key.matrix_world = ref_key_matrix.copy()
                context.view_layer.update()
            obj_to_key.keyframe_insert(data_path="location", frame=f)
            if obj_to_key.rotation_mode == 'QUATERNION':
                obj_to_key.keyframe_insert(data_path="rotation_quaternion", frame=f)
            else:
                obj_to_key.keyframe_insert(data_path="rotation_euler", frame=f)
            return True

        def process_frame(f):
            nonlocal success_count, chain_pose, chain_frame, chain_prev_pose, chain_prev_frame, chain_prev2_pose, chain_prev2_frame
            context.scene.frame_set(f)
            context.view_layer.update()
            
            depsgraph = context.evaluated_depsgraph_get()
            tgt_mat = target_obj.evaluated_get(depsgraph).matrix_world if target_obj else Matrix.Identity(4)
            
            for p in pins:
                if cam_data.solve_mode == 'OBJECT':
                    p.pos_3d = tgt_mat @ ref_local_pos[p.name]
                else:
                    p.pos_3d = ref_3d_pos[p.name]
                
            if needs_calib and cam_data.calib_animation_mode == 'ZOOM':
                cam_ref.lens = base_lens
                cam_ref.shift_x = base_sx
                cam_ref.shift_y = base_sy
                trk_cam.k1, trk_cam.k2, trk_cam.k3 = base_k1, base_k2, base_k3
                
                if cam_data.use_dynamic_zoom:
                    valid_pins = []
                    valid_p2ds = []
                    for p in pins:
                        if p.use_initial and p.has_valid_3d:
                            p2d = get_current_pin_pos_2d(context, cam_data, p)
                            if p2d is not None:
                                valid_pins.append(p)
                                valid_p2ds.append(p2d)
                                
                    if len(valid_pins) >= PinSolverConfig.MIN_PINS_FOR_CALIBRATION:
                        sw = cam_ref.sensor_height * (res_x / res_y) if cam_ref.sensor_fit == 'VERTICAL' or (cam_ref.sensor_fit == 'AUTO' and res_x < res_y) else cam_ref.sensor_width
                        sh = cam_ref.sensor_height if cam_ref.sensor_fit == 'VERTICAL' or (cam_ref.sensor_fit == 'AUTO' and res_x < res_y) else cam_ref.sensor_width * (res_y / res_x)
                        fx = (base_lens / max(1e-4, sw)) * res_x
                        fy = (base_lens / max(1e-4, sh)) * res_y
                        cx = res_x / 2.0 + (base_sx * max(res_x, res_y))
                        cy = res_y / 2.0 - (base_sy * max(res_x, res_y))
                        
                        z_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                        z_dist = np.array([base_k1, base_k2, 0, 0, base_k3], dtype=np.float64)
                        
                        camintr_new, _, _, _, succ, _ = _calibrate_lens(
                            cam_data, valid_p2ds, valid_pins, res_x, res_y, z_intr, z_dist, 'initial', context.scene.camera, apply_to_blender=False, zoom_base_intrinsics=z_intr, zoom_base_distcoef=z_dist
                        )
                        
                        if succ:
                            new_lens = float(camintr_new[0, 0]) * max(1e-4, sw) / res_x
                            new_cx = float(camintr_new[0, 2])
                            new_cy = float(camintr_new[1, 2])
                            new_sx = (new_cx - res_x / 2.0) / max(res_x, res_y)
                            new_sy = (res_y / 2.0 - new_cy) / max(res_x, res_y)
                            
                            if cam_data.calib_focal_length: cam_ref.lens = new_lens
                            if cam_data.calib_optical_center:
                                cam_ref.shift_x = new_sx
                                cam_ref.shift_y = new_sy
                
                if cam_data.calib_focal_length: cam_ref.keyframe_insert("lens", frame=f)
                if cam_data.calib_optical_center:
                    cam_ref.keyframe_insert("shift_x", frame=f)
                    cam_ref.keyframe_insert("shift_y", frame=f)

            if f == cam_data.reference_frame:
                chain_pose = ref_result.copy()
                chain_frame = f
                chain_prev_pose = None
                chain_prev_frame = None
                chain_prev2_pose = None
                chain_prev2_frame = None
                solved_pose_by_frame[f] = ref_result.copy()
                condition_score, pin_count = get_solve_condition_score(context, cam_data, target_data)
                condition_by_frame[f] = condition_score
                pin_count_by_frame[f] = pin_count
                if keyframe_current_target(f):
                    success_count += 1
                return
                
            stabilize_mode = effective_stabilize_mode
            marker_pose = marker_reference_pose_by_frame.get(f).copy() if use_marker_references and f in marker_reference_pose_by_frame else None
            guide_loc = existing_location_by_frame.get(f) if use_guided_location else None
            guided_pose = replace_pose_location(chain_pose, guide_loc) if guide_loc is not None else None
            initial_pose = marker_pose if marker_pose is not None else (guided_pose if guided_pose is not None else chain_pose)
            prediction_pose = extrapolate_location_pose(
                chain_prev_pose, chain_prev_frame, chain_pose, chain_frame, f,
                older_pose=chain_prev2_pose,
                older_frame=chain_prev2_frame,
                use_acceleration=(stabilize_mode == 'STRONG')
            )
            candidates = []
            success, result = solve_camera_pose(
                context, cam_data, target_data, context.scene.camera, target_mode='initial', skip_calib=needs_calib,
                initial_pose_matrix=initial_pose,
                allow_global_fallback=False
            )
            if success and result:
                candidates.append(result.copy())
                
            if ((use_guided_location and guided_pose is not None) or marker_pose is not None) and chain_pose is not None:
                chain_success, chain_result = solve_camera_pose(
                    context, cam_data, target_data, context.scene.camera, target_mode='initial', skip_calib=needs_calib,
                    initial_pose_matrix=chain_pose,
                    allow_global_fallback=False
                )
                if chain_success and chain_result:
                    candidates.append(chain_result.copy())
                 
            if not success and initial_pose is not None and stabilize_mode != 'STRONG':
                success, fallback_result = solve_camera_pose(
                    context, cam_data, target_data, context.scene.camera, target_mode='initial', skip_calib=needs_calib,
                    initial_pose_matrix=initial_pose,
                    allow_global_fallback=True
                )
                if success and fallback_result:
                    loc_err, rot_err = get_pose_delta(initial_pose, fallback_result)
                    prev_loc, _, _ = initial_pose.decompose()
                    depth_scale = max(1.0, prev_loc.length * 0.02)
                    if loc_err <= depth_scale * 6.0 and rot_err <= np.deg2rad(35.0):
                        result = fallback_result
                    else:
                        success = False
                        result = None
                if success and result:
                    candidates.append(result.copy())
                    
            if not use_existing_location and stabilize_mode != 'OFF':
                global_success, global_result = solve_camera_pose(
                    context, cam_data, target_data, context.scene.camera, target_mode='initial', skip_calib=needs_calib,
                    initial_pose_matrix=None,
                    allow_global_fallback=True
                )
                if global_success and global_result:
                    candidates.append(global_result.copy())
                    
            if candidates:
                candidate_condition_score, candidate_pin_count = get_solve_condition_score(context, cam_data, target_data)
                chosen = choose_temporal_pose_candidate(
                    context, cam_data, target_data, initial_pose, candidates,
                    prediction_pose=prediction_pose,
                    mode=stabilize_mode,
                    strength_factor=location_filter_strength,
                    condition_score=candidate_condition_score,
                    pin_count=candidate_pin_count,
                    location_guide_pose=marker_pose if marker_pose is not None else guided_pose,
                    guide_strength=4.5 if marker_pose is not None else (3.0 * guided_location_strength if use_guided_location else 0.0)
                )
                if chosen:
                    result = chosen
                    success = True
            
            if use_guided_location and guide_loc is not None and success and result:
                result = smooth_guided_location_pose(result, guide_loc, guided_location_strength)
                 
            if use_existing_location:
                anchor_loc = existing_location_by_frame.get(f)
                if anchor_loc is None:
                    return
                base_pose = result if success and result else chain_pose
                result = refine_rotation_for_fixed_location(context, cam_data, target_data, replace_pose_location(base_pose, anchor_loc))
                success = True
            
            if success and result:
                if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, result):
                    context.view_layer.update() 
                    chain_prev2_pose = chain_prev_pose.copy() if chain_prev_pose else None
                    chain_prev2_frame = chain_prev_frame
                    chain_prev_pose = chain_pose.copy()
                    chain_prev_frame = chain_frame
                    chain_pose = result.copy()
                    chain_frame = f
                    solved_pose_by_frame[f] = result.copy()
                    condition_score, pin_count = get_solve_condition_score(context, cam_data, target_data)
                    condition_by_frame[f] = condition_score
                    pin_count_by_frame[f] = pin_count
                    if keyframe_current_target(f):
                        success_count += 1

        context.scene.frame_set(cam_data.reference_frame)
        apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, ref_result)
        context.view_layer.update()
        chain_pose = ref_result.copy()
        chain_frame = cam_data.reference_frame
        chain_prev_pose = None
        chain_prev_frame = None
        chain_prev2_pose = None
        chain_prev2_frame = None
        for f in frames_forward:
            process_frame(f)
            
        context.scene.frame_set(cam_data.reference_frame)
        apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, ref_result)
        context.view_layer.update()
        chain_pose = ref_result.copy()
        chain_frame = cam_data.reference_frame
        chain_prev_pose = None
        chain_prev_frame = None
        chain_prev2_pose = None
        chain_prev2_frame = None
        for f in frames_backward:
            process_frame(f)
            
        marker_anchor_frames = set(marker_reference_frames) if use_marker_references else set()
        marker_anchor_frames.add(cam_data.reference_frame)
        fixed_reference_frames = set(marker_anchor_frames) if use_marker_references else {cam_data.reference_frame}
        
        if use_marker_references:
            aligned_pose_by_frame = align_marker_reference_segments(solved_pose_by_frame, marker_reference_pose_by_frame, marker_anchor_frames)
            for f in sorted(aligned_pose_by_frame.keys()):
                context.scene.frame_set(f)
                restore_sequence_pin_positions(context, cam_data, target_obj, pins, ref_3d_pos, ref_local_pos)
                corrected_pose = aligned_pose_by_frame[f].copy() if f in fixed_reference_frames else refine_rotation_for_fixed_location(context, cam_data, target_data, aligned_pose_by_frame[f])
                if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, corrected_pose):
                    context.view_layer.update()
                    solved_pose_by_frame[f] = corrected_pose.copy()
                    keyframe_current_target(f)
            
        if not use_existing_location and effective_stabilize_mode != 'OFF' and len(solved_pose_by_frame) >= 2:
            mode = effective_stabilize_mode
            sorted_frames = sorted(bake_frames)
            anchor_frames = [
                f for f in sorted_frames
                if f in solved_pose_by_frame and (f in marker_anchor_frames or (condition_by_frame.get(f, 1.0) < 0.45 and pin_count_by_frame.get(f, 0) >= 5))
            ]
            for ref_f in sorted(marker_anchor_frames):
                if ref_f in solved_pose_by_frame and ref_f not in anchor_frames:
                    anchor_frames.append(ref_f)
                anchor_frames.sort()
                
            bad_frames = set()
            for f in sorted_frames:
                if f in fixed_reference_frames:
                    continue
                prev_anchors = [af for af in anchor_frames if af < f]
                next_anchors = [af for af in anchor_frames if af > f]
                if not prev_anchors or not next_anchors:
                    continue
                prev_f = max(prev_anchors)
                next_f = min(next_anchors)
                prev_pose = solved_pose_by_frame[prev_f]
                next_pose = solved_pose_by_frame[next_f]
                factor = (f - prev_f) / max(1.0, next_f - prev_f)
                expected = interpolate_pose_matrices(prev_pose, next_pose, factor)
                condition_score = condition_by_frame.get(f, 1.0)
                pin_count = pin_count_by_frame.get(f, 0)
                if f not in solved_pose_by_frame:
                    bad_frames.add(f)
                    continue
                loc_err, rot_err = get_pose_delta(expected, solved_pose_by_frame[f])
                span_loc, span_rot = get_pose_delta(prev_pose, next_pose)
                expected_loc, _, _ = expected.decompose()
                depth_scale = max(1.0, expected_loc.length * 0.02)
                if mode == 'STRONG':
                    loc_limit = max(depth_scale * 2.5, span_loc * 2.0)
                    rot_limit = max(np.deg2rad(14.0), span_rot * 2.0 + np.deg2rad(5.0))
                else:
                    loc_limit = max(depth_scale * 4.0, span_loc * 2.5)
                    rot_limit = max(np.deg2rad(24.0), span_rot * 2.5 + np.deg2rad(8.0))
                low_condition = condition_score >= 0.45 or pin_count < 5
                if low_condition and (loc_err > loc_limit or rot_err > rot_limit):
                    bad_frames.add(f)
            
            if bad_frames:
                for f in sorted(bad_frames):
                    prev_anchors = [af for af in anchor_frames if af < f]
                    next_anchors = [af for af in anchor_frames if af > f]
                    if not prev_anchors or not next_anchors:
                        continue
                    prev_f = max(prev_anchors)
                    next_f = min(next_anchors)
                    factor = (f - prev_f) / max(1.0, next_f - prev_f)
                    anchor_repaired_pose = interpolate_pose_matrices(solved_pose_by_frame[prev_f], solved_pose_by_frame[next_f], factor)
                    if f in solved_pose_by_frame:
                        repaired_pose = interpolate_pose_matrices(solved_pose_by_frame[f], anchor_repaired_pose, min(1.0, location_filter_strength))
                    else:
                        repaired_pose = anchor_repaired_pose
                    context.scene.frame_set(f)
                    restore_sequence_pin_positions(context, cam_data, target_obj, pins, ref_3d_pos, ref_local_pos)
                    if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, repaired_pose):
                        context.view_layer.update()
                        solved_pose_by_frame[f] = repaired_pose.copy()
                        keyframe_current_target(f)
            
            stabilized_pose_by_frame = stabilize_location_pose_path(
                solved_pose_by_frame,
                condition_by_frame,
                pin_count_by_frame,
                mode,
                location_filter_strength,
                fixed_reference_frames,
                marker_reference_pose_by_frame if use_marker_references else None
            )
            for f in sorted(stabilized_pose_by_frame.keys()):
                if f in fixed_reference_frames:
                    continue
                context.scene.frame_set(f)
                restore_sequence_pin_positions(context, cam_data, target_obj, pins, ref_3d_pos, ref_local_pos)
                corrected_pose = refine_rotation_for_fixed_location(context, cam_data, target_data, stabilized_pose_by_frame[f])
                if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, corrected_pose):
                    context.view_layer.update()
                    solved_pose_by_frame[f] = corrected_pose.copy()
                    keyframe_current_target(f)
            
            if use_marker_references:
                reinforced_pose_by_frame = reinforce_fixed_reference_transitions(
                    solved_pose_by_frame,
                    fixed_reference_frames,
                    mode,
                    location_filter_strength
                )
                for f in sorted(reinforced_pose_by_frame.keys()):
                    if f in fixed_reference_frames:
                        continue
                    context.scene.frame_set(f)
                    restore_sequence_pin_positions(context, cam_data, target_obj, pins, ref_3d_pos, ref_local_pos)
                    corrected_pose = refine_rotation_for_fixed_location(context, cam_data, target_data, reinforced_pose_by_frame[f])
                    if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, corrected_pose):
                        context.view_layer.update()
                        solved_pose_by_frame[f] = corrected_pose.copy()
                        keyframe_current_target(f)
        
        roll_smoothed_pose_by_frame = stabilize_roll_pose_path(
            solved_pose_by_frame,
            getattr(cam_data, "sequence_roll_smoothing", 'OFF'),
            cam_data.reference_frame,
            getattr(cam_data, "sequence_roll_smoothing_strength", 1.0),
            fixed_reference_frames
        )
        for f in sorted(roll_smoothed_pose_by_frame.keys()):
            context.scene.frame_set(f)
            restore_sequence_pin_positions(context, cam_data, target_obj, pins, ref_3d_pos, ref_local_pos)
            if apply_solve_result(context, cam_data, target_data, context.scene.camera, target_obj, roll_smoothed_pose_by_frame[f]):
                context.view_layer.update()
                solved_pose_by_frame[f] = roll_smoothed_pose_by_frame[f].copy()
                keyframe_current_target(f)
            
        # ----------------------------------------------------
        # Pass 3 - Evaluation of raw OpenCV data after the process has fully completed
        # ----------------------------------------------------
        total_reproj_error = 0.0
        pin_errors = {p.name: [] for p in pins}
        
        for f in bake_frames:
            context.scene.frame_set(f)
            context.view_layer.update()
            
            depsgraph = context.evaluated_depsgraph_get()
            tgt_mat = target_obj.evaluated_get(depsgraph).matrix_world if target_obj else Matrix.Identity(4)
            
            for p in pins:
                if cam_data.solve_mode == 'OBJECT':
                    p.pos_3d = tgt_mat @ ref_local_pos[p.name]
                else:
                    p.pos_3d = ref_3d_pos[p.name]
                    
            update_reproj_errors(context, cam_data, target_data, force_update=True)
            
            frame_err = 0.0
            valid_idx = 0
            for p in pins:
                if p.reproj_error >= 0:
                    pin_errors[p.name].append(p.reproj_error)
                    frame_err += p.reproj_error
                    valid_idx += 1
                    
            if valid_idx > 0:
                total_reproj_error += (frame_err / valid_idx)

        target_data.mm_avg_error = (total_reproj_error / success_count) if success_count > 0 else -1.0
        for p in pins:
            errs = pin_errors.get(p.name, [])
            if errs:
                p.reproj_error = float(np.max(errs)) 
            else:
                p.reproj_error = -1.0

        context.scene.frame_set(orig_frame)
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        tgt_mat = target_obj.evaluated_get(depsgraph).matrix_world if target_obj else Matrix.Identity(4)
        for p in pins:
            if cam_data.solve_mode == 'OBJECT':
                p.pos_3d = tgt_mat @ ref_local_pos[p.name]
            else:
                p.pos_3d = ref_3d_pos[p.name]

        redraw_all_3d_views(context)
        
        final_err_msg = f" | Seq Avg Error: {(total_reproj_error/success_count):.2f} px" if success_count > 0 else ""
        self.report({'INFO'}, f"Baked {success_count} frames" + final_err_msg)
        return {'FINISHED'}

# ==========================================
# 4. Drawing Overlays
# ==========================================
def draw_shape(x, y, radius, color, shape='CIRCLE'):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    verts, indices = [], []
    if shape == 'CIRCLE':
        for i in range(16):
            th = 2 * np.pi * i / 16
            verts.append((x + radius * np.cos(th), y + radius * np.sin(th)))
        for i in range(1, 16): indices.append((0, i, i+1))
        indices.append((0, 16, 1))
        verts.insert(0, (x, y))
    else:
        verts = [(x-radius, y-radius), (x+radius, y-radius), (x+radius, y+radius), (x-radius, y+radius)]
        indices = [(0, 1, 2), (0, 2, 3)]
    batch = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=indices)
    gpu.state.blend_set('ALPHA'); shader.bind(); shader.uniform_float("color", color); batch.draw(shader); gpu.state.blend_set('NONE')

def draw_line(x1, y1, x2, y2, color, line_width=1.0):
    try: gpu.state.line_width_set(line_width)
    except: pass
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": [(x1, y1), (x2, y2)]})
    gpu.state.blend_set('ALPHA'); shader.bind(); shader.uniform_float("color", color); batch.draw(shader); gpu.state.blend_set('NONE')
    try: gpu.state.line_width_set(1.0)
    except: pass

def draw_callback_overlay():
    context = bpy.context
    if not getattr(context, "space_data", None) or context.space_data.type != 'VIEW_3D': return
    cam_data, target_data, target_obj = get_active_target_data(context)
    if not cam_data or not target_data or not cam_data.show_overlays: return
        
    region = context.region
    rv3d = context.region_data
    if region.type != 'WINDOW': return
    is_cam_view = (rv3d.view_perspective == 'CAMERA')
    
    cam_region = region if is_cam_view else None
    cam_rv3d = rv3d if is_cam_view else None
    
    settings = context.scene.pinsolver_settings
    pins, active_idx = get_pins(cam_data, target_data)

    out_c = list(settings.text_outline_color)
    out_c[3] *= settings.overlay_opacity
    use_out = settings.text_use_outline

    def _draw_txt(txt, x, y, sz, c):
        blf.size(0, sz)
        if use_out:
            blf.color(0, *out_c)
            for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1), (1,0), (-1,0), (0,1), (0,-1)]:
                blf.position(0, x + dx, y + dy, 0)
                blf.draw(0, txt)
        blf.color(0, *c)
        blf.position(0, x, y, 0)
        blf.draw(0, txt)

    def _draw_txt_centered(txt, y, sz, c):
        blf.size(0, sz)
        x = (region.width - blf.dimensions(0, txt)[0]) / 2
        _draw_txt(txt, x, y, sz, c)

    if cam_data.is_edit_mode:
        tc = (0.2, 0.8, 1.0, settings.overlay_opacity)
        _draw_txt_centered(UIStrings.OVERLAY_EDIT_TITLE, 60, 20, tc)
        
        sub_str = UIStrings.OVERLAY_EDIT_SUB_LAYOUT if cam_data.ui_mode == 'LAYOUT' else UIStrings.OVERLAY_EDIT_SUB_MM
        sc = (1.0, 1.0, 1.0, 0.9 * settings.overlay_opacity)
        _draw_txt_centered(sub_str, 35, 14, sc)
        
    elif cam_data.is_tweak_mode: 
        active_pins = sum(1 for p in pins if p.use_tweak)
        if active_pins == 0: title, tc = UIStrings.OVERLAY_TWEAK_ADD, (1.0, 0.6, 0.0, settings.overlay_opacity)
        elif active_pins == 1: title, tc = UIStrings.OVERLAY_TWEAK_PAN, (0.4, 0.8, 1.0, settings.overlay_opacity)
        elif active_pins == 2: title, tc = UIStrings.OVERLAY_TWEAK_ORBIT, (0.6, 1.0, 0.6, settings.overlay_opacity)
        else: title, tc = UIStrings.OVERLAY_TWEAK_FULL.format(count=active_pins), (1.0, 0.2, 0.2, settings.overlay_opacity)
            
        m_txt = target_obj.name if target_obj else "Camera"
        sub = UIStrings.OVERLAY_TWEAK_SUB.format(target=m_txt)
        _draw_txt_centered(title, 60, 20, tc)
        
        sc = (1.0, 1.0, 1.0, 0.9 * settings.overlay_opacity)
        _draw_txt_centered(sub, 35, 14, sc)
        
    if target_data.last_error and not cam_data.is_tweak_mode:
        ec = (1.0, 0.2, 0.2, settings.overlay_opacity)
        _draw_txt_centered(target_data.last_error, 15, 14, ec)

    blf.size(0, settings.text_size)

    draw_name = settings.show_name_tweak if cam_data.is_tweak_mode else settings.show_name_solve
    bounds = get_camera_frame_bounds(context, cam_region, cam_rv3d)
    camintr, distcoef, res_x, res_y = get_cv_camera_params(context, cam_data)

    num_pins = len(pins)
    draw_order = [i for i in range(num_pins) if i != active_idx]
    if 0 <= active_idx < num_pins:
        draw_order.append(active_idx)

    for i in draw_order:
        pin = pins[i]
        
        is_active = False
        if cam_data.ui_mode == 'LAYOUT':
            if cam_data.is_edit_mode:
                is_active = True
            elif cam_data.is_tweak_mode and pin.use_tweak:
                is_active = True
            elif not cam_data.is_tweak_mode and pin.use_initial:
                is_active = True
        else: # MATCHMOVE
            is_active = pin.use_initial
            
        if not is_active: continue
        
        p2d = get_current_pin_pos_2d(context, cam_data, pin)
        has_2d = (p2d is not None)
        has_3d = pin.has_valid_3d
        
        if cam_data.ui_mode == 'MATCHMOVE' and not has_2d and not has_3d:
            continue
            
        is_selected = (i == active_idx)
        
        color = list(pin.color)
        text_color = list(settings.text_color) if settings.text_use_custom_color else list(pin.color)
        
        opacity_mult = 1.0 if is_selected else 0.5
        
        if not has_3d and not is_selected:
            color = [1, 1, 1, 0.4 * settings.overlay_opacity]
            if not settings.text_use_custom_color:
                text_color = [1, 1, 1, 0.5 * settings.overlay_opacity]
            else:
                text_color[3] = 0.5 * settings.overlay_opacity
        else:
            color[3] = opacity_mult * settings.overlay_opacity
            text_color[3] = opacity_mult * settings.overlay_opacity
            
        current_radius = int(settings.pin_radius * 1.2) if is_selected else settings.pin_radius
        
        c2d_3dpin = location_3d_to_region_2d(region, rv3d, Vector(pin.pos_3d)) if has_3d else None
        c2d_2dpin_display = None

        if is_cam_view and cam_region and cam_rv3d and has_2d:
            if cam_data.use_distortion_overlay:
                c2d_2dpin_display = _get_undistorted_2d_coords_cached(p2d, bounds, camintr, distcoef, res_x, res_y)
            else:
                px = get_pin_pixel_coords(context, p2d, bounds=bounds)
                c2d_2dpin_display = Vector(px) if px else None
            
            if c2d_2dpin_display:
                if not cam_data.is_tweak_mode or cam_data.is_edit_mode:
                    draw_shape(c2d_2dpin_display.x, c2d_2dpin_display.y, current_radius, color, 'CIRCLE')
                    if draw_name:
                        _draw_txt(f"2D: {pin.name}", c2d_2dpin_display.x + 10, c2d_2dpin_display.y + 10, settings.text_size, text_color)

        if c2d_3dpin:
            draw_shape(c2d_3dpin.x, c2d_3dpin.y, current_radius, color, 'SQUARE')
            if draw_name:
                text_3d = f"3D: {pin.name}"
                if settings.show_weight_3d and not pin.is_track_linked:
                    text_3d += f" [W:{pin.weight:.2f}]"
                if settings.show_error_3d and not cam_data.is_tweak_mode and pin.reproj_error >= 0:
                    if cam_data.ui_mode == 'LAYOUT':
                        text_3d += f" [Err:{pin.reproj_error:.2f}]"
                    else:
                        text_3d += f" [Err:{pin.reproj_error:.2f}]"
                
                _draw_txt(text_3d, c2d_3dpin.x + 10, c2d_3dpin.y - 15, settings.text_size, text_color)

        if is_cam_view and c2d_3dpin and c2d_2dpin_display and not cam_data.is_tweak_mode:
            err = pin.reproj_error
            if err >= 0 and sum(1 for p in pins if getattr(p, 'use_initial') and p.has_valid_3d) >= 3:
                r, g = min(1.0, err / 15.0), max(0.0, 1.0 - (err / 15.0))
                line_color = (r, g, 0.0, settings.line_opacity * settings.overlay_opacity)
                draw_line(c2d_3dpin.x, c2d_3dpin.y, c2d_2dpin_display.x, c2d_2dpin_display.y, line_color, settings.line_width)

    blf.shadow(0, 0, 0.0, 0.0, 0.0, 0.0)

# ==========================================
# 5. UI Panels
# ==========================================
class PINSOLVER_UL_targets(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, "obj", text="")

class PINSOLVER_UL_pins(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        split_main = layout.split(factor=0.6, align=True)
        
        r_left = split_main.split(factor=0.15, align=True)
        r_left.prop(item, "color", text="")
        
        r_name = r_left.row(align=True)
        if not item.has_valid_3d:
            r_name.label(text="", icon='ERROR')
        r_name.prop(item, "name", text="", emboss=False)
        
        r_right = split_main.row(align=True)
        r_right.prop(item, "weight", text="")
        
        op_solve = r_right.operator("view3d.pinsolver_toggle_pin_mode", text="", icon='CON_TRACKTO', depress=item.use_initial)
        op_solve.mode = 'SOLVE'
        op_solve.index = index
        
        op_tweak = r_right.operator("view3d.pinsolver_toggle_pin_mode", text="", icon='VIEW_PAN', depress=item.use_tweak)
        op_tweak.mode = 'TWEAK'
        op_tweak.index = index

class PINSOLVER_UL_mm_pins(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        camera = context.scene.camera
        is_active_tracker = True
        if camera and hasattr(camera, "pinsolver_data"):
            cam_data = camera.pinsolver_data
            p2d = get_current_pin_pos_2d(context, cam_data, item)
            is_active_tracker = (p2d is not None)

        split_main = layout.split(factor=0.55, align=True)
        
        r_left = split_main.split(factor=0.2, align=True)
        r_left.enabled = is_active_tracker 
        r_left.prop(item, "color", text="")
        
        r_name = r_left.row(align=True)
        if not item.has_valid_3d:
            r_name.label(text="", icon='ERROR')
        r_name.prop(item, "name", text="", emboss=False)
        
        r_right = split_main.row(align=True)
        
        r_center = r_right.row(align=True)
        r_center.enabled = is_active_tracker
        r_center.prop(item, "weight", text="")
        if item.reproj_error >= 0:
            r_center.label(text=f"{item.reproj_error:.1f}")
        else:
            r_center.label(text="")
            
        icon_str = 'HIDE_OFF' if item.use_initial else 'HIDE_ON'
        r_right.prop(item, "use_initial", text="", icon=icon_str, emboss=False)

class PINSOLVER_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PinSolver'; bl_label = "PinSolver"

    @staticmethod
    def draw_calib_row(parent, p_data, p_name, t_data, t_prop, label, can_calib):
        r = parent.row(align=True); split = r.split(factor=0.15, align=True)
        c = split.column(align=True); c.enabled = can_calib; c.prop(p_data, p_name, text="")
        split.prop(t_data, t_prop, text=label)

    def draw(self, context):
        layout = self.layout
        if not HAS_OPENCV:
            layout.alert = True; layout.label(text="OpenCV Required!", icon='ERROR'); layout.alert = False
            layout.label(text="Ensure 'opencv-python' wheel is installed", icon='INFO')
            return

        camera = context.scene.camera
        layout.row().prop(context.scene, "camera", text="", icon='OUTLINER_OB_CAMERA')
        if not camera or camera.type != 'CAMERA': layout.label(text="Please set an active Camera", icon='INFO'); return

        cam_data, target_data, target_obj = get_active_target_data(context)
        
        layout.row().prop(cam_data, "ui_mode", expand=True)
        layout.separator()
        
        split = layout.split(factor=0.45)
        r1 = split.row(align=True)
        r1.operator("view3d.pinsolver_toggle_show_pins", text="Show Pins", icon='RESTRICT_VIEW_OFF' if cam_data.show_overlays else 'RESTRICT_VIEW_ON', depress=cam_data.show_overlays)
        r1.prop(cam_data, "use_distortion_overlay", text="", icon='OUTLINER_OB_CAMERA')
        
        if target_data:
            if cam_data.ui_mode == 'LAYOUT':
                err_val = target_data.layout_avg_error
                if err_val >= 0 and not cam_data.is_tweak_mode:
                    split.label(text=f"Avg Err: {err_val:.1f} px")
                else:
                    split.label(text="Frame Err: ---")
            else:
                err_val = target_data.mm_avg_error
                if err_val >= 0 and not cam_data.is_tweak_mode:
                    split.label(text=f"Seq Avg Err: {err_val:.1f} px")
                else:
                    split.label(text="Seq Avg Err: ---")
        else:
            if cam_data.ui_mode == 'LAYOUT':
                split.label(text="Frame Err: ---")
            else:
                split.label(text="Seq Avg Err: ---")
            
        layout.separator()
        col = layout.column(align=True); col.label(text="Solve Target:", icon='CON_TRACKTO')
        
        row_target = col.row(align=True)
        for t_id, t_name in [('CAMERA', "Camera"), ('PARENT', "Parent"), ('OBJECT', "Object")]:
            op = row_target.operator("view3d.pinsolver_set_solve_target", text=t_name, depress=(cam_data.solve_mode == t_id))
            op.target = t_id
        
        if cam_data.solve_mode == 'OBJECT':
            box = layout.box(); row = box.row()
            row.template_list("PINSOLVER_UL_targets", "", cam_data, "target_objects", cam_data, "active_target_index", rows=3)
            cb = row.column(align=True); cb.operator("view3d.pinsolver_add_target", icon='ADD', text="")
            if cam_data.target_objects: cb.operator("view3d.pinsolver_remove_target", icon='REMOVE', text="")
            if not target_obj: layout.label(text="Add/Select an Object to align", icon='INFO')

        layout.separator()
        
        if cam_data.ui_mode == 'MATCHMOVE':
            box_mm = layout.box()
            box_mm.prop(cam_data, "target_clip", text="Tracker Clip")
            if cam_data.target_clip:
                box_mm.prop(cam_data, "tracking_object_idx", text="Track Layer")
            box_mm.operator("view3d.pinsolver_sync_trackers", icon='TRACKING')
            box_mm.operator("view3d.pinsolver_auto_raycast", icon='CON_TRACKTO')
            layout.separator()

        if target_data:
            pins, active_idx = get_pins(cam_data, target_data)
            
            if cam_data.ui_mode == 'LAYOUT':
                row = layout.row()
                row.template_list("PINSOLVER_UL_pins", "", target_data, "layout_pins", target_data, "layout_pin_idx", rows=5)
                cp = row.column(align=True)
                cp.operator("view3d.pinsolver_add_pin", icon='ADD', text="")
                if pins: 
                    cp.operator("view3d.pinsolver_remove_pin", icon='REMOVE', text="")
                    cp.separator()
                    cp.operator("view3d.pinsolver_clear_pins", icon='TRASH', text="")
                    
            elif cam_data.ui_mode == 'MATCHMOVE':
                row = layout.row()
                row.template_list("PINSOLVER_UL_mm_pins", "", target_data, "mm_pins", target_data, "mm_pin_idx", rows=5)
                cp = row.column(align=True)
                if pins:
                    cp.operator("view3d.pinsolver_clear_pins", icon='TRASH', text="")

            if pins and active_idx < len(pins):
                pin = pins[active_idx]
                box = layout.box()
                
                if not pin.has_valid_3d:
                    box.label(text="Missing 3D Position (Raycast Needed)", icon='ERROR')
                else:
                    err_icon = 'CHECKMARK' if pin.reproj_error <= PinSolverConfig.REPROJ_ERROR_GOOD_THRESHOLD_PX else 'ERROR'
                    
                    if cam_data.ui_mode == 'LAYOUT':
                        err_str = f" | Err: {pin.reproj_error:.2f}px" if pin.reproj_error >= 0 and not cam_data.is_tweak_mode else ""
                        box.label(text=f"{pin.name}{err_str}", icon=err_icon if pin.reproj_error >= 0 else 'LAYER_ACTIVE')
                    else:
                        err_str = f" | Max Err: {pin.reproj_error:.2f}px" if pin.reproj_error >= 0 and not cam_data.is_tweak_mode else ""
                        box.label(text=f"Track: {pin.track_name}{err_str}", icon='TRACKING')

                box.prop(pin, "weight")

                if cam_data.ui_mode == 'LAYOUT':
                    cd = box.column(align=True)
                    r2 = cd.row(align=True)
                    r2.label(text="2D(●):")
                    op2d = r2.operator("view3d.pinsolver_pick_2d", text="Pick 2D", icon='EYEDROPPER', depress=(target_data.picking_state=='PICK_2D'))
                    op2d.target_index = active_idx
                    r3 = cd.row(align=True)
                    r3.label(text="3D(■):")
                    op3d = r3.operator("view3d.pinsolver_pick_3d", text="Pick 3D", icon='EYEDROPPER', depress=(target_data.picking_state=='PICK_3D'))
                    op3d.target_index = active_idx
                else:
                    cd = box.column(align=True)
                    r3 = cd.row(align=True)
                    r3.label(text="3D(■):")
                    op3d = r3.operator("view3d.pinsolver_pick_3d", text="Pick", icon='EYEDROPPER', depress=(target_data.picking_state=='PICK_3D'))
                    op3d.target_index = active_idx
                    r3.operator("view3d.pinsolver_auto_raycast_single", text="Raycast", icon='CON_TRACKTO')

                r_coords = box.row(align=True)
                r_coords.prop(target_data, "show_pin_details", text="Manual Coordinates", icon='TRIA_DOWN' if target_data.show_pin_details else 'TRIA_RIGHT', emboss=False)
                if target_data.show_pin_details:
                    inn = box.column(align=True)
                    if cam_data.ui_mode == 'LAYOUT':
                        inn.prop(pin, "pos_2d", text="2D")
                    inn.prop(pin, "pos_3d", text="3D")

            layout.separator()
            if target_data.last_error and not cam_data.is_tweak_mode:
                layout.label(text=target_data.last_error, icon='ERROR')
                
            ca = layout.column(align=True); ca.scale_y = 1.3
            
            if cam_data.ui_mode == 'LAYOUT':
                if cam_data.is_edit_mode:
                    ca.operator("view3d.pinsolver_edit_pins", icon='CHECKMARK', text="Finish Editing (ESC/Shortcut)", depress=True)
                else:
                    ca.operator("view3d.pinsolver_edit_pins", icon='EDITMODE_HLT', text="Interactive Pin Editor")
                    
                ca.separator()
                
                valid_pin_count = sum(1 for p in pins if p.use_initial and p.has_valid_3d)
                r_solve = ca.row()
                r_solve.enabled = (valid_pin_count > 0)
                r_solve.operator("view3d.pinsolver_solve", icon='PLAY', text="Solve Alignment")
                
                if cam_data.is_tweak_mode: 
                    ca.operator("view3d.pinsolver_tweak", icon='PAUSE', text="Stop Tweak Mode (ESC/Shortcut)", depress=True)
                else: 
                    ca_twk = ca.row()
                    ca_twk.operator("view3d.pinsolver_tweak", icon='VIEW_PAN', text="Interactive Tweak (Solve)")
                    
                row_planar = layout.row()
                row_planar.prop(cam_data, "use_planar_solve", text="Planar Mode (Flat Surface)")
                if cam_data.use_planar_solve:
                    box_warn = layout.box()
                    box_warn.label(text="⚠️ All 3D pins must be on a perfectly flat plane.", icon='INFO')
                    if valid_pin_count < 4:
                        box_warn.label(text="Planar Mode requires 4+ active pins!", icon='ERROR')
                    
            elif cam_data.ui_mode == 'MATCHMOVE':
                r_actions = ca.row(align=True)
                if cam_data.is_edit_mode:
                    r_actions.operator("view3d.pinsolver_edit_pins", icon='CHECKMARK', text="Finish Edit", depress=True)
                else:
                    r_actions.operator("view3d.pinsolver_edit_pins", icon='EDITMODE_HLT', text="3D Pin Editor")
                r_actions.operator("view3d.pinsolver_send_to_layout", icon='FORWARD', text="Send to Layout")
                
                layout.separator()
                layout.label(text="Sequence Solver", icon='CAMERA_DATA')
                
                col_bake = layout.column(align=True)
                
                col_bake.row(align=True).prop(cam_data, "bake_target", expand=True)
                if cam_data.bake_target == 'SCENE':
                    col_bake.label(text=f"Range: Frame {context.scene.frame_start} to {context.scene.frame_end}")
                else:
                    col_bake.label(text="Range: Timeline Markers")
                    
                ref_row = col_bake.row(align=True)
                ref_row.prop(cam_data, "reference_frame")
                ref_row.operator("view3d.pinsolver_set_reference_frame", text="", icon='TIME')
                
                opt_row = col_bake.row(align=True)
                opt_row.prop(cam_data, "show_sequence_options", text="Options", icon='TRIA_DOWN' if cam_data.show_sequence_options else 'TRIA_RIGHT', emboss=False)
                if cam_data.show_sequence_options:
                    opt_box = col_bake.column(align=True)
                    opt_box.prop(cam_data, "sequence_motion_source", text="Location")
                    if cam_data.sequence_motion_source == 'MARKER_REFS' and cam_data.bake_target != 'SCENE':
                        opt_box.label(text="Timeline Markers source requires Scene Range", icon='ERROR')
                    if cam_data.sequence_motion_source in {'PNP', 'GUIDED', 'MARKER_REFS'}:
                        if cam_data.sequence_motion_source == 'GUIDED':
                            opt_box.label(text="Existing curve guides solve and final location")
                            opt_box.prop(cam_data, "sequence_guided_location_strength", slider=True)
                        elif cam_data.sequence_motion_source == 'MARKER_REFS':
                            opt_box.label(text="Marker frames stay fixed; neighbors are blended")
                        else:
                            opt_box.label(text="Solves location from active pins")
                        opt_box.separator()
                        opt_box.label(text="Location Filter")
                        filter_row = opt_box.row(align=True)
                        filter_row.prop(cam_data, "sequence_stabilize_mode", expand=True)
                        if cam_data.sequence_stabilize_mode != 'OFF':
                            opt_box.prop(cam_data, "sequence_location_filter_strength", slider=True)
                        if cam_data.sequence_motion_source == 'MARKER_REFS':
                            opt_box.label(text="Filter also eases into fixed marker frames")
                    else:
                        opt_box.label(text="Keeps existing location; solves rotation from pins")
                    opt_box.separator()
                    opt_box.label(text="Roll Smoothing")
                    opt_box.row(align=True).prop(cam_data, "sequence_roll_smoothing", expand=True)
                    if cam_data.sequence_roll_smoothing != 'OFF':
                        opt_box.prop(cam_data, "sequence_roll_smoothing_strength", slider=True)
                 
                col_bake.separator()
                 
                valid_pin_count = sum(1 for p in pins if p.use_initial and p.has_valid_3d)
                min_sequence_pins = 2 if cam_data.sequence_motion_source == 'EXISTING' else 4
                marker_refs_invalid = cam_data.sequence_motion_source == 'MARKER_REFS' and cam_data.bake_target != 'SCENE'

                r_single = col_bake.row()
                r_single.enabled = (valid_pin_count > 0)
                r_single.operator("view3d.pinsolver_solve", icon='PLAY', text="Single Solve (Current Frame)")
                 
                r_bake = col_bake.row()
                r_bake.scale_y = 2.0
                r_bake.enabled = (valid_pin_count >= min_sequence_pins and not marker_refs_invalid)
                if marker_refs_invalid:
                    r_bake.operator("view3d.pinsolver_bake_animation", icon='ERROR', text="Timeline Markers Requires Scene Range")
                elif valid_pin_count < min_sequence_pins:
                    if cam_data.sequence_motion_source == 'EXISTING':
                        r_bake.operator("view3d.pinsolver_bake_animation", icon='ERROR', text="Need 2+ Valid Pins for Existing Location")
                    else:
                        r_bake.operator("view3d.pinsolver_bake_animation", icon='ERROR', text="Need 4+ Valid Pins for Sequence")
                else:
                    r_bake.operator("view3d.pinsolver_bake_animation", icon='REC')

                row_planar = layout.row()
                row_planar.prop(cam_data, "use_planar_solve", text="Planar Mode (Flat Surface)")
                if cam_data.use_planar_solve:
                    box_warn = layout.box()
                    box_warn.label(text="⚠️ All 3D pins must be on a perfectly flat plane.", icon='INFO')
                    if valid_pin_count < 4:
                        box_warn.label(text="Planar Mode requires 4+ active pins!", icon='ERROR')

        layout.separator()
        
        layout.row(align=True).prop(cam_data, "show_calibration", text="Lens Calibration", icon='TRIA_DOWN' if cam_data.show_calibration else 'TRIA_RIGHT', emboss=False)
        
        if target_data:
            pins, _ = get_pins(cam_data, target_data)
            if cam_data.is_tweak_mode:
                ap = sum(1 for p in pins if getattr(p, 'use_tweak') and p.has_valid_3d)
            else:
                ap = sum(1 for p in pins if getattr(p, 'use_initial') and p.has_valid_3d)
        else:
            ap = 0
            
        cc = ap >= PinSolverConfig.MIN_PINS_FOR_CALIBRATION

        has_pinned = False
        if cam_data.calib_focal_length or cam_data.pin_disp_focal: has_pinned = True
        if cam_data.calib_optical_center or cam_data.pin_disp_center: has_pinned = True
        if cam_data.calib_k1 or cam_data.calib_k2 or cam_data.calib_k3 or cam_data.pin_disp_dist: has_pinned = True

        if cam_data.show_calibration:
            inn = layout.column()
            
            if cam_data.ui_mode == 'MATCHMOVE':
                inn.label(text="Animation Settings:")
                r_anim = inn.row()
                r_anim.prop(cam_data, "calib_animation_mode", expand=True)
                
                if cam_data.calib_animation_mode == 'STATIC':
                    inn.prop(cam_data, "calib_static_method", text="")
                else:
                    r_z = inn.row(align=True)
                    btn_text = "Zooming (Dynamic)" if cam_data.use_dynamic_zoom else "Static Lens"
                    btn_icon = 'VIEWZOOM' if cam_data.use_dynamic_zoom else 'PAUSE'
                    r_z.prop(cam_data, "use_dynamic_zoom", text=btn_text, toggle=True, icon=btn_icon)
                    r_z.prop_decorator(cam_data, "use_dynamic_zoom")
                inn.separator()
            elif cam_data.ui_mode == 'LAYOUT':
                inn.prop(cam_data, "target_clip", text="Clip")
                inn.separator()
            
            if camera.data:
                inn.prop(camera.data, "sensor_width", text="Sensor Size (mm)")
                inn.separator()
                
                is_zoom_mode = (cam_data.ui_mode == 'MATCHMOVE' and cam_data.calib_animation_mode == 'ZOOM')
                
                r_fleft = inn.row(align=True)
                sp_f = r_fleft.split(factor=0.9, align=True)
                sp_f.label(text="Focal Length:" if cc else f"Focal Length: (Locked {PinSolverConfig.MIN_PINS_FOR_CALIBRATION}+ Pins)", icon='NONE' if cc else 'LOCKED')
                sp_f.prop(cam_data, "pin_disp_focal", text="", icon='PINNED' if cam_data.pin_disp_focal else 'UNPINNED', emboss=False)
                
                r = inn.row(align=True); sp = r.split(factor=0.15, align=True)
                c = sp.column(align=True); c.enabled = cc; c.prop(cam_data, "calib_focal_length", text="")
                sp.prop(camera.data, "lens", text="mm")
                inn.separator()
                
                r_fleft = inn.row(align=True)
                sp_f = r_fleft.split(factor=0.9, align=True)
                sp_f.label(text="Optical Center (Shift):" if cc else "Optical Center: (Locked)", icon='NONE' if cc else 'LOCKED')
                sp_f.prop(cam_data, "pin_disp_center", text="", icon='PINNED' if cam_data.pin_disp_center else 'UNPINNED', emboss=False)
                
                r = inn.row(align=True); sp = r.split(factor=0.15, align=True)
                cbc = sp.column(align=True); cbc.enabled = cc; cbc.prop(cam_data, "calib_optical_center", text="")
                sr = sp.row(align=True); sr.prop(camera.data, "shift_x", text="X"); sr.prop(camera.data, "shift_y", text="Y")
                inn.separator()

                r_fleft = inn.row(align=True)
                sp_f = r_fleft.split(factor=0.9, align=True)
                sp_f.label(text="Distortion Model: Polynomial" if cc else "Distortion: (Locked)", icon='NONE' if cc else 'LOCKED')
                sp_f.prop(cam_data, "pin_disp_dist", text="", icon='PINNED' if cam_data.pin_disp_dist else 'UNPINNED', emboss=False)
                
                dc = inn.column(align=True)
                has_clip = bool(cam_data.target_clip)
                
                dc.enabled = has_clip and cc and not is_zoom_mode
                
                if not has_clip:
                    r = dc.row(align=True); sp = r.split(factor=0.15, align=True); c = sp.column(align=True); c.prop(cam_data, "calib_k1", text=""); sp.prop(cam_data, "dummy_k", text="K1")
                    r = dc.row(align=True); sp = r.split(factor=0.15, align=True); c = sp.column(align=True); c.prop(cam_data, "calib_k2", text=""); sp.prop(cam_data, "dummy_k", text="K2")
                    r = dc.row(align=True); sp = r.split(factor=0.15, align=True); c = sp.column(align=True); c.prop(cam_data, "calib_k3", text=""); sp.prop(cam_data, "dummy_k", text="K3")
                else:
                    tc = cam_data.target_clip.tracking.camera
                    r = dc.row(align=True); sp = r.split(factor=0.15, align=True); c = sp.column(align=True); c.prop(cam_data, "calib_k1", text=""); sp.prop(tc, "k1", text="K1")
                    r = dc.row(align=True); sp = r.split(factor=0.15, align=True); c = sp.column(align=True); c.prop(cam_data, "calib_k2", text=""); sp.prop(tc, "k2", text="K2")
                    r = dc.row(align=True); sp = r.split(factor=0.15, align=True); c = sp.column(align=True); c.prop(cam_data, "calib_k3", text=""); sp.prop(tc, "k3", text="K3")
                
                if is_zoom_mode:
                    inn.label(text="* Blender does not support animated distortion.", icon='INFO')
                    
                inn.separator()
                inn.operator("view3d.pinsolver_sync_clip", text="Sync Lens to Clip", icon='FILE_REFRESH')

        elif has_pinned and camera.data:
            inn = layout.column()
            
            is_zoom_mode = (cam_data.ui_mode == 'MATCHMOVE' and cam_data.calib_animation_mode == 'ZOOM')
            
            if cam_data.calib_focal_length or cam_data.pin_disp_focal:
                r = inn.row(align=True)
                sp = r.split(factor=0.15, align=True)
                c = sp.column(align=True); c.enabled = cc; c.prop(cam_data, "calib_focal_length", text="")
                sp.prop(camera.data, "lens", text="Focal Length (mm)")
                
            if cam_data.calib_optical_center or cam_data.pin_disp_center:
                r = inn.row(align=True)
                sp = r.split(factor=0.15, align=True)
                cbc = sp.column(align=True); cbc.enabled = cc; cbc.prop(cam_data, "calib_optical_center", text="")
                sr = sp.row(align=True); sr.prop(camera.data, "shift_x", text="Shift X"); sr.prop(camera.data, "shift_y", text="Shift Y")
                
            if cam_data.calib_k1 or cam_data.calib_k2 or cam_data.calib_k3 or cam_data.pin_disp_dist:
                dc = inn.column(align=True)
                has_clip = bool(cam_data.target_clip)
                tc = cam_data.target_clip.tracking.camera if has_clip else None
                
                dc.enabled = has_clip and cc and not is_zoom_mode
                
                if cam_data.calib_k1 or cam_data.pin_disp_dist:
                    r = dc.row(align=True)
                    sp = r.split(factor=0.15, align=True)
                    c = sp.column(align=True); c.enabled = has_clip and cc and not is_zoom_mode; c.prop(cam_data, "calib_k1", text="")
                    if has_clip: sp.prop(tc, "k1", text="Distortion K1")
                    else: sp.prop(cam_data, "dummy_k", text="Distortion K1")
                if cam_data.calib_k2 or cam_data.pin_disp_dist:
                    r = dc.row(align=True)
                    sp = r.split(factor=0.15, align=True)
                    c = sp.column(align=True); c.enabled = has_clip and cc and not is_zoom_mode; c.prop(cam_data, "calib_k2", text="")
                    if has_clip: sp.prop(tc, "k2", text="Distortion K2")
                    else: sp.prop(cam_data, "dummy_k", text="Distortion K2")
                if cam_data.calib_k3 or cam_data.pin_disp_dist:
                    r = dc.row(align=True)
                    sp = r.split(factor=0.15, align=True)
                    c = sp.column(align=True); c.enabled = has_clip and cc and not is_zoom_mode; c.prop(cam_data, "calib_k3", text="")
                    if has_clip: sp.prop(tc, "k3", text="Distortion K3")
                    else: sp.prop(cam_data, "dummy_k", text="Distortion K3")

        layout.row(align=True).prop(cam_data, "show_settings", text="PinSolver Settings", icon='TRIA_DOWN' if cam_data.show_settings else 'TRIA_RIGHT', emboss=False)
        if cam_data.show_settings:
            settings = context.scene.pinsolver_settings
            col = layout.column(align=True)
            col.prop(settings, "lock_camera_z")
            col.separator()
            col.prop(settings, "show_name_solve")
            col.prop(settings, "show_name_tweak")
            col.prop(settings, "show_error_3d")
            col.prop(settings, "show_weight_3d")
            col.separator()
            col.prop(settings, "text_use_custom_color")
            if settings.text_use_custom_color:
                col.prop(settings, "text_color")
            col.prop(settings, "text_use_outline")
            if settings.text_use_outline:
                col.prop(settings, "text_outline_color")
            col.separator()
            col.prop(settings, "pin_radius")
            col.prop(settings, "text_size")
            col.prop(settings, "line_width")
            col.prop(settings, "line_opacity")
            col.prop(settings, "overlay_opacity")
            col.separator()
            col.prop(settings, "add_pin_offset")

# ==========================================
# 6. Registration & State Management
# ==========================================
class PinSolverAddon:
    @classmethod
    def register(cls):
        for c in classes: bpy.utils.register_class(c)
        bpy.types.Object.pinsolver_data = bpy.props.PointerProperty(type=PinSolverData)
        bpy.types.Scene.pinsolver_settings = bpy.props.PointerProperty(type=PinSolverSettings)

    @classmethod
    def unregister(cls):
        global _draw_handle
        if _draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
            _draw_handle = None
            
        for c in reversed(classes): bpy.utils.unregister_class(c)
        del bpy.types.Object.pinsolver_data
        del bpy.types.Scene.pinsolver_settings

classes = (
    PinSolverSettings, PinSolverPin, PinSolverTargetItem, PinSolverData, 
    PINSOLVER_OT_toggle_show_pins, PINSOLVER_OT_set_solve_target, PINSOLVER_OT_toggle_pin_mode,
    PINSOLVER_UL_targets, PINSOLVER_UL_pins, PINSOLVER_UL_mm_pins, PINSOLVER_OT_add_target, PINSOLVER_OT_remove_target,
    PINSOLVER_OT_pick_2d, PINSOLVER_OT_pick_3d, PINSOLVER_OT_auto_raycast_single, PINSOLVER_OT_sync_clip, PINSOLVER_OT_clear_pins,
    PINSOLVER_OT_add_pin, PINSOLVER_OT_remove_pin, PINSOLVER_OT_solve, 
    PINSOLVER_OT_edit_pins, PINSOLVER_OT_tweak, PINSOLVER_PT_panel,
    PINSOLVER_OT_send_to_layout, PINSOLVER_OT_sync_trackers, PINSOLVER_OT_auto_raycast, PINSOLVER_OT_set_reference_frame,
    PINSOLVER_OT_bake_animation
)

def register(): PinSolverAddon.register()
def unregister(): PinSolverAddon.unregister()
