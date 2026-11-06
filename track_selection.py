"""Active Matchmove pin / Clip track selection sync, without cached RNA references."""
import bpy


_updating = False


def _binding(context):
    from . import get_active_target_data

    scene = context.scene
    if scene is None or not scene.pinsolver_settings.sync_track_selection:
        return None
    cam_data, target_data, _ = get_active_target_data(context)
    if cam_data is None or target_data is None or cam_data.ui_mode != 'MATCHMOVE' or not cam_data.show_overlays:
        return None
    clip = cam_data.target_clip
    if clip is None:
        return None
    index = int(cam_data.tracking_object_idx)
    if not 0 <= index < len(clip.tracking.objects):
        return None
    layer = clip.tracking.objects[index]
    active_layer = clip.tracking.objects.active
    if active_layer is None or active_layer.as_pointer() != layer.as_pointer():
        return None
    return target_data, layer.tracks


def _redraw(scene):
    for window in bpy.context.window_manager.windows:
        if window.scene == scene:
            for area in window.screen.areas:
                if area.type in {'VIEW_3D', 'CLIP_EDITOR'}:
                    area.tag_redraw()


def update_pin_index(self, context):
    global _updating
    if _updating:
        return
    try:
        binding = _binding(context)
        if binding is None:
            return
        target_data, tracks = binding
        if self.as_pointer() != target_data.as_pointer():
            return
        index = target_data.mm_pin_idx
        if not 0 <= index < len(target_data.mm_pins):
            return
        pin = target_data.mm_pins[index]
        track = tracks.get(pin.track_name) if pin.is_track_linked else None
        if track is None:
            return
        _updating = True
        for other in tracks:
            if other.as_pointer() != track.as_pointer():
                other.select = False
        track.select = True
        tracks.active = track
        _redraw(context.scene)
    except (ReferenceError, RuntimeError, AttributeError, ValueError, TypeError):
        # A joined/deleted track or a scene change may invalidate the binding.
        return
    finally:
        _updating = False


def sync_from_clip(context):
    global _updating
    if _updating:
        return
    try:
        binding = _binding(context)
        if binding is None:
            return
        target_data, tracks = binding
        active = tracks.active
        if active is None or not active.select:
            return
        name = active.name
        index = next((i for i, pin in enumerate(target_data.mm_pins)
            if pin.is_track_linked and pin.track_name == name), None)
        if index is None or index == target_data.mm_pin_idx:
            return
        _updating = True
        target_data.mm_pin_idx = index
        _redraw(context.scene)
    except (ReferenceError, RuntimeError, AttributeError, ValueError, TypeError):
        return
    finally:
        _updating = False


def poll_selection():
    if bpy.app.is_job_running('RENDER'):
        return .15
    seen_scenes = set()
    for window in list(bpy.context.window_manager.windows):
        try:
            scene_id = window.scene.as_pointer()
            if scene_id in seen_scenes:
                continue
            seen_scenes.add(scene_id)
            with bpy.context.temp_override(window=window):
                sync_from_clip(bpy.context)
        except (ReferenceError, RuntimeError):
            continue
    return .15


def register():
    if not bpy.app.timers.is_registered(poll_selection):
        bpy.app.timers.register(poll_selection, first_interval=.15, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(poll_selection):
        bpy.app.timers.unregister(poll_selection)
