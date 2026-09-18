## Overview
PinSolver is a Blender add-on that leverages OpenCV's powerful PnP (Perspective-n-Point) algorithm to provide intuitive, fast camera and object alignment. By linking 2D coordinates (●) on an image to 3D coordinates (■) in 3D space, it seamlessly supports everything from static object placement to full-fledged video matchmoving. 

## Prerequisites
* **Accurately Scaled Models:** To achieve high-precision alignment and matchmoving, it is absolutely critical that the 3D models or scan data placed in your scene match real-world proportions and dimensions. Incorrect model scale will yield inaccurate calculation results.

## Use Cases
* **VFX Compositing, Layout, and Survey Work for Live-Action:**
  Useful for camera layouts against background models, aligning with scan data, and placing props. It also offers a streamlined matchmoving feature using PnP to perform camera solves based on Blender's tracking data.
* **Camera projection. Texture projection.**
* **Aligning architectural renderings**
* **Aligning models with reference photos**
* **Lens Parameter Estimation:**
  Even if the camera metadata is unknown, PinSolver automatically estimates the focal length, optical center, and lens distortion based on the spatial relationship between the pins and the 3D model. Matchmove mode also supports variable zoom. *(Note: Variable animated distortion is not supported due to Blender's internal constraints.)*

## Features

### 🔀 Dual Workflow Modes
PinSolver includes two distinct modes depending on your objective:
* **Layout Mode:** Dedicated to single-frame static camera matching and prop placement.
* **Matchmove Mode:** Integrates with Blender's 2D tracking data to dynamically bake camera or object animations.  

### 🎯 Core PnP Solver
* **1-Pin (Raycast):** Instantly snaps the target along the camera's ray (line of sight) using a single point.
* **2-Pin (Pan/Orbit):** Calculates basic orientation and translation using two points.
* **3+ Pins (Full 6-DoF):** Calculates pixel-perfect, full 3D position and rotation (6 Degrees of Freedom) using three or more points.
* **Weighting System:** Fine-tune the priority of each pin using an intuitive 0.00–1.00 slider to strictly control calculation priorities.

### 🔄 Versatile Target Modes
* **Camera Mode:** Moves the active camera's position and rotation to match the background.
* **Parent Mode:** Moves the camera's parent object instead. Ideal when the camera utilizes a complex rig, has existing constraints, or already contains animation. 
* **Object Mode:** Reverses the math. It keeps the camera fixed and moves the "selected 3D object" to match the camera's perspective. Selecting a parent object allows for moving multiple objects simultaneously.

### ✋ Interactive Workflow (Undo Supported)
* **Interactive Tweak (Real-time Solve):** Dragging a 3D pin (■) directly in the camera view recalculates the alignment in real-time, allowing the target to follow your mouse smoothly. If you push it too far, press `Ctrl+Z` (Undo) to instantly revert to the previous pose.
* **Interactive Pin Editor:** Safely add, delete, and fine-tune pins across split viewports without triggering a solve computation.
* **Clear All Pins:** A single trash can button safely and instantly resets all accumulated pins.

### ☁️ Point Cloud Support
* Point clouds are supported across Layout Mode, Matchmove Mode, and Pin Align, including direct viewport picking.

### 📐 Pin Align
* Aligns a Source mesh or point cloud to a Target mesh or point cloud from matching 3D pin pairs, with optional ICP refinement.


---

## Mode-Specific Features

### Layout Mode Features
* **Solve Alignment:** Performs standard alignment based on your designated 2D and 3D pin data.
* **Interactive Pin Editor:** Intuitively manipulate pins by dragging them directly in the 3D viewport, using keyboard shortcuts (`A` to add, `X` to delete) instead of relying on manual numeric input.
* **Tweak Mode:** Dragging a pin recalculates the camera/object position in real-time, allowing for visual fine-tuning while previewing the result.

### Matchmove Mode Features
*(Recommended for use after aligning the camera initially in Layout Mode.)*
* **Sync 2D Trackers:** Import markers tracked in Blender's Movie Clip Editor as PinSolver 2D pins with a single click. *(Note: At least 4 active trackers are required at all times.)* 
* **Auto Raycast:** Automatically shoots rays from the camera through the 2D pin coordinates to hit the 3D model's surface, acquiring and placing all 3D pins in bulk. Pins aimed at empty space are automatically excluded from the calculation. *(If you enable “Auto Raycast New Tracks,” raycasting will be performed automatically during baking.)*
* **Sequence Solver:** Solves continuously across a specified frame range, automatically baking the camera or object movement into keyframes. Supports focal length changes for zoom lenses.  
> **Disclaimer:** Please keep in mind that this is a simplified feature and might not work perfectly in all situations. However, because it leverages PnP's absolute 3D constraints, it excels in scenes with minimal parallax or subtle camera movements—exactly the kind of shots where the standard 'Solve' in the Movie Clip Editor usually fails.
---

## Important Notes
* **Calibration Requirements:** To automatically calculate Lens Calibration (focal length, optical center, etc.), you must have at least **6 active pins**.
* **Distortion Calculation Requirements:** To calculate and apply lens distortion (K1, K2, K3), a **Movie Clip** must be assigned to the **Target Clip**.  When using distortion correction, do not forget to enable **"Render Undistorted"** for the background image in the Camera View.

---

## How to Use

### Layout Mode (Static Image Alignment)
1. Set **Workflow Mode** to `Layout`.
2. Click the `+` button under the pin list to add pins (minimum 3).
3. Click `Interactive Pin Editor`. In the viewport, drag the 2D pins (●) to features on the background, and 3D pins (■) to corresponding points on the 3D model. Alternatively, use the Pick tool. Split-viewport picking is supported.
4. Check desired estimation options in the **Lens Calibration** panel if needed.
5. Click `Solve Alignment` (Play icon) to move the camera (or object) to the correct position.
6. Switch to `Interactive Tweak` for minor layout adjustments if necessary. You can also start the layout process directly using Interactive Tweak.

> **💡 Calibration Tips (Layout):**
> * A minimum of **6 active pins** is required. If the check boxes are grayed out or show a lock icon, calibration is disabled due to insufficient pins.
> * Distortion correction is only available when a Movie Clip is selected. Calibration supports the Polynomial distortion model only. Always check "Render Undistorted" for your background image in Camera View.
> * If the camera icon next to "Show Pins" (Undistort 2D Pins) is enabled, 2D pins in the viewport are automatically visually offset to match the distortion correction.
> * When using calibration inside Tweak mode, the calibration solver takes priority over simple translation.

### Matchmove Mode (Video / Multi-sequence)
1. Set **Workflow Mode** to `Matchmove`.
2. Assign tracked footage to `Tracker Clip` and click `Sync 2D Trackers` to import markers. If you add or delete trackers later, click it again to resync.
3. Move to the frame where the camera was successfully aligned in Layout mode, and click `Raycast Current Tracks` to snap 3D pins to the model's surface. Adjust pins manually if needed.   
>*💡Enabling `Auto Raycast New Tracks` causes 3DPin to automatically raycast based on the frame when the tracker first becomes active during baking. Depending on the model's accuracy, this may result in raycasts not being positioned correctly, potentially leading to a loss of accuracy. *
4. On that aligned frame, click `Set Reference Frame` (Clock icon) to establish it as the ground truth.
5. Click the `Sequence Solver` button to calculate and bake the animation across the specified range. (If set to "Timeline Markers," it will only bake on frames where a timeline marker exists).

> **💡 Calibration Tips (Matchmove):**
> * In Matchmove mode, alongside average/median estimation of focal length, optical center, and distortion for the entire range, you can also perform dynamic zoom lens analysis. Specifying the zoom range with keyframes helps stabilize the analysis.
> * Due to Blender's internal constraints, dynamic analysis of lens distortion cannot be keyframed and is therefore not supported.

### Matchmove Options

**Location Source:** Controls how camera position data is handled during sequence solving.
* **Solve Location** — Standard PnP solve.
* **Existing Location** — Keeps the existing camera position and solves rotation only.
* **Guided Location** — Uses the existing camera animation as a guide.
* **Timeline Markers** — Uses timeline marker camera positions as references.


**Location Filter:** Stabilizes difficult Matchmove solves.

**Roll Smoothing:** Reduces sudden roll jitter during Matchmove solving.


---
### Pin Align (3D Model Alignment)
1. In the 3D Viewport sidebar, open the **PinSolver > Pin Align** panel.
2. Set the mesh or point cloud to move as **Source** and the reference geometry as **Target**. With two supported objects selected, the selection button assigns the active object as Target.
3. Add at least three **Alignment Pins**. Pick matching points on both objects; Source and Target points can be picked in either order. Use Continuous Pin Pair mode (●) to create multiple pairs consecutively. Hold Alt while picking to snap to mesh vertices or Point Cloud points.
4. Use **Preview Pins** to validate the rough alignment. Enable **Pin Scale** only when the meshes need a uniform scale adjustment. Pin Scale estimates uniform scale from the Alignment Pins, while ICP Scale allows ICP refinement itself to adjust uniform scale.  
5. Use **Preview ICP** to refine the Source against the Target. Set a **Source Mask** when only part of the Source should contribute to the match. For meshes, the Source Mask can restrict ICP to selected faces, selected vertices, or a vertex group. Point-cloud sources do not support face-based masks.
6. Click **Apply** to keep the previewed transform, or **Revert** to restore the Source transform.

---
## Q&A / Troubleshooting

**Q: Pressing 'A' in the Interactive Pin Editor or Tweak mode doesn't add a pin.**  
**A:** To add a pin, there must be 3D geometry (mesh or point cloud) directly under your cursor along the camera's line of sight. Pins cannot be raycasted into empty space (e.g., just the background image). EEnsure your cursor is hovering over visible 3D geometry before pressing 'A'.

**Q: The alignment doesn't match up correctly when I click `Solve Alignment`.**  
**A:** There are several potential causes for this issue:
* If your camera has a `Camera Solver` constraint applied or contains complex keyframe animations, parent the camera to an Empty and change the **Solve Target** to **`Parent`**.
* Your camera settings (focal length, sensor size) might be incorrect. Consider using the Lens Calibration feature to estimate the correct values.
* The geometry or scale of the target 3D object might not accurately match the real-world object.
* The 3D Pin positions might be slightly misaligned with the target geometry.

**Q: The Lens Calibration checkboxes (Focal Length, etc.) are locked and cannot be clicked.**  
**A:** Due to mathematical constraints, automatic lens calibration requires a minimum of **6 active pins**. Add more pins to your scene to enable these options.

**Q: In Tweak mode, dragging a pin feels heavy or the camera doesn't follow the mouse properly.**  
**A:** Do you have any `Lens Calibration` options (like Focal Length) checked? If enabled, the solver tries to absorb the discrepancy by changing the zoom or lens distortion rather than just translating the camera. For smooth positional tweaking, it is highly recommended to perform an initial solve to determine your lens values, and then **uncheck all calibration options** before entering Tweak mode.

## License
PinSolver is GPL-3.0-or-later. OpenCV is Apache-2.0 licensed.
