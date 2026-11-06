"""Local zoom calibration from the preceding solved pose, with fixed distortion."""
import numpy as np
from .reprojection import robust_cost, robust_weights


def refine_zoom_intrinsics(points, pixels, intrinsics, distortion, rvec, tvec, estimate_focal, estimate_center, point_weights=None, robust_scale=None):
    import cv2

    points = np.ascontiguousarray(points, dtype=np.float64)
    observed = np.asarray(pixels, dtype=np.float64).reshape(-1)
    weights = np.ones(len(points)) if point_weights is None else np.asarray(point_weights,dtype=np.float64)
    pixel_weights = np.repeat(np.sqrt(weights),2)
    rotation = cv2.Rodrigues(rvec)[0]
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    depths = (points @ rotation.T + tvec)[:, 2]
    if not np.isfinite(depths).all() or np.any(depths <= 1e-8):
        raise ValueError('Zoom initialization has points behind the camera')
    center = -rotation.T @ tvec
    view_axis = rotation[2]
    # Constrain only focal/depth ambiguity; leave lateral camera motion unconstrained.
    prior_scale = intrinsics[0, 0] / np.median(depths) * np.sqrt(weights.sum()) * .05
    values = list(np.asarray(rvec).reshape(3)) + list(tvec)
    if estimate_focal:
        values.append(np.log(intrinsics[0, 0]))
    if estimate_center:
        values.extend(intrinsics[:2, 2])
    values = np.array(values, dtype=np.float64)
    focal_ratio = intrinsics[1, 1] / intrinsics[0, 0]

    def evaluate(parameters, with_jacobian=False):
        matrix = intrinsics.copy()
        index = 6
        if estimate_focal:
            if not -10 < parameters[index] < 20:
                return None
            matrix[0, 0] = np.exp(parameters[index])
            matrix[1, 1] = matrix[0, 0] * focal_ratio
            index += 1
        if estimate_center:
            matrix[:2, 2] = parameters[index:index+2]
        rot, rot_jac = cv2.Rodrigues(parameters[:3])
        if np.any((points @ rot.T + parameters[3:6])[:, 2] <= 1e-8):
            return None
        projected, derivative = cv2.projectPoints(points, parameters[:3], parameters[3:6], matrix, distortion)
        current_center = -rot.T @ parameters[3:6]
        errors = projected.reshape(-1)-observed
        prior = prior_scale*np.dot(current_center-center,view_axis)
        row_weights = pixel_weights
        if robust_scale is not None:
            row_weights = pixel_weights*np.repeat(np.sqrt(robust_weights(errors,robust_scale)),2)
        residual = np.concatenate((errors*row_weights, [prior]))
        if not np.isfinite(residual).all():
            return None
        cost = .5*float(np.dot(errors*pixel_weights,errors*pixel_weights))
        if robust_scale is not None:
            cost = robust_cost(errors,weights,robust_scale)
        cost += .5*prior**2
        if not with_jacobian:
            return cost
        columns = [derivative[:, :6]]
        if estimate_focal:
            columns.append((derivative[:, 6]*matrix[0, 0]+derivative[:, 7]*matrix[1, 1])[:, None])
        if estimate_center:
            columns.append(derivative[:, 8:10])
        center_jac = np.zeros((3, len(parameters)))
        for axis in range(3):
            center_jac[:, axis] = -rot_jac[axis].reshape(3, 3).T @ parameters[3:6]
        center_jac[:, 3:6] = -rot.T
        return residual, np.vstack((np.column_stack(columns)*row_weights[:,None], prior_scale*(view_axis @ center_jac)[None,:])), matrix, cost

    damping = .01
    for _ in range(40):
        residual, jacobian, _, cost = evaluate(values, True)
        hessian = jacobian.T @ jacobian
        step = np.linalg.solve(hessian+damping*np.diag(np.maximum(np.diag(hessian), 1e-10)), -jacobian.T @ residual)
        candidate = values + step
        trial = evaluate(candidate) if np.isfinite(candidate).all() else None
        if trial is not None and trial < cost:
            values = candidate
            damping = max(1e-9, damping*.3)
        else:
            damping *= 10
        if np.linalg.norm(step) < 1e-8 or damping > 1e12:
            break
    _, _, matrix, _ = evaluate(values, True)
    return matrix, values[:3].reshape(3, 1), values[3:6].reshape(3, 1)
