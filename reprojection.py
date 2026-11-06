"""Shared pixel-domain importance, robust loss, and local camera refinement."""
import numpy as np


def pin_importance(value):
    return 1 + int(np.clip(value, 0.0, 1.0)*99)


def pixel_scale(width, height):
    return max(1.5, min(width, height)*.002)


def robust_weights(errors, scale):
    lengths = np.linalg.norm(np.asarray(errors).reshape(-1, 2), axis=1)
    return np.minimum(1.0, scale/np.maximum(lengths, 1e-12))


def robust_cost(errors, weights, scale):
    lengths = np.linalg.norm(np.asarray(errors).reshape(-1, 2), axis=1)
    losses = np.where(lengths <= scale, .5*lengths**2, scale*(lengths-.5*scale))
    return float(np.dot(weights, losses))


def robust_score(errors, weights, scale):
    return np.sqrt(2*robust_cost(errors, weights, scale)/max(float(np.sum(weights)), 1e-12))


def project_center_pose(points, rvec, center, intrinsics, distortion):
    import cv2

    rotation, rotation_jac = cv2.Rodrigues(rvec)
    tvec = -rotation @ center
    depths = (points-center) @ rotation[2]
    projected, derivative = cv2.projectPoints(points, rvec, tvec, intrinsics, distortion)
    jacobian = derivative[:, :6].copy()
    for axis in range(3):
        jacobian[:, axis] -= derivative[:, 3:6] @ (rotation_jac[axis].reshape(3, 3) @ center)
    jacobian[:, 3:6] = -derivative[:, 3:6] @ rotation
    return projected.reshape(-1, 2), jacobian, derivative, depths


def refine_pose(points, pixels, intrinsics, distortion, rvec, tvec, weights, scale, fixed_center=False):
    import cv2

    points = np.ascontiguousarray(points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64)
    center = -cv2.Rodrigues(rvec)[0].T @ np.asarray(tvec).reshape(3)
    values = np.r_[np.asarray(rvec).reshape(3), center]
    columns = 3 if fixed_center else 6

    def evaluate(parameters):
        projected, jacobian, _, depths = project_center_pose(points, parameters[:3], parameters[3:6], intrinsics, distortion)
        if np.any(depths <= 1e-8) or not np.isfinite(projected).all():
            return None
        errors = projected-pixels
        return errors, jacobian[:, :columns], robust_cost(errors, weights, scale)

    current = evaluate(values)
    if current is None:
        return rvec, tvec
    damping = .001
    for _ in range(12):
        errors, jacobian, cost = current
        row_weights = np.repeat(np.sqrt(weights*robust_weights(errors, scale)), 2)
        jacobian = jacobian*row_weights[:, None]
        residual = errors.reshape(-1)*row_weights
        hessian = jacobian.T @ jacobian
        step = np.linalg.solve(hessian+damping*np.diag(np.maximum(np.diag(hessian), 1e-10)), -jacobian.T @ residual)
        candidate = values.copy()
        candidate[:columns] += step
        trial = evaluate(candidate) if np.isfinite(candidate).all() else None
        if trial is not None and trial[2] < cost:
            values, current = candidate, trial
            damping = max(1e-9, damping*.3)
        else:
            damping *= 10
        if np.linalg.norm(step) < 1e-8 or damping > 1e10:
            break
    rotation = cv2.Rodrigues(values[:3])[0]
    return values[:3].reshape(3, 1), (-rotation @ values[3:6]).reshape(3, 1)
