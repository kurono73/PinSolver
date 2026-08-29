from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class IcpSettings:
    iterations: int
    tolerance: float
    max_correspondence_distance: Optional[float]
    rejection_scale: float
    trim_fraction: float = 0.8
    pyramid_levels: int = 3
    refinement_method: str = "AUTO"
    tangent_weight: float = 0.35
    coverage_balance: float = 0.50
    allow_scale: bool = False
    fine_polish_passes: int = 0
    fine_polish_strength: float = 0.0
    nearest_chunk_size: int = 128
    max_scale_deviation: float = 0.05
    use_fast_nearest: bool = True
    reciprocal_correspondence: bool = True
    exact_final_nearest: bool = False


@dataclass
class IcpResult:
    success: bool
    transform: np.ndarray
    residual: float
    iterations: int
    source_inliers: int
    target_points: int
    message: str
    confidence: float = 0.0
    overlap_ratio: float = 0.0
    p95_residual: float = -1.0
    coverage_ratio: float = 0.0
