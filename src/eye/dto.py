from   dataclasses import dataclass
from   enum import IntEnum
from   img import *
import numpy as np


class ImgMode(IntEnum):
    RAW          = 0
    GRAYSCALED   = 1
    BINARIZED    = 2
    MORPHOED     = 3
    CENTERED     = 4
    RADIUS_PUPIL = 5
    RADIUS_BOTH  = 6
    UNWRAPPED    = 7
    _COUNT       = 8


@dataclass
class EyeRaw:
    img: np.ndarray

@dataclass
class EyeGrayscaled:
    img: np.ndarray

@dataclass
class EyeBinarized:
    pupil_mask: np.ndarray
    iris_mask:  np.ndarray

@dataclass
class EyeMorphoed:
    pupil_mask: np.ndarray
    iris_mask:  np.ndarray

@dataclass
class EyeCentered:
    pupil_mask:   np.ndarray
    iris_mask:    np.ndarray
    pupil_center: CoordsF

@dataclass
class EyeRadiusPupil:
    pupil_mask:   np.ndarray
    iris_mask:    np.ndarray
    pupil_center: CoordsF
    pupil_radius: float

@dataclass
class EyeRadiusBoth:
    pupil_mask:   np.ndarray
    iris_mask:    np.ndarray
    pupil_center: CoordsF
    pupil_radius: float
    iris_radius:  float

@dataclass
class EyeUnwrapped:
    img_orig:       np.ndarray
    pupil_center:   CoordsF
    pupil_radius:   float
    iris_radius:    float
    iris_unwrapped: np.ndarray
