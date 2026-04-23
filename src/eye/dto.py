from   dataclasses import dataclass
from   enum import IntEnum
from   img import *
import numpy as np


class ImgMode(IntEnum):
    RAW        = 0
    GRAYSCALED = 1
    BINARIZED  = 2
    MORPHOED   = 3
    CENTERED   = 4
    RADIUS_PUPIL   = 5
    RESULT     = 6
    _COUNT     = 7


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
    pupil_center: tuple[float, float]

@dataclass
class EyeRadiusPupil:
    pupil_mask:   np.ndarray
    iris_mask:    np.ndarray
    pupil_center: tuple[float, float]
    pupil_radius: float

@dataclass
class EyeResult:
    img_orig:     np.ndarray
    pupil_center: tuple[float, float]
    pupil_radius: float
