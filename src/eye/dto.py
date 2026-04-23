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
    RESULT     = 5
    _COUNT     = 6


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
class EyeResult:
    img_orig:     np.ndarray
    pupil_mask:   np.ndarray
    iris_mask:    np.ndarray
    pupil_center: tuple[float, float]
