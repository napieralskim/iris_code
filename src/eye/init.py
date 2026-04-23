from   config import ConfigEye
from   eye.dto import *
import logging
import sys
logger = logging.getLogger(__name__)


def eye_raw(img: np.ndarray, _: ConfigEye) -> EyeRaw:
    return EyeRaw(img)


def eye_grayscaled(img: np.ndarray, cfg: ConfigEye) -> EyeGrayscaled:
    gray = img_grayscale(img,
        cfg.grayscale.weight_r,
        cfg.grayscale.weight_g,
        cfg.grayscale.weight_b)
    return EyeGrayscaled(gray)


def eye_binarized(img: np.ndarray, cfg: ConfigEye) -> EyeBinarized:
    prev = eye_grayscaled(img, cfg)
    mask_pupil, mask_iris = img_binarize(prev.img,
        cfg.binarize.thresh_pupil_rel,
        cfg.binarize.thresh_iris_rel)
    return EyeBinarized(mask_pupil, mask_iris)


# TODO only pupil for now
def eye_morphoed(img: np.ndarray, cfg: ConfigEye) -> EyeMorphoed:
    prev = eye_binarized(img, cfg)
    pupil_mask_new = img_morpho(prev.pupil_mask,
        cfg.morpho.hole_size_max,
        cfg.morpho.disk_radius)
    return EyeMorphoed(pupil_mask_new, prev.iris_mask)


def eye_center(img: np.ndarray, cfg: ConfigEye) -> EyeCentered:
    prev = eye_morphoed(img, cfg)
    methods = {
        "centroid":   lambda mask: img_center_by_centroid(mask),
        "projection": lambda mask: img_center_by_projection(mask),
    }
    if cfg.center.method not in methods:
        logger.fatal(f"eye_center: Method `{cfg.center.method}` not supported.")
        sys.exit(1)
    pupil_center = methods[cfg.center.method](prev.pupil_mask)
    return EyeCentered(prev.pupil_mask, prev.iris_mask, pupil_center)


def eye_radius_pupil(img: np.ndarray, cfg: ConfigEye) -> EyeRadiusPupil:
    prev = eye_center(img, cfg)
    methods = {
        "area": lambda mask: img_radius_by_area(mask),
    }
    if cfg.radius_pupil.method not in methods:
        logger.fatal(f"eye_center: Method `{cfg.center.method}` not supported.")
        sys.exit(1)
    pupil_radius = methods[cfg.radius_pupil.method](prev.pupil_mask)
    return EyeRadiusPupil(prev.pupil_mask, prev.iris_mask, prev.pupil_center, pupil_radius)


def eye_result(img: np.ndarray, cfg: ConfigEye) -> EyeResult:
    prev = eye_radius_pupil(img, cfg) # TODO should invoke last step
    return EyeResult(img, prev.pupil_center, prev.pupil_radius)


def eye_main(img: np.ndarray, cfg: ConfigEye, img_mode: ImgMode):
    match img_mode:
        case ImgMode.RAW:          return eye_raw(img, cfg)
        case ImgMode.GRAYSCALED:   return eye_grayscaled(img, cfg)
        case ImgMode.BINARIZED:    return eye_binarized(img, cfg)
        case ImgMode.MORPHOED:     return eye_morphoed(img, cfg)
        case ImgMode.CENTERED:     return eye_center(img, cfg)
        case ImgMode.RADIUS_PUPIL: return eye_radius_pupil(img, cfg)
        case ImgMode.RESULT:       return eye_result(img, cfg)
