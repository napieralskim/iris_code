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


def eye_morphoed(img: np.ndarray, cfg: ConfigEye) -> EyeMorphoed:
    prev = eye_binarized(img, cfg)
    pupil_mask_new = img_morpho(prev.pupil_mask,
        cfg.morpho_pupil.hole_size_max,
        cfg.morpho_pupil.disk_radius)
    iris_mask_rew = img_morpho(prev.iris_mask,
        cfg.morpho_iris.hole_size_max,
        cfg.morpho_iris.disk_radius)
    return EyeMorphoed(pupil_mask_new, iris_mask_rew)


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
        logger.fatal(f"eye_radius_pupil: Method `{cfg.center.method}` not supported.")
        sys.exit(1)
    pupil_radius = methods[cfg.radius_pupil.method](prev.pupil_mask)
    return EyeRadiusPupil(prev.pupil_mask, prev.iris_mask, prev.pupil_center, pupil_radius)


def eye_radius_both(img: np.ndarray, cfg: ConfigEye) -> EyeRadiusBoth:
    gray = img_grayscale(img, cfg.grayscale.weight_r, cfg.grayscale.weight_g, cfg.grayscale.weight_b)
    prev = eye_radius_pupil(img, cfg)
    methods = {
        "profile_horizontal": lambda mask: img_radius_by_profile_horizontal(
            gray, prev.pupil_center, prev.pupil_radius,
            cfg.radius_iris.radius_rel_min, cfg.radius_iris.radius_rel_max, cfg.radius_iris.sigma
        ) 
    }
    if cfg.radius_iris.method not in methods:
        logger.fatal(f"eye_radius_both: Method `{cfg.center.method}` not supported.")
        sys.exit(1)
    iris_radius = methods[cfg.radius_iris.method](prev.iris_mask)
    return EyeRadiusBoth(prev.pupil_mask, prev.iris_mask, prev.pupil_center, prev.pupil_radius, iris_radius) # TODO these get very long, refactor?


def eye_unwrapped(img: np.ndarray, cfg: ConfigEye) -> EyeUnwrapped:
    prev = eye_radius_both(img, cfg)
    iris_unwrapped = img_unwrap(img, prev.pupil_center, prev.pupil_radius, prev.iris_radius, cfg.unwrap.res_theta, cfg.unwrap.res_r)
    return EyeUnwrapped(img, prev.pupil_center, prev.pupil_radius, prev.iris_radius, iris_unwrapped)


def eye_main(img: np.ndarray, cfg: ConfigEye, img_mode: ImgMode):
    stages = {
        ImgMode.RAW:          eye_raw,
        ImgMode.GRAYSCALED:   eye_grayscaled,
        ImgMode.BINARIZED:    eye_binarized,
        ImgMode.MORPHOED:     eye_morphoed,
        ImgMode.CENTERED:     eye_center,
        ImgMode.RADIUS_PUPIL: eye_radius_pupil,
        ImgMode.RADIUS_BOTH:  eye_radius_both,
        ImgMode.UNWRAPPED:    eye_unwrapped,
    }
    if img_mode in stages:
        return stages[img_mode](img, cfg)
    else:
        logger.warning("eye_main: Unsupported `ImgMode`.")
        return None
