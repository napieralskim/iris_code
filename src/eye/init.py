from eye.dto import *


def eye_raw(img: np.ndarray) -> EyeRaw:
    return EyeRaw(img)

def eye_grayscaled(img: np.ndarray) -> EyeGrayscaled:
    return EyeGrayscaled(img_grayscale(img))

def eye_binarized(img: np.ndarray) -> EyeBinarized:
    prev = eye_grayscaled(img)
    mask_pupil, mask_iris = img_binarize(prev.img, thresh_pupil_rel=3.5, thresh_iris_rel=1.5) # TODO
    return EyeBinarized(mask_pupil, mask_iris)

# TODO only pupil for now
def eye_morphoed(img: np.ndarray) -> EyeMorphoed:
    prev = eye_binarized(img)
    pupil_mask_new = img_morpho(prev.pupil_mask, hole_size_max=500, disk_radius=20) # TODO
    return EyeMorphoed(pupil_mask_new, prev.iris_mask)

def eye_center(img: np.ndarray) -> EyeCentered:
    prev = eye_morphoed(img)
    pupil_center = img_center_by_centroid(prev.pupil_mask) # TODO
    return EyeCentered(prev.pupil_mask, prev.iris_mask, pupil_center)

def eye_result(img: np.ndarray) -> EyeResult:
    prev = eye_center(img) # TODO
    return EyeResult(img, prev.pupil_mask, prev.iris_mask, prev.pupil_center)

def eye_main(img: np.ndarray, img_mode: ImgMode):
    match img_mode:
        case ImgMode.RAW:        return eye_raw(img)
        case ImgMode.GRAYSCALED: return eye_grayscaled(img)
        case ImgMode.BINARIZED:  return eye_binarized(img)
        case ImgMode.MORPHOED:   return eye_morphoed(img)
        case ImgMode.CENTERED:   return eye_center(img)
        case ImgMode.RESULT:     return eye_result(img)
