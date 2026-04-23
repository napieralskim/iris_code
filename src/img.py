import logging
import numpy as np
import skimage as ski


logger = logging.getLogger(__name__)


def img_grayscale(
    img: np.ndarray,
    wr: float = 0.299,
    wg: float = 0.587,
    wb: float = 0.114
) -> np.ndarray:
    
    if len(img.shape) != 3:
        logger.info(f"grayscale: Image is not three-dimensional. Skipping.")
        return img
    
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = wr * r + wg * g + wb * b
    gray = gray.astype(np.uint8)
    return gray


def img_binarize(img_gray: np.ndarray, thresh_pupil_rel: float, thresh_iris_rel: float) \
    -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(img_gray)
    thresh_pupil = mean / thresh_pupil_rel
    thresh_iris  = mean / thresh_iris_rel
    return (
        img_gray < thresh_pupil,
        img_gray < thresh_iris,
    )


def img_morpho(mask: np.ndarray, hole_size_max: int,  disk_radius: float) -> np.ndarray:
    mask_new = ski.morphology.remove_small_holes(mask, max_size=hole_size_max)
    mask_new = ski.morphology.isotropic_opening(mask_new, disk_radius)
    return mask_new


def img_center_by_centroid(mask: np.ndarray) -> tuple[float, float]:
    centroid = np.argwhere(mask == 1).mean(axis=0)
    return centroid[1], centroid[0]


def img_center_by_projection(mask: np.ndarray) -> tuple[float, float]:
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    x_c = np.argmax(col_sums)
    y_c = np.argmax(row_sums)

    return (float(x_c), float(y_c))


def img_radius_by_area(mask: np.ndarray) -> float:
    area = np.sum(mask)
    radius = np.sqrt(area / np.pi)
    return radius
