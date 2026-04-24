# This file shouldn't import `Config`.
import logging
import numpy as np
import scipy as sp
import scipy.ndimage
import skimage as ski
from   utils import Coords, CoordsF
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


def img_center_by_centroid(mask: np.ndarray) -> CoordsF:
	centroid = np.argwhere(mask == 1).mean(axis=0)
	return CoordsF(x=centroid[1], y=centroid[0])


def img_center_by_projection(mask: np.ndarray) -> CoordsF:
	col_sums = np.sum(mask, axis=0)
	row_sums = np.sum(mask, axis=1)

	x_c = np.argmax(col_sums)
	y_c = np.argmax(row_sums)

	return CoordsF(x=float(x_c), y=float(y_c))


def img_radius_by_area(mask: np.ndarray) -> float:
	area = np.sum(mask)
	radius = np.sqrt(area / np.pi)
	return radius


def img_radius_by_profile_horizontal(
	img_gray: np.ndarray, center: CoordsF, pupil_radius: float,
	iris_radius_rel_min: float, iris_radius_rel_max: float, sigma: float
) -> float:
	center_int = Coords.from_floats(center)

	radius_min = int(pupil_radius * iris_radius_rel_min)
	radius_max = int(pupil_radius * iris_radius_rel_max)
	radius_fallback = int(pupil_radius * 2.5)

	profile = img_gray[center_int.y, (center_int.x + radius_min) : (center_int.x + radius_max)].astype(float)
	if len(profile) < 1:
		logger.error("img_radius_by_profile_horizontal: The profile is empty!")
		return radius_fallback
	profile = ski.filters.gaussian(profile, sigma=sigma)
	gradient = np.diff(profile)
	radius_right = radius_min + np.argmax(gradient)

	profile = img_gray[center_int.y, (center_int.x - radius_max) : (center_int.x - radius_min)].astype(float)
	if len(profile) < 1:
		logger.error("img_radius_by_profile_horizontal: The profile is empty!")
		return radius_fallback
	profile = ski.filters.gaussian(profile, sigma=sigma)
	gradient = np.diff(profile)
	radius_left = radius_max - np.argmin(gradient)

	return 0.5 * float(radius_left + radius_right)


def img_unwrap(
	img: np.ndarray, center: CoordsF,
	radius_inner: float, radius_outer: float,
	res_theta: int, res_r: int
) -> np.ndarray:
	theta_space = np.linspace(0, 2 * np.pi, res_theta)
	r_space = np.linspace(radius_inner, radius_outer, res_r)
	
	theta_grid, r_grid = np.meshgrid(theta_space, r_space)
	
	src_x = center.x + r_grid * np.cos(theta_grid)
	src_y = center.y + r_grid * np.sin(theta_grid)

	unwrapped_channels = []
	for channel_i in range(3):
		channel = img[:, :, channel_i]
		unwrapped_channel = sp.ndimage.map_coordinates(channel, [src_y, src_x], order=1, mode='constant')
		unwrapped_channels.append(unwrapped_channel)
	
	unwrapped = np.stack(unwrapped_channels, axis=-1)
	return unwrapped
