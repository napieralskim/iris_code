import numpy as np
from   numpy.typing import NDArray
import scipy as sp


def daug_split_8(iris_gray: np.ndarray) -> np.ndarray:
    iris_height, iris_width = iris_gray.shape
    y_ranges = np.linspace(0, iris_height, 9).round().astype(int)
    
    signals = []

    for i in range(8):
        y0, y1 = y_ranges[i], y_ranges[i+1]
        strip = iris_gray[y0 : y1, :]
        signal = daug_strip_flatten(strip)

        mask = daug_strip_mask(i, iris_width)
        signal *= mask
        signals.append(signal)
        
    return np.stack(signals)


def daug_strip_flatten(strip: np.ndarray) -> np.ndarray:
    strip_height = strip.shape[0]

    gauss_args = np.linspace(-2, 2, strip_height) # TODO configurable
    weights = sp.stats.norm.pdf(gauss_args)
    weights /= weights.sum()

    signal = np.sum(strip * weights[:, np.newaxis], axis=0)
    return signal


# Angle ranges per stripe:
# 0-3 => 360° - 30° at the bottom
# 4-5 => 113° at each side
# 6-7 => 90° at each side
# TODO assert iris_width == 360
def daug_strip_mask(strip_index: int, iris_width: int) -> NDArray[np.bool]:
    mask = np.full(iris_width, False, dtype=np.bool)
    if strip_index < 4:
        mask[:] = True
        mask[75:105] = 0 
    elif strip_index < 6:
        mask[0:57] = True; mask[303:360] = True
        mask[123:237] = True
    else:
        mask[0:45] = True; mask[315:360] = True
        mask[135:225] = True
    return mask
