from   config import ConfigEyeDaugman
import numpy as np
from   numpy.typing import NDArray
import scipy as sp


def daug_code(iris_signals: np.ndarray, cfg_daug: ConfigEyeDaugman) -> np.ndarray:
    _, iris_w = iris_signals.shape
    cfg_gabor = cfg_daug.gabor

    kernel = daug_gabor_kernel(
        length=cfg_gabor.length,
        sigma=cfg_gabor.sigma,
        freq=cfg_gabor.frequency)
    
    iris_code = np.zeros((8, 2 * iris_w), dtype=np.bool)
    for i in range(8):
        convolved = np.convolve(iris_signals[i], kernel, mode='same')
        
        iris_code[i, : iris_w] = np.real(convolved) > 0
        iris_code[i, iris_w :] = np.imag(convolved) > 0
        
    return iris_code


# iris gray
def daug_signals(iris: np.ndarray) -> np.ndarray:
    iris_h, iris_w = iris.shape
    y_ranges = np.linspace(0, iris_h, 9).round().astype(int)
    
    signals = []

    for i in range(8):
        y0, y1 = y_ranges[i], y_ranges[i+1]
        strip = iris[y0 : y1, :]
        signal = daug_strip_flatten(strip)

        mask = daug_strip_mask(i, iris_w)
        signal *= mask
        signal_valid = signal[mask]
        signal = (signal - np.mean(signal_valid)) # / (np.std(signal_valid) + 1e-6)
        signals.append(signal)
        
    return np.stack(signals)


def daug_strip_flatten(strip: np.ndarray) -> np.ndarray:
    strip_height = strip.shape[0]

    gauss_args = np.linspace(-2, 2, strip_height) # TODO configurable
    weights = sp.stats.norm.pdf(gauss_args)
    weights /= weights.sum()

    signal = np.sum(strip * weights[:, np.newaxis], axis=0)
    return signal


def _daug_strip_range_set(strip: NDArray[np.bool], deg0: float, deg1: float, value: np.bool):
    strip_len = len(strip)
    x0 = round(deg0 * strip_len / 360)
    x1 = round(deg1 * strip_len / 360)
    strip[x0:x1] = value

# Angle ranges per stripe:
# 0-3 => 360° - 30° at the bottom
# 4-5 => 113° at each side
# 6-7 => 90° at each side
def daug_strip_mask(strip_index: int, iris_width: int) -> NDArray[np.bool]:
    strip = np.zeros(iris_width, dtype=np.bool)
    # ? == round(degree * iris_width / 360)
    if strip_index < 4:
        strip[:] = True
        _daug_strip_range_set(strip, 75, 105, np.False_)
    elif strip_index < 6:
        _daug_strip_range_set(strip, 0, 57, np.True_)
        _daug_strip_range_set(strip, 303, 360, np.True_)
        _daug_strip_range_set(strip, 123, 237, np.True_)
    else:
        _daug_strip_range_set(strip, 0, 45, np.True_)
        _daug_strip_range_set(strip, 315, 360, np.True_)
        _daug_strip_range_set(strip, 135, 225, np.True_)
    return strip

def daug_gabor_kernel(length, sigma, freq):
    x = np.arange(length) - length // 2
    gaussian = np.exp(-x**2 / (2 * sigma**2))
    wave = np.exp(1j * 2 * np.pi * freq * x)
    kernel = gaussian * wave
    kernel -= kernel.mean()
    return kernel
