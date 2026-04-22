# Example usage:
#   python src/main.py assets/mmu-iris-dataset/1/left/aeval1.bmp

# Note to self:
#   swaymsg 'for_window [app_id = "org.matplotlib.*"] floating enable'

import logging
import matplotlib.figure as mpfig
import matplotlib.axes as mpaxes
import matplotlib.image as mpimg
import matplotlib.pyplot as plot
import numpy as np
import skimage
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO, # TODO
    format = "%(levelname)s: %(message)s")


def img_grayscale(img: np.ndarray) -> np.ndarray:
    wr = 0.299
    wg = 0.587
    wb = 0.114

    if len(img.shape) != 3:
        logger.info(f"grayscale: Image is not three-dimensional. Skipping.")
        return img
    
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = wr * r + wg * g + wb * b
    gray = gray.astype(np.uint8)
    return gray


def img_binarize(img_gray: np.ndarray, thresh_scale: float = 1.0) -> np.ndarray:
    thresh = skimage.filters.threshold_otsu(img_gray)
    thresh *= thresh_scale
    img_binary = img_gray > thresh
    return img_binary


def img_process(img: np.ndarray) -> np.ndarray:
    img = img_grayscale(img)
    img = img_binarize(img, 0.4) # TODO unhardcode
    return img


def img_plot(
    img_path: str, img_title: str,
    plot_img: mpimg.AxesImage | None, plot_fig: mpfig.Figure, plot_axes: mpaxes.Axes
) -> mpimg.AxesImage:
    img = mpimg.imread(img_path)
    img = img_process(img)
    plot_axes.set_title(img_title)
    if plot_img is None:
        plot_img = plot_axes.imshow(img)
    else:
        plot_img.set_data(img)
        plot_axes.figure.canvas.draw_idle()
    return plot_img


def main():
    img_paths = sys.argv[1:]
    if len(img_paths) < 1:
        logger.fatal("No image specified!")
        sys.exit(1)

    plot_fig, plot_axes = plot.subplots()
    plot_img: mpimg.AxesImage | None = None
    plot_axes.axis('off')

    img_index = 0
    img_titler = lambda i: f"[{i + 1}/{len(img_paths)}] {img_paths[i]}"
    plot_img = img_plot(img_paths[0], img_titler(0), plot_img, plot_fig, plot_axes)

    def key_press_handle(event):
        nonlocal img_index
        key_bindings = { 'left': -1, 'right': +1, 'h': -1, 's': +1 }
        if event.key not in key_bindings:
            return
        img_index += key_bindings[event.key]
        img_index %= len(img_paths)

        img_path = img_paths[img_index]
        img_title = img_titler(img_index)
        img_plot(img_path, img_title, plot_img, plot_fig, plot_axes)

    plot_fig.canvas.mpl_connect('key_press_event', key_press_handle)
    plot.show()

if __name__ == "__main__":
    main()
