# Example usage:
#   python src/main.py assets/mmu-iris-dataset/1/left/aeval1.bmp

# Note to self:
#   swaymsg 'for_window [app_id = "org.matplotlib.*"] floating enable'
#   python src/main.py assets/mmu-iris-dataset/14/left/liujwl2.bmp # A tricky one!

from   enum import IntEnum
import logging
import matplotlib.axes as axes
import matplotlib.image as image
import matplotlib.patches as patches
import matplotlib.pyplot as plot
import numpy as np
import skimage as ski
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


def img_binarize(img_gray: np.ndarray, thresh_pupil_rel: float) -> np.ndarray:
    mean = np.mean(img_gray)
    thresh_pupil = mean / thresh_pupil_rel
    return img_gray < thresh_pupil


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


def img_process(img: np.ndarray, img_mode: ImgMode) \
    -> tuple[np.ndarray, tuple[float, float]]:
    pupil_center = (float('nan'), float('nan'))
    if img_mode == ImgMode.RAW: return (img, pupil_center)
    img = img_grayscale(img)
    if img_mode == ImgMode.GRAYSCALED: return (img, pupil_center)
    img = img_binarize(img, 3.5) # TODO unhardcode
    if img_mode == ImgMode.BINARIZED: return (img, pupil_center)
    img = img_morpho(img, hole_size_max=500, disk_radius=20) # TODO
    pupil_center = img_center_by_centroid(img)
    # pupil_center = img_center_by_projection(img)
    return (img, pupil_center)


def img_plot(img_path: str, img_title: str, img_mode: ImgMode, plot_axes: axes.Axes):
    img_raw = image.imread(img_path)
    img, pupil_center = img_process(img_raw, img_mode)

    plot_axes.set_title(img_title)
    plot_axes.imshow(img_raw if img_mode == ImgMode.RESULT else img)
    plot_axes.figure.canvas.draw_idle()

    for patch in plot_axes.patches:
        patch.remove()
    plot_axes.add_patch(
        patches.Circle(pupil_center, radius=2, color='red', fill=True))


class ImgMode(IntEnum):
    RAW = 0
    GRAYSCALED = 1
    BINARIZED = 2
    MORPHOED = 3
    RESULT = 4
    _COUNT = 5


def main():
    img_paths = sys.argv[1:]
    if len(img_paths) < 1:
        logger.fatal("No image specified!")
        sys.exit(1)

    plot_fig, plot_axes = plot.subplots()
    plot_axes.axis('off')

    img_index = 0
    img_mode: ImgMode = ImgMode.RAW
    img_titler = lambda i: f"[{i + 1}/{len(img_paths)}] {img_paths[i]}"
    img_plot(img_paths[0], img_titler(0), img_mode, plot_axes)

    def key_press_handle(event):
        nonlocal img_index
        nonlocal img_mode
        key = event.key
        if   key == 'left':  img_index = (img_index - 1) % len(img_paths)
        elif key == 'right': img_index = (img_index + 1) % len(img_paths)
        elif key == 'shift+left':  img_index = (img_index - 10) % len(img_paths)
        elif key == 'shift+right': img_index = (img_index + 10) % len(img_paths)
        elif key == ' ':      img_mode = ImgMode((img_mode + 1) % int(ImgMode._COUNT))
        elif key == 'ctrl+ ': img_mode = ImgMode((img_mode - 1) % int(ImgMode._COUNT))
        else: return

        img_path = img_paths[img_index]
        img_title = img_titler(img_index)
        img_plot(img_path, img_title, img_mode, plot_axes)

    plot_fig.canvas.mpl_connect('key_press_event', key_press_handle)
    plot.show()

if __name__ == "__main__":
    main()
