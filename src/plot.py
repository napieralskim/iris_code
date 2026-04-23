import matplotlib.axes    as mpaxes
import matplotlib.image   as mpimg
import matplotlib.patches as mppatches
from   eye.init import eye_main
from   eye.dto import *


def plot_center(axes: mpaxes.Axes, center: tuple[float, float], color: str) -> None:
    axes.add_patch(
        mppatches.Circle(center, radius=2, color=color, fill=True)) # TODO


def plot_masks(axes: mpaxes.Axes, mask_pupil: np.ndarray, mask_iris: np.ndarray):
    color_pupil = [1.0, 0.0, 0.0] # TODO
    color_iris  = [0.0, 0.0, 1.0]

    h, w = mask_pupil.shape
    vis = np.zeros((h, w, 3), dtype=np.float32)

    vis[mask_pupil] += color_pupil
    vis[mask_iris]  += color_iris

    axes.imshow(vis)


def plot_main(img_path: str, img_title: str, img_mode: ImgMode, axes: mpaxes.Axes):
    img_raw = mpimg.imread(img_path)
    res = eye_main(img_raw, img_mode)

    for patch in axes.patches:
        patch.remove()

    match res:
        case EyeRaw(img):
            axes.imshow(img)
        case EyeGrayscaled(img):
            axes.imshow(img)
        case EyeBinarized(pupil_mask, iris_mask):
            plot_masks(axes, pupil_mask, iris_mask)
        case EyeMorphoed(pupil_mask, iris_mask):
            plot_masks(axes, pupil_mask, iris_mask)
        case EyeCentered(pupil_mask, iris_mask, pupil_center):
            plot_masks(axes, pupil_mask, iris_mask)
            plot_center(axes, pupil_center, color='white') # TODO
        case EyeResult(img_orig, _, _, pupil_center):
            axes.imshow(img_orig)
            plot_center(axes, pupil_center, color='red') # TODO

    axes.set_title(img_title)
    axes.figure.canvas.draw_idle()
