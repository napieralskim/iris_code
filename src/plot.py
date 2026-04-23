from matplotlib.axes import Axes
import matplotlib.image as mpimg
import matplotlib.patches as mppatches
from   config import Config, ConfigPlot
from   eye.init import eye_main
from   eye.dto import *


def plot_center(axes: Axes, cfg: ConfigPlot, center: tuple[float, float]) -> None:
    axes.add_patch(mppatches.Circle(
        center,
        radius=cfg.center.radius,
        color=cfg.center.color, 
        fill=True))


def plot_masks(axes: Axes, cfg: ConfigPlot, mask_pupil: np.ndarray, mask_iris: np.ndarray):
    color_pupil = cfg.masks.color_pupil
    color_iris  = cfg.masks.color_iris

    h, w = mask_pupil.shape
    vis = np.zeros((h, w, 3), dtype=np.float32)

    vis[mask_pupil] += color_pupil
    vis[mask_iris]  += color_iris

    axes.imshow(vis)


def plot_main(axes: Axes, cfg: Config, img_path: str, img_title: str, img_mode: ImgMode):
    img_raw = mpimg.imread(img_path)
    res = eye_main(img_raw, cfg.eye, img_mode)

    for patch in axes.patches:
        patch.remove()

    match res:
        case EyeRaw(img):
            axes.imshow(img)
        case EyeGrayscaled(img):
            axes.imshow(img)
        case EyeBinarized(pupil_mask, iris_mask):
            plot_masks(axes, cfg.plot, pupil_mask, iris_mask)
        case EyeMorphoed(pupil_mask, iris_mask):
            plot_masks(axes, cfg.plot, pupil_mask, iris_mask)
        case EyeCentered(pupil_mask, iris_mask, pupil_center):
            plot_masks(axes, cfg.plot, pupil_mask, iris_mask)
            plot_center(axes, cfg.plot, pupil_center)
        case EyeResult(img_orig, _, _, pupil_center):
            axes.imshow(img_orig)
            plot_center(axes, cfg.plot, pupil_center)

    axes.set_title(img_title)
    axes.figure.canvas.draw_idle()
