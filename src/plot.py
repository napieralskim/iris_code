from   config import Config, ConfigPlot
from   eye.init import eye_main
from   eye.dto import *
from   matplotlib.axes import Axes
from   matplotlib.figure import Figure
from   matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.patches as mppatches
import numpy as np
from   pathlib import Path


def plot_masks(ax: Axes, cfg: ConfigPlot, mask_pupil: np.ndarray, mask_iris: np.ndarray) -> None:
    color_pupil = cfg.masks.color_pupil
    color_iris  = cfg.masks.color_iris

    h, w = mask_pupil.shape
    vis = np.zeros((h, w, 3), dtype=np.float32)

    vis[mask_pupil] += color_pupil
    vis[mask_iris]  += color_iris

    ax.imshow(vis)


def plot_center(ax: Axes, cfg: ConfigPlot, center: tuple[float, float]) -> None:
    ax.add_patch(mppatches.Circle(
        center,
        radius=cfg.center.radius,
        color=cfg.center.color, 
        fill=True))


def plot_ring(ax: Axes, cfg: ConfigPlot, center: tuple[float, float], radius: float, color: str) -> None:
    ax.add_patch(mppatches.Circle(
        center,
        radius=radius,
        color=color,
        fill=False))


def plot_main(ax: Axes, ax_extra: Axes, grid: GridSpec, cfg_all: Config, img_path: str, img_title: str, img_mode: ImgMode):
    cfg = cfg_all.plot

    img_raw = mpimg.imread(img_path)
    res = eye_main(img_raw, cfg_all.eye, img_mode)

    for patch in ax.patches:
        patch.remove()
    ax.set_subplotspec(grid[:, :])
    ax_extra.set_visible(False)

    match res:
        case EyeRaw(img):
            ax.imshow(img)
        case EyeGrayscaled(img):
            ax.imshow(img)
        case EyeBinarized(pupil_mask, iris_mask):
            plot_masks(ax, cfg, pupil_mask, iris_mask)
        case EyeMorphoed(pupil_mask, iris_mask):
            plot_masks(ax, cfg, pupil_mask, iris_mask)
        case EyeCentered(pupil_mask, iris_mask, pupil_center):
            plot_masks(ax, cfg, pupil_mask, iris_mask)
            plot_center(ax, cfg, pupil_center)
        case EyeRadiusPupil(pupil_mask, iris_mask, pupil_center, pupil_radius):
            plot_masks(ax, cfg, pupil_mask, iris_mask)
            plot_center(ax, cfg, pupil_center)
            plot_ring(ax, cfg, pupil_center, pupil_radius, cfg.radius_pupil.color)
        case EyeRadiusBoth(pupil_mask, iris_mask, pupil_center, pupil_radius, iris_radius):
            plot_masks(ax, cfg, pupil_mask, iris_mask)
            plot_center(ax, cfg, pupil_center)
            plot_ring(ax, cfg, pupil_center, pupil_radius, cfg.radius_pupil.color)
            plot_ring(ax, cfg, pupil_center, iris_radius,  cfg.radius_iris.color)
        case \
            EyeUnwrapped(img_orig, pupil_center, pupil_radius, iris_radius, img_iris) | \
            EyeSplit    (img_orig, pupil_center, pupil_radius, iris_radius, img_iris) | \
            EyeEncoded  (img_orig, pupil_center, pupil_radius, iris_radius, img_iris):
            ax.set_subplotspec(grid[0, :])
            ax_extra.set_visible(True)

            ax.imshow(img_orig)
            plot_center(ax, cfg, pupil_center)
            plot_ring(ax, cfg, pupil_center, pupil_radius, cfg.radius_pupil.color)
            plot_ring(ax, cfg, pupil_center, iris_radius,  cfg.radius_iris.color)

            ax_extra.imshow(img_iris) # either the raw unwrapped iris or iris signals
        case _:
            logger.warning("plot_main: Unsupported `ImgMode`.")

    ax.set_title(img_title)
    ax.figure.canvas.draw_idle()
