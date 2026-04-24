# Example usage:
#   python src/main.py assets/mmu-iris-dataset/*/*/*.bmp

# NOTES TO SELF:
# Floating window on swaywm:
#   swaymsg 'for_window [app_id = "org.matplotlib.*"] floating enable'
# Generate `Config` classes:
#   datamodel-codegen --input config.yaml --input-file-type yaml --output src/config.py --parent-scoped-naming --class-name Config
# Hard ones:
#   448, 98, 389, 339 => reflection hole not fully enclosed
#   435 => dark eyebrow in the corner
#   298, 299 => eyelashes cover the pupil


import yaml

from   config import Config
import logging
import matplotlib.pyplot as plot
from   eye.dto import ImgMode
from   plot import plot_main
import sys
logger = logging.getLogger(__name__)


def main(cfg: Config):
    img_paths = sys.argv[1:]
    if len(img_paths) < 1:
        logger.fatal("No image specified!")
        sys.exit(1)

    plot_fig, plot_axes = plot.subplots()
    plot_axes.axis('off')

    img_index = 0
    img_mode: ImgMode = ImgMode.RAW
    img_titler = lambda i: f"[{i + 1}/{len(img_paths)}] {img_paths[i]}"
    plot.figure(plot_fig) # TODO
    plot_main(plot_axes, cfg, img_paths[0], img_titler(0), img_mode)

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
        plot.figure(plot_fig) # TODO
        plot_main(plot_axes, cfg, img_path, img_title, img_mode)

    plot_fig.canvas.mpl_connect('key_press_event', key_press_handle)
    plot.show()


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config_raw = yaml.safe_load(f)
    config = Config.model_validate(config_raw)
    main(config)
