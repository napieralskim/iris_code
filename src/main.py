# Example usage:
#   python src/main.py assets/mmu-iris-dataset/1/left/aeval1.bmp

# NOTES TO SELF:
# Floating window on swaywm:
#   swaymsg 'for_window [app_id = "org.matplotlib.*"] floating enable'
# Hard ones:
#   57, 58, 71, 73, 136, 272, 334
#   91 eyebrow


import logging
import matplotlib.pyplot as plot
from   eye.dto import ImgMode
from   plot import plot_main
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO, # TODO
    format = "%(levelname)s: %(message)s")


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
    plot_main(img_paths[0], img_titler(0), img_mode, plot_axes)

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
        plot_main(img_path, img_title, img_mode, plot_axes)

    plot_fig.canvas.mpl_connect('key_press_event', key_press_handle)
    plot.show()

if __name__ == "__main__":
    main()
