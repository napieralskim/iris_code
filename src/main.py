# Example usage:
#   python src/main.py assets/mmu-iris-dataset/1/left/aeval1.bmp

# Note to self:
#   swaymsg 'for_window [app_id = "org.matplotlib.*"] floating enable'

import logging
from   logging import Logger
import matplotlib.image as mpimg
import matplotlib.pyplot as plot
import numpy as np
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO, # TODO
    format = "%(levelname)s: %(message)s")


def grayscale(img: np.ndarray):

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


def main():

    if len(sys.argv) < 2:
        logger.fatal("No image specified")
        sys.exit(1)

    img_path = sys.argv[1]
    logger.info(f"Using image: {img_path}")
    img = mpimg.imread(img_path)

    img = grayscale(img)

    plot.imshow(img)
    plot.axis('off')
    plot.show()

if __name__ == "__main__":
    main()
