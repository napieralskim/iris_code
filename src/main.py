# Example usage:
#   python src/main.py assets/mmu-iris-dataset/1/left/aeval1.bmp

# Note to self:
#   swaymsg 'for_window [app_id = "org.matplotlib.*"] floating enable'

import logging
from   logging import Logger
import matplotlib.image as mpimg
import matplotlib.pyplot as plot
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO, # TODO
    format = "%(levelname)s: %(message)s")

def main():

    if len(sys.argv) < 2:
        logger.fatal("No image specified")
        sys.exit(1)

    img_path = sys.argv[1]
    logger.info(f"Using image: {img_path}")    

    img = mpimg.imread(img_path)
    plot.imshow(img)
    plot.axis('off')
    plot.show()

if __name__ == "__main__":
    main()
