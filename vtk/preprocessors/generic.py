from .base import BasePreprocessor
import cv2
import numpy as np


class GenericPreprocessor(BasePreprocessor):
    """
    A generic preprocessor that uses the CPU for its calculations.
    """
    def __init__(self):
        super().__init__()

    def resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize the given image to a specific width and height.
        :param image: Image to run operation on.
        :param width: Width to resize image to.
        :param height: Height to resize image to.
        :returns: Resized image.
        """
        return cv2.resize(image, (width, height))

    def recolor(self, image: np.ndarray, color_scheme: int) -> np.ndarray:
        """
        Recolor an image to another color scheme.
        :param image: Image to run operation on.
        :param color_scheme: Color scheme integer to change image into. Inherited from cv2.COLOR_*.
        :returns: Recolored image.
        """
        return cv2.cvtColor(image, color_scheme)
