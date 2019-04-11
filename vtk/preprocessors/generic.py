from .base import BasePreprocessor
import cv2
import numpy as np


class GenericPreprocessor(BasePreprocessor):
    """
    A generic preprocessor that uses the CPU for its calculations.
    """
    def __init__(self, mat: np.ndarray):
        self.mat = mat
        super().__init__()

    def resize(self, width: int, height: int) -> np.ndarray:
        """
        Resize the given image to a specific width and height.
        :param width: Width to resize image to.
        :param height: Height to resize image to.
        :returns: Resized image.
        """
        return cv2.resize(self.mat, (width, height))

    def recolor(self, color_scheme: int) -> np.ndarray:
        """
        Recolor an image to another color scheme.
        :param color_scheme: Color scheme integer to change image into. Inherited from cv2.COLOR_*.
        :returns: Recolored image.
        """
        return cv2.cvtColor(self.mat, color_scheme)
