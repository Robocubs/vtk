from .base import BasePreprocessor
from typing import Union
from tensorflow import test as tftest
import numpy as np
import cv2


class CudaPreprocessor(BasePreprocessor):
    """
    This class uses CUDA for all pre-processing operations.

    Do not use this class unless you understand the implications, mainly converting cv2.cuda_GpuMat to numpy.ndarray and
    back having a very high amount of latency. Use the generic preprocessor instead.
    """
    def __init__(self, image: Union[np.ndarray, cv2.cuda_GpuMat]):
        if not tftest.is_gpu_available():
            raise SyntaxError("Cannot use CUDA when CUDA is not present. You need both CUDA and a version of OpenCV 4 "
                              "compiled from source.")
        if not isinstance(image, cv2.cuda_GpuMat):
            image = cv2.cuda_GpuMat(image)
        super().__init__(image)

    def resize(self, width: int, height: int) -> cv2.cuda_GpuMat:
        """
        Resize the given image to a specific width and height.
        """
        out = cv2.cuda_GpuMat()
        cv2.cuda.resize(self.mat, out, (width, height))  # pragma: no cover
        return out

    def recolor(self, color_scheme: int) -> cv2.cuda_GpuMat:
        """
        Recolor an image to another color scheme.
        """
        out = cv2.cuda_GpuMat()
        cv2.cuda.cvtColor(self.mat, out, color_scheme)  # pragma: no cover
        return out
