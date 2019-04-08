from .base import BasePreprocessor
import cv2


class CudaPreprocessor(BasePreprocessor):
    """
    This class uses CUDA for all pre-processing operations.

    Do not use this class unless you understand the implications, mainly converting cv2.cuda_GpuMat to numpy.ndarray and
    back having a very high amount of latency. Use the generic preprocessor instead.
    """

    def resize(self, width: int, height: int) -> cv2.cuda_GpuMat:
        """
        Resize the given image to a specific width and height.
        """
        return cv2.resize(self.mat, (width, height)) # pragma: no cover

    def recolor(self, color_scheme: int) -> cv2.cuda_GpuMat:
        """
        Recolor an image to another color scheme.
        """
        return cv2.cvtColor(self.mat, color_scheme) # pragma: no cover
