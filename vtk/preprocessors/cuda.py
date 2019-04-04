from .base import BasePreprocessor
import cv2

class CudaPreprocessor(BasePreprocessor):
    """
    This class uses CUDA for all pre-processing operations.
    """
    def resize(self, width: int, height: int) -> cv2.cuda_GpuMat:
    	"""
    	Resize the given image to a specific width and height.
    	"""
    	return cv2.resize(self.mat, (width, height))
    def recolor(self, color_scheme: int) -> cv2.cuda_GpuMat:
    	"""
    	Recolor an image to another color scheme.
    	"""
    	return cv2.cvtColor(self.mat, color_scheme)