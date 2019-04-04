from .base import BasePreprocessor
import cv2

class GenericPreprocessor(BasePreprocessor):
	"""
	A generic preprocessor that uses the CPU for its calculations.
	"""
	def resize(self, width: int, height: int) -> cv2.Mat:
		"""
		Resize the given image to a specific width and height.
		"""
		return cv2.resize(self.mat, (width, height))
	def recolor(self, color_scheme: int) -> cv2.Mat:
		"""
		Recolor an image to another color scheme.
		"""
		return cv2.cvtColor(self.mat, color_scheme)