from abc import ABC, abstractmethod
from typing import Union
import cv2

class BasePreprocessor(ABC):
	"""
	Base preprocessor class. All preprocessors inherit from this class.
	"""
	def __init__(self, mat: cv2.Mat):
		self.mat = mat
		super().__init__()

	@abstractmethod
	def resize(self, width: int, height: int) -> Union[cv2.Mat, cv2.cuda_GpuMat]:
		pass

	@abstractmethod
	def recolor(self, color_scheme: int) -> Union[cv2.Mat, cv2.cuda_GpuMat]:
		pass