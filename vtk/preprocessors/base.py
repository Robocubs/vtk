from abc import ABC, abstractmethod
from typing import Union
import cv2
import numpy as np

class BasePreprocessor(ABC):
    """
    Base preprocessor class. All preprocessors inherit from this class.
    """
    def __init__(self, mat: np.ndarray):
        self.mat = mat
        super().__init__()

    @abstractmethod
    def resize(self, width: int, height: int) -> Union[np.ndarray, cv2.cuda_GpuMat]:
        pass

    @abstractmethod
    def recolor(self, color_scheme: int) -> Union[np.ndarray, cv2.cuda_GpuMat]:
        pass