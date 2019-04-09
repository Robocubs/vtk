from abc import ABC, abstractmethod
from typing import Union
import cv2
import numpy as np


class BasePreprocessor(ABC):
    """
    Base preprocessor class. All preprocessors inherit from this class.
    """

    def __init__(self, mat: Union[np.ndarray, cv2.cuda_GpuMat]):  # pragma: no cover
        self.mat = mat
        super().__init__()

    @abstractmethod
    def resize(self, width: int, height: int) -> Union[np.ndarray, cv2.cuda_GpuMat]:  # pragma: no cover
        pass

    @abstractmethod
    def recolor(self, color_scheme: int) -> Union[np.ndarray, cv2.cuda_GpuMat]:  # pragma: no cover
        pass
