from abc import ABC, abstractmethod
import numpy as np

class BaseInferrer(ABC):
    """
    Base abstract class used for inference on models.
    Main implementations are TensorFlow (auto optimized), TensorFlow + TRT (auto optimized) and OpenCV DNN (not optimized, not recommended).

    @param session Pointer to session of TensorFlow to restore graph into.
    @param graph Path to graph file.
    @param descriptor Used only by OpenCV.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare the model for running inference. TensorFlow and Tensorflow with TensorRT load graph files and create pointers to loaded graphs for inference, and OpenCV with DNN loads graph and graph descriptor for inference.
        """
        pass

    @abstractmethod
    def run(self, image: np.ndarray, threshold: float = None) -> dict:
        """
        Run inference on a frame. Pretty much self explanitory.
        """
        pass