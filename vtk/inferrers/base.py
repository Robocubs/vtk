from abc import ABC, abstractmethod
from typing import Any


class BaseInferrer(ABC):  # pragma: no cover
    """
    Base abstract class used for inference on models.
    Main implementations are TensorFlow (auto optimized), TensorFlow + TRT (auto optimized) and OpenCV DNN
    (not optimized, not recommended).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Use an implementation class, not the base class.")
