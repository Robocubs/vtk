from abc import ABC, abstractmethod
from typing import Any


class BasePreprocessor(ABC):  # pragma: no cover
    """
    Base preprocessor class. All preprocessors inherit from this class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def resize(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Don't use the base preprocessor class. Use an implementation.")

    @abstractmethod
    def recolor(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Don't use the base preprocessor class. Use an implementation.")
