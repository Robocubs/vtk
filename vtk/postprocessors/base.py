from abc import ABC, abstractmethod


class BasePostprocessor(ABC):  # pragma: no cover
    """
    Base postprocessor class. This is a class takes in information from an inferrer and runs a specified
    operation on the input.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare(self, *args, **kwargs):
        raise NotImplementedError("Don't use the base postprocessor class. Use an implementation.")

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError("Don't use the base postprocessor class. Use an implementation.")

    @abstractmethod
    def close(self, *args, **kwargs):
        raise NotImplementedError("Don't use the base postprocessor class. Use an implementation.")
