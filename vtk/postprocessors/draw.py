from .base import BasePostprocessor
import cv2
import numpy as np


class DrawingPostprocessor(BasePostprocessor):
    """
    This postprocessor draws the boxes on an image.
    """
    def __init__(self):
        super().__init__()

    def prepare(self):  # pragma: no cover
        raise NotImplementedError("Don't call the prepare method on the DrawingPostprocessor class. It is a stub.")

    def run(self, image: np.ndarray, inference: dict) -> np.ndarray:
        """
        Draw the resulting boxes on an inferred image.
        :param image: Image to draw boxes on.
        :param inference: Output from an inference class.
        :return: Image with boxes drawn.
        """
        for i in inference["detections"]:
            cv2.rectangle(image, (int(i["bbox"][0]), int(i["bbox"][1])), (int(i["bbox"][2]), int(i["bbox"][3])), (125, 255, 51), 2)
        return image

    def close(self):  # pragma: no cover
        raise NotImplementedError("Don't call the close method on the DrawingPostprocessor class. It is a stub.")
