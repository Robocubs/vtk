from .base import BaseInferrer
from typing import Optional
import numpy as np
import cv2


class OpenCVInferrer(BaseInferrer):
    """
    Run the object detection graph in OpenCV's built-in DNN module.

    The use of this is NOT recommended!
    Here is why:
      * It produces inaccurate results.
      * It does not run on the GPU (CPU only).
      * It requires building OpenCV from source with the DNN module.
      * It requires a graph descriptor file produced from the frozen inference graph.
      * It does not provide detection class IDs or confidences, which can be used to determine the object type and how
        confident the network is in the result.
    It is only provided for the purposes of a fallback when TensorFlow does not work. Use at your own risk!
    """

    def __init__(self, graph: str, descriptor: str, input_size: Optional[int] = 300, threshold: Optional[float] = 0.8):
        """
        Initialize a new instance of the graph.
        :param graph: Path to graph file.
        :param descriptor: Path to graph descriptor file produced from frozen graph.
        :param input_size: Image input size from graph.
        :param threshold: Threshold to determine whether a detection is important or not.
        """
        self.graph = graph
        self.descriptor = descriptor
        self.input_size = input_size
        self.threshold = threshold
        self.detections = []
        self.net = cv2.dnn.readNetFromTensorflow(self.graph, self.descriptor)
        super().__init__()

    def run(self, image: np.ndarray) -> dict:
        """
        Run inference on a frame using the defined graph.
        :param image: Image to run inference on.
        :return: A dictionary with two elements: num_detections (length of detections list) and detections (dictionary
            with class ID, score and bounding box coordinates for each detection). Class ID and score fields are None.
        """
        rows = image.shape[0]
        cols = image.shape[1]
        self.net.setInput(
            cv2.dnn.blobFromImage(image, size=(self.input_size, self.input_size), swapRB=True, crop=False))
        outs = self.net.forward()
        for detection in outs[0, 0, :, :]:
            score = float(detection[2])
            if score > self.threshold:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                self.detections.append({
                    "classId": None,
                    "score": score,
                    "bbox": [int(left), int(top), int(right), int(bottom)]
                })
        return {
            "num_detections": len(self.detections),
            "detections": self.detections
        }
