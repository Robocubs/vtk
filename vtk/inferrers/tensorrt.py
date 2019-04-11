from .base import BaseInferrer
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import os


class TensorRTInferrer(BaseInferrer):
    """
    Run inference on a graph using NVIDIA TensorRT optimizations. Use this if you have an NVIDIA GPU and a TensorFlow
    binary with TensorRT support to accelerate inference time to almost real time.
    """

    def __init__(self, graph: str, precision: str = "int8", threshold: float = 0.8):
        """
        Initialize the graph with the specified frozen graph.
        :param graph: Path to graph file.
        :param precision: Use this if your graph is not quantized. Possible options are int8, float16, and float32.
        :param threshold: Threshold to determine how significant a detection is.
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.graphdef = tf.GraphDef()
        self.session = tf.Session()
        self.precision = precision
        self.threshold = threshold
        with self.session.graph.as_default():
            with open(graph, "rb") as f:
                self.graphdef.ParseFromString(f.read())
                self.trt_graph = trt.create_inference_graph(
                    input_graph_def=self.graphdef,
                    outputs=["num_detections:0", "detection_scores:0", "detection_boxes:0", "detection_classes:0"],
                    precision_mode=self.precision
                )
                tf.import_graph_def(self.graphdef, name="")
        super().__init__()

    def run(self, image: np.ndarray) -> dict:  # pragma: no cover
        """
        Run inference on a frame using the defined graph.
        :param image: Image to run inference on.
        :return: A dictionary with two elements: num_detections (length of detections list) and detections (dictionary
            with class ID, score and bounding box coordinates for each detection).
        """
        with self.session.as_default():
            out = self.session.run([
                self.session.graph.get_tensor_by_name("num_detections:0"),
                self.session.graph.get_tensor_by_name("detection_scores:0"),
                self.session.graph.get_tensor_by_name("detection_boxes:0"),
                self.session.graph.get_tensor_by_name("detection_classes:0")
            ], feed_dict={
                "image_tensor:0": image.reshape(1, image.shape[0], image.shape[1], 3)
            })
        rows = image.shape[0]
        cols = image.shape[1]
        num_detections = int(out[0][0])
        detections = []
        for i in range(num_detections):
            class_id = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > self.threshold:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                detections.append({
                    "classId": class_id,
                    "score": score,
                    "bbox": [x, y, right, bottom]
                })
        return {
            "num_detections": len(detections),
            "detections": detections
        }
