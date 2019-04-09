from .base import BaseInferrer
import tensorflow as tf
import numpy as np
import os


class TensorFlowInferrer(BaseInferrer):
    """
    Run inference on a graph using plain TensorFlow. Use this if you don't have TensorRT, are using an AMD GPU, do not want to use TensorRT, or want to run inference exclusively on the CPU (not recommended).
    """

    def __init__(self, graph: str):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.graphdef = tf.GraphDef()
        self.session = tf.Session()
        with self.session.graph.as_default():
            with open(graph, "rb") as f:
                self.graphdef.ParseFromString(f.read())
                tf.import_graph_def(self.graphdef, name="")
        super().__init__()

    def prepare(self) -> None:
        """
        Prepare the model for inference. This loads the model into memory, if not already completed.
        """
        pass

    def run(self, image: np.ndarray, threshold: float = 0.8) -> dict:
        """
        Run inference on an image.
        """
        with self.session.as_default():
            self.out = self.session.run([
                self.session.graph.get_tensor_by_name("num_detections:0"),
                self.session.graph.get_tensor_by_name("detection_scores:0"),
                self.session.graph.get_tensor_by_name("detection_boxes:0"),
                self.session.graph.get_tensor_by_name("detection_classes:0")
            ], feed_dict={
                "image_tensor:0": image.reshape(1, image.shape[0], image.shape[1], 3)
            })
        rows = image.shape[0]
        cols = image.shape[1]
        num_detections = int(self.out[0][0])
        detections = []
        for i in range(num_detections):
            classId = int(self.out[3][0][i])
            score = float(self.out[1][0][i])
            bbox = [float(v) for v in self.out[2][0][i]]
            if score > threshold:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                detections.append({
                    "classId": classId,
                    "score": score,
                    "bbox": [x, y, right, bottom]
                })
        return {
            "num_detections": len(detections),
            "detections": detections
        }
