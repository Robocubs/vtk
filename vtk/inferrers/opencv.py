from .base import BaseInferrer
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
	  * It does not provide detection class IDs or confidences, which can be used to determine the object type and how confident the network is in the result.
	It is only provided for the purposes of a fallback when TensorFlow does not work. Use at your own risk!

	@param graph Path to frozen inference graph.
	@param descriptor Path to descriptor file.
	@param input_size Size of square image to provide to OpenCV.
	"""
	def __init__(self, graph: str, descriptor: str, input_size: int = 300):
		self.graph = graph
		self.descriptor = descriptor
		self.input_size = input_size
		self.detections = []
	def prepare(self) -> None:
		"""
		Prepare the model for inference. This loads the model into memory, if not already completed.
		"""
		self.net = cv2.dnn.readNetFromTensorflow(self.graph, self.descriptor)
	def run(self, image: np.ndarray, threshold: float = 0.8) -> dict:
		"""
		Run inference on an image.
		"""
		self.rows = image.shape[0]
		self.cols = image.shape[1]
		self.net.setInput(cv2.dnn.blobFromImage(image, size=(self.input_size, self.input_size), swapRB=True, crop=False))
		self.outs = self.net.forward()
		for detection in self.outs[0, 0, :, :]:
			score = float(detection[2])
			if score > threshold:
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