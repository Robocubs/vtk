from vtk.inferrers.tensorrt import TensorRTInferrer
from tensorflow import test as tftest
import cv2
import numpy as np
import unittest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
is_gpu_available = not tftest.is_gpu_available()
gpu_unavailable_message = "CUDA is unavailable on this computer, skipping this test."
inferrer = TensorRTInferrer("testdata/models/frozen_inference_graph.pb")


def inference_helper(holder, image: np.ndarray):
    result = inferrer.run(image)
    holder.assertGreaterEqual(result["num_detections"], 1)


def inference_on_empty_image(holder):
    with holder.assertRaises(ValueError):
        inferrer.run(np.zeros((300, 300), dtype=int))


@unittest.skipIf(is_gpu_available, gpu_unavailable_message)
class TensorRTInferrerTest(unittest.TestCase):
    def test_inference_ball_1(self):
        image = cv2.imread("testdata/detect/Ball_001.jpg")
        inference_helper(self, image)

    def test_inference_ball_2(self):
        image = cv2.imread("testdata/detect/Ball_002.jpg")
        inference_helper(self, image)

    def test_inference_ball_3(self):
        image = cv2.imread("testdata/detect/Ball_003.jpg")
        inference_helper(self, image)

    def test_inference_ball_4(self):
        image = cv2.imread("testdata/detect/Ball_004.jpg")
        inference_helper(self, image)

    def test_inference_ball_5(self):
        image = cv2.imread("testdata/detect/Ball_005.jpg")
        inference_helper(self, image)

    def test_inference_ball_6(self):
        image = cv2.imread("testdata/detect/Ball_006.jpg")
        inference_helper(self, image)

    def test_inference_ball_7(self):
        image = cv2.imread("testdata/detect/Ball_007.jpg")
        inference_helper(self, image)

    def test_inference_ball_8(self):
        image = cv2.imread("testdata/detect/Ball_008.jpg")
        inference_helper(self, image)

    def test_inference_ball_9(self):
        image = cv2.imread("testdata/detect/Ball_009.jpg")
        inference_helper(self, image)

    def test_inference_ball_10(self):
        image = cv2.imread("testdata/detect/Ball_010.jpg")
        inference_helper(self, image)

    def test_inference_hatch_1(self):
        image = cv2.imread("testdata/detect/Hatch_001.jpg")
        inference_helper(self, image)

    def test_inference_hatch_2(self):
        image = cv2.imread("testdata/detect/Hatch_002.jpg")
        inference_helper(self, image)

    def test_inference_hatch_3(self):
        image = cv2.imread("testdata/detect/Hatch_003.jpg")
        inference_helper(self, image)

    def test_inference_hatch_4(self):
        image = cv2.imread("testdata/detect/Hatch_004.jpg")
        inference_helper(self, image)

    def test_inference_hatch_5(self):
        image = cv2.imread("testdata/detect/Hatch_005.jpg")
        inference_helper(self, image)

    def test_inference_hatch_6(self):
        image = cv2.imread("testdata/detect/Hatch_006.jpg")
        inference_helper(self, image)

    def test_inference_hatch_7(self):
        image = cv2.imread("testdata/detect/Hatch_007.jpg")
        inference_helper(self, image)

    def test_inference_hatch_8(self):
        image = cv2.imread("testdata/detect/Hatch_008.jpg")
        inference_helper(self, image)

    def test_inference_hatch_9(self):
        image = cv2.imread("testdata/detect/Hatch_009.jpg")
        inference_helper(self, image)

    def test_inference_hatch_10(self):
        image = cv2.imread("testdata/detect/Hatch_010.jpg")
        inference_helper(self, image)

    def test_inference_zeros(self):
        inference_on_empty_image(self)
