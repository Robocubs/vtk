from vtk.inferrers.opencv import OpenCVInferrer
import cv2
import numpy as np
import unittest
import os


def inference_helper(holder, image: np.ndarray):
    inferrer = OpenCVInferrer("testdata/models/frozen_inference_graph.pb", "testdata/models/cvgraph.pbtxt")
    inferrer.prepare()
    result = inferrer.run(image)
    holder.assertGreaterEqual(result["num_detections"], 1)


def inference_on_empty_image(holder):
    inferrer = OpenCVInferrer("testdata/models/frozen_inference_graph.pb", "testdata/models/cvgraph.pbtxt")
    inferrer.prepare()
    with holder.assertRaises(cv2.error):
        inferrer.run(np.zeros((300, 300), dtype=int))


class OpenCVInferrerTest(unittest.TestCase):
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