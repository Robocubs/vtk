import unittest
import random
import numpy as np
import cv2
from vtk.postprocessors.draw import DrawingPostprocessor


def create_random_image() -> np.ndarray:
    """
    Generate a blank image to use for the tests.
    :return:
    """
    return np.zeros((800, 600, 3), dtype=np.uint8)


def generate_positive_detections():
    """
    Generate random inputs for the test cases.
    :return: Dictionary mimicking the structure of the inferrer output.
    """
    detections = []
    for i in range(10):
        detections.append({
            "classId": int(random.choice(range(15))),
            "score": float(random.random()),
            "bbox": [
                random.choice(range(200)),
                random.choice(range(150)),
                100 + random.choice(range(200)),
                100 + random.choice(range(150))
            ]
        })
    return {
        "num_detections": len(detections),
        "detections": detections
    }


def generate_negative_detections():
    """
    Generate random inputs for the test cases that do not pass.
    :return: Dictionary mimicking the structure of the inferrer output.
    """
    detections = []
    for i in range(10):
        detections.append({
            "classId": int(random.choice(range(15))),
            "score": float(random.random()),
            "bbox": [
                800 + random.choice(range(200)),
                600 + random.choice(range(150)),
                1000 + random.choice(range(200)),
                1150 + random.choice(range(150))
            ]
        })
    return {
        "num_detections": len(detections),
        "detections": detections
    }


class DrawingPostprocessorTest(unittest.TestCase):
    def setUp(self):
        self.postprocessor = DrawingPostprocessor()
        self.image: np.ndarray = create_random_image()
        self.positive_detections: dict = generate_positive_detections()
        self.negative_detections: dict = generate_negative_detections()

    def test_draw_positive(self):
        image = self.postprocessor.run(self.image, self.positive_detections)
        self.assertIsInstance(image, np.ndarray)

    def test_draw_negative(self):
        image = self.postprocessor.run(self.image, self.negative_detections)
        self.assertIsInstance(image, np.ndarray)
