from vtk.preprocessors.cuda import CudaPreprocessor
import cv2, unittest
from tensorflow import test as tftest
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

is_gpu_available = not tftest.is_gpu_available()
gpu_unavailable_message = "CUDA is unavailable on this computer, skipping this test."


def resize_helper(handle, name: str):
    """
    Run resize test on GPU.
    :param handle: Instance of "self" from test method.
    :param name: Name of file to get original and expected results.
    :return: None. Just passes the test or fails it.
    """
    image = cv2.imread("testdata/originals/{name}.jpg".format(name=name), 1)
    resized = cv2.imread("testdata/resized/{name}.jpg".format(name=name), 1)
    preproc = CudaPreprocessor(image)
    resized_check = cv2.cuda_GpuMat.download(preproc.resize(300, 300))
    if isinstance(resized_check, np.ndarray):
        if resized.shape == resized_check.shape:
            # Mostly equal, but let's make sure...
            difference = cv2.subtract(resized, resized_check)
            b, g, r = cv2.split(difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                # Definitively equal.
                handle.assertTrue(True)  # Pass the test.
    else:
        handle.fail("Something went wrong. Check the conversion from cv2.cuda_GpuMat to np.ndarray.")


def recolor_helper(handle, name: str):
    """
    Run recolor test on GPU.
    :param handle: Instance of "self" from test method.
    :param name: Name of file to get original and expected results.
    :return: None. Just passes the test or fails it.
    """
    image = cv2.imread("testdata/originals/{name}.jpg".format(name=name), 1)
    recolored = cv2.imread("testdata/cvtcolor/{name}.jpg".format(name=name), 1)
    preproc = CudaPreprocessor(image)
    recolored_check = cv2.cuda_GpuMat.download(preproc.recolor(cv2.COLOR_RGB2BGR))
    if isinstance(recolored_check, np.ndarray):
        if recolored.shape == recolored_check.shape:
            # Mostly equal, but let's make sure...
            difference = cv2.subtract(recolored, recolored_check)
            r, g, b = cv2.split(difference)
            if cv2.countNonZero(r) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(b) == 0:
                handle.assertTrue(True)  # Definitely true.
    else:
        handle.fail("Something went wrong. Check the conversion from cv2.cuda_GpuMat to np.ndarray.")

@unittest.skipIf(is_gpu_available, gpu_unavailable_message)
class CudaPreprocessorTest(unittest.TestCase):
    def test_ball_zero_resize(self):
        resize_helper(self, "ball_0")

    def test_ball_zero_recolor(self):
        recolor_helper(self, "ball_0")

    def test_ball_five_resize(self):
        resize_helper(self, "ball_5")

    def test_ball_five_recolor(self):
        recolor_helper(self, "ball_5")

    def test_ball_ten_resize(self):
        resize_helper(self, "ball_10")

    def test_ball_ten_recolor(self):
        recolor_helper(self, "ball_10")

    def test_ball_fifteen_resize(self):
        resize_helper(self, "ball_15")

    def test_ball_fifteen_recolor(self):
        recolor_helper(self, "ball_15")

    def test_ball_twenty_resize(self):
        resize_helper(self, "ball_20")

    def test_ball_twenty_recolor(self):
        recolor_helper(self, "ball_20")

    def test_ball_twentyfive_resize(self):
        resize_helper(self, "ball_25")

    def test_ball_twentyfive_recolor(self):
        recolor_helper(self, "ball_25")

    def test_ball_thirty_resize(self):
        resize_helper(self, "ball_30")

    def test_ball_thirty_recolor(self):
        recolor_helper(self, "ball_30")

    def test_hatch_zero_resize(self):
        resize_helper(self, "hatch_0")

    def test_hatch_zero_recolor(self):
        recolor_helper(self, "hatch_0")

    def test_hatch_five_resize(self):
        resize_helper(self, "hatch_5")

    def test_hatch_five_recolor(self):
        recolor_helper(self, "hatch_5")

    def test_hatch_ten_resize(self):
        resize_helper(self, "hatch_10")

    def test_hatch_ten_recolor(self):
        recolor_helper(self, "hatch_10")

    def test_hatch_fifteen_resize(self):
        resize_helper(self, "hatch_15")

    def test_hatch_fifteen_recolor(self):
        recolor_helper(self, "hatch_15")

    def test_hatch_twenty_resize(self):
        resize_helper(self, "hatch_20")

    def test_hatch_twenty_recolor(self):
        recolor_helper(self, "hatch_20")

    def test_hatch_twentyfive_resize(self):
        resize_helper(self, "hatch_25")

    def test_hatch_twentyfive_recolor(self):
        recolor_helper(self, "hatch_25")

    def test_hatch_thirty_resize(self):
        resize_helper(self, "hatch_30")

    def test_hatch_thirty_recolor(self):
        recolor_helper(self, "hatch_30")
