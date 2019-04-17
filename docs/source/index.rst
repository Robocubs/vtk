.. VTK documentation master file, created by
	sphinx-quickstart on Fri Apr 12 11:04:11 2019.
	You can adapt this file completely to your liking, but it should at least
	contain the root `toctree` directive.

VTK: Object Detection for Humans
================================

.. image:: https://travis-ci.com/Robocubs/vtk.svg?branch=master
	:target: https://travis-ci.com/Robocubs/vtk

.. image:: https://coveralls.io/repos/github/Robocubs/vtk/badge.svg?branch=master
	:target: https://coveralls.io/github/Robocubs/vtk?branch=master

VTK is the simple machine learning toolkit that does away with complex APIs provided by the various machine learning frameworks. To put it in simple terms, it's the only non-GMO machine learning abstraction, safe for human consumption.

.. note:: The use of **Python 3** is *highly* recommended over Python 2. Don't even try using VTK without Python 3 -- we make use of many features unavailable in Python 2, like Unicode by default and type hints, both of which make developing applications significantly less painful! If you are already using Python 3, congratulations -- you are indeed a person of excellent taste.

-------------------

**Behold, the power of VTK**::
	
	>>> from vtk.preprocessors.generic import GenericPreprocessor
	>>> from vtk.inferrers.tensorrt import TensorRTInferrer
	>>> from vtk.postprocessors.draw import DrawingPostprocessor
	>>> pre = GenericPreprocessor()
	>>> infer = TensorRTInferrer(graph)
	>>> post = DrawingPostprocessor()
	>>> recolored = pre.recolor(image, cv2.COLOR_BGR2RGB)
	>>> output = infer.run(image)
	>>> final = post.run(image, output)

It's really that easy! The image stored in the last variable is the initial input image with the network output drawn as rectangles.

**Now, the same code, without VTK**::
	
	>>> import tensorflow as tf
	>>> import tensorflow.contrib.tensorrt as trt
	>>> import numpy as np
	>>> import cv2
	>>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	>>> with session.graph.as_default():
	>>>     with open(graph, "rb") as f:
	>>>         graphdef.ParseFromString(f.read())
	>>>         trt_graph = trt.create_inference_graph(
	>>>             input_graph_def=graphdef,
	>>>             outputs=["num_detections:0", "detection_scores:0", "detection_boxes:0", "detection_classes:0"],
	>>>             precision_mode=self.precision
	>>>         )
	>>>         tf.import_graph_def(graphdef, name="")
	>>>     out = session.run([
	>>>         session.graph.get_tensor_by_name("num_detections:0"),
	>>>         session.graph.get_tensor_by_name("detection_scores:0"),
	>>>         session.graph.get_tensor_by_name("detection_boxes:0"),
	>>>         session.graph.get_tensor_by_name("detection_classes:0")
	>>>     ], feed_dict={
	>>>         "image_tensor:0": image.reshape(1, image.shape[0], image.shape[1], 3)
	>>>     })
	>>> rows = image.shape[0]
	>>> cols = image.shape[1]
	>>> num_detections = int(out[0][0])
	>>> detections = []
	>>> for i in range(num_detections):
	>>>     class_id = int(out[3][0][i])
	>>>     score = float(out[1][0][i])
	>>>     bbox = [float(v) for v in out[2][0][i]]
	>>>     if score > self.threshold:
	>>>         x = bbox[1] * cols
	>>>         y = bbox[0] * rows
	>>>         right = bbox[3] * cols
	>>>         bottom = bbox[2] * rows
	>>>         detections.append({
	>>>              "classId": class_id,
	>>>              "score": score,
	>>>              "bbox": [x, y, right, bottom]
	>>>         })
	>>> for i in detections["detections"]:
	>>>     cv2.rectangle(image, (i["bbox"][0], i["bbox"][1]), (i["bbox"][2], i["bbox"][3]), (125, 125, 0), thickness=2)

VTK gets rid of all the annoying boilerplate code that you would have needed in the first place. Easy as that!

The Guide
---------

.. toctree::
   :maxdepth: 2

   setup/introduction
   setup/install
   setup/modelselect
   setup/usage

The Developer's Manual
----------------------

.. toctree::
	:maxdepth: 3

	modules/preprocessor