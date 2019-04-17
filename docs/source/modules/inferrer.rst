Inferrers
=========

.. image:: https://images.unsplash.com/photo-1424223022789-26fd8f34bba2?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1920&q=80

*Photo by Laura Lee Moreau on Unsplash*

Inferrers are modules that use either TensorFlow or OpenCV to find objects in images with a provided trained object detection model.

These inferrers are derived from the `BaseInferrer <https://github.com/Robocubs/vtk/tree/master/vtk/inferrers/base.py>`_ class, which defines the methods that all inferrers need to have.

Documented below are these main methods.

.. py:function:: TensorFlowInferrer(graph)

	Initialize an inferrer with a path to the graph to load. The same applies to the TensorRTInferrer class.

	:param str graph: Relative path to the graph file.

.. py:function:: OpenCVInferrer(graph, descriptor, input_size = 300)

	OpenCV requires a descriptor file to interpret the graph properly. It also requires an image input size, but it has been set to a default of 300, since most models use this as their default image input size.

	:param str graph: Relative path to the graph file.
	:param str descriptor: Relative path to the descriptor file. See `this link <https://git.io/fjYFr>`_ for more information.
	:param int input_size: Default 300. Set this to change the input blob size with OpenCV.

.. py:function:: run(image)
	
	Run inference on an image. All inferrers implement this method.

	:param np.ndarray image: Image to run inference on.
	:returns: Dictionary with two elements, num_detections, and detections, which is a list of dictionaries with the detection class, confidence and bounding box coordinates in the form of [top_x, top_y, bottom_x, bottom_y].
	:rtype: dict
