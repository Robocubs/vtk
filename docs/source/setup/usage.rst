Using VTK
=========

.. image:: https://images.unsplash.com/photo-1519162721257-18cd195350c2?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1920&q=80

*Photo by NeONBRAND on Unsplash*

Now we get to the good part: using VTK for your own vision project.

VTK is built in a set of modular components that follow a simple specification. These components are themselves are built on the `abstract base classes <https://docs.python.org/3/library/abc.html>`_ system.

To try it out, after following the instructions in the Installation document, open a Python shell in your terminal with the ``python3`` command in the VTK folder, and try it out.::

   >>> from vtk.inferrers.tensorflow import TensorFlowInferrer
   >>> from vtk.postprocessors.draw import DrawingPostprocessor
   >>> import cv2
   >>> image = cv2.imread("tests/testdata/originals/ball_0.jpg")
   >>> inferrer = TensorFlowInferrer("tests/testdata/models/frozen_inference_graph.pb")
   >>> postprocessor = DrawingPostprocessor
   >>> results = inferrer.run(image)
   >>> output = postprocessor.run(image, results)
   >>> cv2.imwrite("output.jpg", output)
   >>> exit()

The new file created, ``output.jpg``, is the input image with the resulting detection boxes drawn on it.

You can use these components to build your own programs based on VTK. The component architecture makes it really easy to make your own programs to build your own programs.