Preprocessors
=============

.. image:: https://images.unsplash.com/photo-1514986888952-8cd320577b68?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1920&q=80

*Photo by Alyson McPhee on Unsplash*

Preprocessors take an initial image from a webcam or file, and transform it into a processable image, typically by resizing the image or converting the color space of the image to a proper format.

These preprocessors are derived from the `BasePreprocessor <https://github.com/Robocubs/vtk/tree/master/vtk/preprocessors/base.py>`_ class, which defines the methods that all preprocessors need to have.

Documented below are the main methods of preprocessor classes.

.. py:function:: resize(image, width, height)

   Resize an image.

   :param np.ndarray image: The image object to resize.
   :param int width: The width of the resulting image.
   :param int height: The height of the resulting image.
   :return: The resized image object.
   :rtype: np.ndarray

.. py:function:: recolor(image, colorspace)
   
   Change the color space of an image using OpenCV's color conversion methods.

   :param np.ndarray image: Image to change color space of.
   :param int colorspace: Color space to change image to. Comes from cv2.COLOR_* enum.
   :return: The recolored image object.
   :rtype: np.ndarray