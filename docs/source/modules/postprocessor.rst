Postprocessors
==============

.. image:: https://images.unsplash.com/photo-1464195244916-405fa0a82545?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1920&q=80

*Photo by Jade Wulfraat on Unsplash*

Postprocessors are the modules that take data from the preprocessors and inferrers, tying it together into one or more applicable pieces. The main implementation of a postprocessor provided by VTK is the ``DrawingPostprocessor`` class, which takes inference results and draws them on the image.

All postprocessors inherit from the `BasePostprocessor <https://github.com/Robocubs/vtk/tree/master/vtk/postprocessors/base.py>`_ class, which defines the below methods that all postprocessors must follow. 

.. py:function:: prepare(*args, **kwargs)
   
   This function sets the postprocessor up, if necessary. All arguments are set by the postprocessor itself.

.. py:function:: run(*args, **kwargs)
   
   This function runs the postprocessor action. All arguments are set by the postprocessor class.

.. py:function:: close(*args, **kwargs)

   This function closes the postprocessor session. All arguments are set by the postprocessor class.