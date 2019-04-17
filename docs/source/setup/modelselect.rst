Selecting a Model
=================

.. image:: https://images.unsplash.com/photo-1496769843785-93aa0be525dc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1920&q=80

*Photo by Philipp Lublasser on Unsplash*

VTK does not use the typical computer vision concepts of *calibration* or *thresholding*. Instead, it operates using *models*, which are the representations of what a machine learning training session has inferred from the training data provided to it. 

Models have to be *trained* on an extremely large dataset of true positives (the model correctly predicts the correct class), false positives (the model incorrectly predicts the positive class) , false negatives (the model incorrectly predicts the negative class), and true negatives (the model correctly predicts the negative class).

This training process takes a large amount of time and lots of computing power (usually rented on a cloud service) to create, which typically costs a lot of money to complete.

Adventurous learners can build their own models from scratch or using premade templates that they train their data over, whereas those who are just starting with machine learning should be using pre-made models.

Luckily, we have created a model for you to use; you can find it in the ``tests/testdata/models`` folder. (Hint: The models almost always end with the extension ``.pb``. The other file, ``cvgraph.pbtxt``, is used with OpenCV when TensorFlow does not work.)

Email Nicholas Hubbard at ``nhubbard@users.noreply.github.com`` with your email address and what you want to do if you need help with creating your own model. You can also file an issue and we can send you in the right direction.