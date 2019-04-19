Installation
============

.. image:: https://images.unsplash.com/photo-1550221997-0f417b27dd74?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1920&q=80

*Photo by Brett Jordan on Unsplash*

To install VTK, you need to have a few things prepared in advance. Mainly, you need Nvidia CUDA installed first; check the `README file <https://github.com/Robocubs/vtk/tree/master/README.md>`_ to see how to install it.

OpenCV and TensorFlow are the essential dependencies of VTK. You have a few options here:

* Build both OpenCV and TensorFlow from source for maximum optimization (recommended for those with nothing else to do with their time).
* Build just TensorFlow from source for maximum speed on inference (recommended for those willing to do so).
* Install prebuilt varieties of both TensorFlow and OpenCV from PyPI (recommended for beginners).

We will **not** be covering building OpenCV and TensorFlow from source, as every computer setup is different and every software difference contributes to a different solution being used.

To install the beginner's set of dependencies with VTK, follow the instructions below:

#. Install Python 3.6 (or newer) on your machine. **Do not use Python 2. It will not work and issues resulting from Python 2 will be closed as "won't fix".**
#. Using the `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ tool, which you can install in a terminal using the command ``pip install virtualenv``, create a new virtual environment:

	* ``virtualenv -p python3 venv``

#. Activate this new virtual environment using the ``source`` command:

	* ``source venv/bin/activate``

#. Clone VTK from GitHub using the ``git`` command:

	* ``git clone https://github.com/Robocubs/vtk``

#. Enter the VTK folder and install all required packages using the ``pip3`` command:

	* ``cd vtk``
	* ``pip3 install -r requirements.txt``
	* **or** ``pip3 install -r .travis-requirements.txt`` if your computer does not have an NVIDIA GPU

#. Verify that the requirements have been correctly installed by running VTK's test suite:

	* ``cd tests``
	* ``coverage run --source=../vtk/ -m nose2``
	* ``cd ..``

#. Finally, install VTK in your virtual environment using the below commands:

	* ``python3 setup.py install``

Alright! Now, it's time to start using VTK in your project.