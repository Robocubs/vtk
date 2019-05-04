# VTK

[![Build Status](https://travis-ci.com/Robocubs/vtk.svg?branch=master)](https://travis-ci.com/Robocubs/vtk)
[![Coverage Status](https://coveralls.io/repos/github/Robocubs/vtk/badge.svg?branch=master)](https://coveralls.io/github/Robocubs/vtk?branch=master)
[![Documentation Status](https://readthedocs.org/projects/vtk/badge/?version=latest)](https://vtk.readthedocs.io/en/latest/?badge=latest)

An easy-to-use vision toolkit for working with SSD and SSD-Lite TensorFlow Object Detection models.

### Who?

VTK was created by Nicholas Hubbard, computer vision lead on Team 1701, The Robocubs, out of Detroit, MI.

### What?

VTK defines a common API for pre-processors, inferrers and post-processors for images to run through deep learning projects. We also provide tests to verify the functionality of these APIs and example programs to work with the API.

### When?

VTK was created in a code sprint at the Alpena #2 event in Alpena, MI for the 2019 FIRST Deep Space game. It won the autonomous award for its unparalleled accuracy and calibration-free detection.

### Where?

Your robot, hopefully! VTK can be used anywhere with an NVIDIA graphics card and a Python interpreter.

### Why?

Computer vision has always been a juggling act. You will eventually get lighting and distance calibrations right - until the lighting on the field changes. You will get the filtering algorithms used right - until you test it and the algorithm runs at extremely low frame rates, to the point of near unusability.

Enter VTK. VTK makes it easy to use a previously unattainable vision goal - calibration-free object detection - with just a change in the hardware platform you use. With VTK, deep learning algorithms can be used by anyone with an NVIDIA graphics card and some Python experience.

### How?

VTK is built on the TensorFlow Object Detection library, created by a generous team of Google researchers who created a modular platform for working with the thousands of object detection algorithm combinations.

Atop this platform, VTK provides optimization libraries (such as TensorRT) and transformations (like color space conversions and image resizing) for working with images for running inference on.

### Can I use it?

Of course you can! Follow the instructions below to work with VTK. This guide assumes you are using Ubuntu for your development platform, as it's the easiest platform to work with on this project.

##### Installing Prerequisites

You can safely skip to step 3 if you are running on an Nvidia Jetson, as it is included in the JetPack developer kit.

<ol>
	<li>Download the NVIDIA driver and CUDA <b>10.0</b> (this is important) from <a href="https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64">this link</a>. Go eat a donut or something while it downloads.</li>
	<li>Install the NVIDIA driver and CUDA onto your machine by issuing this sequence of commands:</li>
	<ul>
		<li><code>sudo apt update</code></li>
		<li><code>sudo apt upgrade</code></li>
		<li><code>sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb</code></li>
		<li>Literally hit the TAB key auto-complete the command below: <br><code>sudo apt-key add /var/cuda-repo-TAB/7fa2af80.pub</code></li>
		<li><code>sudo apt update</code></li>
		<li><code>sudo apt install cuda</code></li>
	</ul>
	<li>Install Python and its associated dependencies with this command:</li>
	<ul>
		<li><code>sudo apt install python3 python3-dev</code></li>
	</ul>
</ol>

##### Installing and Running VTK

<ol>
	<li>Install Git, if it isn't already installed:</li>
	<ul>
		<li><code>sudo apt install git</code></li>
	</ul>
	<li>Download the project from GitHub:</li>
	<ul>
		<li><code>git clone https://github.com/Robocubs/vtk</code></li>
	</ul>
	<li>Install all the dependencies for the project with this command (this will take a while):</li>
	<ul>
		<li><code>cd vtk</code></li>
		<li><code>sudo python3 -m pip install -r requirements.txt</code></li>
	</ul>
	<li>Run the test program (make sure you have a webcam plugged in!):</li>
	<ul>
		<li><code>python3 testing.py</code></li>
	</ul>
	<li><i>Or</i>, run the unit tests (less interactive, more automatic):</li>
	<ul>
		<li><code>cd tests</code></li>
		<li><code>coverage run --source=../vtk/ -m nose2</code></li>
	</ul>
	<li>If either one of these two tests pass, you can now install VTK:</li>
	<ul>
		<li><code>python3 setup.py install</code></li>
	</ul>
</ol>

You're done! You can also use the `testing.py` and various unit test scripts as inspiration for your own projects. File an issue if you are having problems on your device. I can help whenever necessary.
