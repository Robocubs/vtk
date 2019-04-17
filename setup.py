"""
 /$$    /$$ /$$$$$$$$ /$$   /$$
| $$   | $$|__  $$__/| $$  /$$/
| $$   | $$   | $$   | $$ /$$/ 
|  $$ / $$/   | $$   | $$$$$/  
 \  $$ $$/    | $$   | $$  $$  
  \  $$$/     | $$   | $$\  $$ 
   \  $/      | $$   | $$ \  $$
    \_/       |__/   |__/  \__/
"""
import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

# Define package metadata here.
NAME = "vtk"
DESCRIPTION = "A vision toolkit to simplify use of neural networks."
URL = "https://github.com/Robocubs/vtk"
EMAIL = "nhubbard@users.noreply.github.com"
AUTHOR = "Nicholas Hubbard"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None

# Required packages.
REQUIRED = []
with open("requirements.txt") as f:
	for line in f:
		REQUIRED.append(line)

# Get and set current path.
here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use is as the long description.
try:
	with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
		LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
	LONG_DESCRIPTION = DESCRIPTION

# Load the __version__.py module from the package.
about = {}
if not VERSION:
	project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
	with open(os.path.join(here, project_slug, "__version__.py")) as f:
		exec(f.read(), about)
else:
	about["__version__"] = VERSION

# Add command to upload new version to both PyPI and GitHub.
class UploadCommand(Command):
	"""Support upload abilities."""
	description = "Build and publish package."
	user_options = []

	@staticmethod
	def status(s):
		"""Print things in bold."""
		print("\033[1m{0}\033[0m".format(s))

	def initialize_options(self):
		pass

	def finalize_options(self):
		pass

	def run(self):
		try:
			self.status("Removing any previous builds.")
			rmtree(os.path.join(here, "dist"))
		except OSError:
			pass

		self.status("Building distribution.")
		os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

		self.status("Uploading package to PyPI.")
		os.system("twine upload dist/*")

		self.status("Pushing Git tags...")
		os.system("git tag v{0}".format(about["__version__"]))
		os.system("git push --tags")

		sys.exit()

setup(
	name=NAME,
	version=about["__version__"],
	description=DESCRIPTION,
	long_description=LONG_DESCRIPTION,
	long_description_content_type="text/markdown",
	author=AUTHOR,
	author_email=EMAIL,
	python_requires=REQUIRES_PYTHON,
	url=URL,
	packages=find_packages(exclude=["tests", "examples"]),
	install_requires=REQUIRED,
	include_package_data=True,
	license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
    cmdclass={
    	"upload": UploadCommand
    }
)
