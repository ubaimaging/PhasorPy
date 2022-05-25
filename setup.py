import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '1.0.2'
PACKAGE_NAME = 'PhasorPy'
AUTHOR = 'Bruno Schuty Teske'
AUTHOR_EMAIL = 'schutyteske@gmail.com'
URL = 'https://github.com/ubaimaging/phasorPy'

LICENSE = 'MIT'
DESCRIPTION = 'This is a library to performe phasor analysis in microscopy images'
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8') 
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy', 'tifffile', 'matplotlib', 'scikit-image'
      ]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)
