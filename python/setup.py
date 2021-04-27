import sys
from setuptools import setup
from setuptools_rust import RustExtension

features = None
if '--gpu' in sys.argv:
  features = ['gpu-runtime']
  sys.argv.remove('--gpu')

setup(
  name="plant-python",
  version="0.1.0",
  packages=["plant"],
  rust_extensions=[RustExtension("plant.plant", features=features)],
  zip_safe=False,
)
