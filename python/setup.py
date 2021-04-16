from setuptools import setup
from setuptools_rust import RustExtension

setup(
  name="plant-python",
  version="0.1.0",
  packages=["plant"],
  rust_extensions=[RustExtension("plant.plant")],
  zip_safe=False,
)
