import setuptools
from setuptools.extension import Extension
import version

with open("README.md", "r") as fh:
    long_description = fh.read()
    ext_name = "clusteringForVarietyIdentification"

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().split("\n")

setuptools.setup(
    name=ext_name,
    version=version.__version__,
    description="IDM's pipeline for identifying crop varieties from genotyping data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InstituteforDiseaseModeling/clusteringForVarietyIdentification",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements
)