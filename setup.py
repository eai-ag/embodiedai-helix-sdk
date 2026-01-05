from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="embodiedai-helix-sdk",
    version="0.1.0",
    author="Embodied AI",
    description="A Python SDK to control Embodied AI's Helix robot arm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "roslibpy>=1.0.0",
    ],
)
