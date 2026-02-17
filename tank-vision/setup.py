from setuptools import setup, find_packages

setup(
    name="tank-vision",
    version="0.1.0",
    description="Military Tank Camera AI System - YOLOv11 based object detection and threat assessment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ultralytics>=8.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "albumentations>=1.3.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "lapx>=0.5.2",
    ],
    extras_require={
        "data": [
            "huggingface-hub>=0.20.0",
            "roboflow>=1.1.0",
            "pycocotools>=2.0.7",
        ],
        "dev": [
            "pytest>=7.4.0",
        ],
    },
)
