from setuptools import setup, find_packages

setup(
    name="adaptive_fusion_deepfake",
    version="1.0.0",
    description="Adaptive Fusion for Cross-Dataset Generalization in Deepfake Detection",
    author="Thesis Research Project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "opencv-python>=4.8.0",
        "PyWavelets>=1.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
