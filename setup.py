
from setuptools import setup, find_packages
import os
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('
setup(
    name="gene-regulation-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning model for gene regulation network prediction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="YOUR_PATH_HERE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gene-regulation-train=run:main",
            "gene-regulation-cv=cross_validate_ecoli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    keywords=[
        "deep-learning",
        "gene-regulation",
        "bioinformatics",
        "pytorch",
        "graph-neural-networks",
        "attention-mechanism",
    ],
    project_urls={
        "YOUR_PATH_HERE",
        "YOUR_PATH_HERE",
        "Documentation": "https://github.com/yourusername/gene-regulation-prediction
    },
) 