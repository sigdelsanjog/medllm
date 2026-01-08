"""
Setup script for llm-med package

This file defines how your package will be installed via pip.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="llm-med",
    version="0.1.0",  # Update this for each release
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A lightweight medical question-answering language model trained on MedQuAD dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medllm",  # Replace with your GitHub repo URL
    project_urls={
        "Bug Reports": "https://github.com/yourusername/medllm/issues",
        "Source": "https://github.com/yourusername/medllm",
        "Documentation": "https://github.com/yourusername/medllm#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*", "dataset", "logs", "scripts"]),
    classifiers=[
        # Development status
        "Development Status :: 3 - Alpha",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating systems
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "training": [
            "tensorboard>=2.10.0",
            "wandb>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medllm-generate=inference.generator:main",
            "medllm-train=training.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="medical nlp language-model transformer gpt pytorch healthcare qa",
)
