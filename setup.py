# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-st",
    version="0.1.0",
    author="Zeyu Zou",
    # author_email="your.email@example.com",
    description="Retrieval-Augmented Generation for Spatial Transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-st",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ragst-train=ragst.scripts.train:main",
            "ragst-predict=ragst.scripts.predict:main",
            "ragst-download-data=ragst.scripts.download_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ragst": ["configs/*.yaml", "data/gene_lists/*.txt"],
    },
)