from setuptools import setup, find_packages

# setup.py is used to package your project so it can be installed
# locally (pip install -e .) or shared with others (via PyPI).
# This makes your repo look polished and professional.

setup(
    name="music-genre-classifier",   # Project name
    version="0.1.0",                 # Initial version
    author="Your Name",
    author_email="your.email@example.com",
    description=(
        "A deep learning project for classifying music into genres "
        "and recommending similar tracks using audio features and embeddings."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",

    url="https://github.com/yourusername/music-genre-classifier",  # Update with your repo link

    # Automatically find all packages inside src/
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Source code lives in src/

    # Dependencies required to run the project
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "librosa",
        "scikit-learn",
        "torch",
        "torchvision",
    ],

    # Extra dependencies for development (testing, linting, notebooks)
    extras_require={
        "dev": ["jupyter", "pytest", "black", "flake8"]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    python_requires=">=3.8",
)
