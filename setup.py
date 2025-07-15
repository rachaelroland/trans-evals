from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trans-evals",
    version="0.1.0",
    author="Rachael Roland",
    author_email="",
    description="A comprehensive framework for evaluating LLM bias toward trans and non-binary individuals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rachaelroland/trans-evals",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "trans-evals=trans_evals.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "trans_evals": ["configs/*.yaml", "data/*.json"],
    },
)