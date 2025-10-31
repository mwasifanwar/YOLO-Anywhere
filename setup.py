from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="yolo-anywhere",
    version="1.0.0",
    description="Zero-shot object detection with YOLO-World",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    author="M Wasif Anwar",
    author_email="your-email@example.com",
    url="https://github.com/mwasifanwar/YOLO-Anywhere",
    entry_points={
        'console_scripts': [
            'yolo-anywhere=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)