from setuptools import setup, find_packages
from pathlib import Path


def parse_requirements(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l and not l.startswith("#")]

setup(
    name="music_genre_classifier",
    version="0.1",
    packages=find_packages(include=["classifier", "classifier.*"]),
    install_requires=parse_requirements("flask/requirements.txt"),
)
