from setuptools import setup, find_packages


setup(
    name="music_genre_classifier",
    version="0.1",
    packages=find_packages(include=["classifier", "classifier.*"]),
    install_requires=[
        "tensorflow-cpu==2.18.0",
        "keras==3.8.0",
        "numpy==2.0.2",
        "scikit-learn==1.6.1",
        "matplotlib==3.10.0",
        "librosa==0.10.2.post1",
        "mlflow==2.22.0",
        "prefect==2.20.19",
    ]
)
