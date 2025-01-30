# Music Genre Classification with Deep Learning

<p align="center">
  <img src="images/music-genre-classification-project.png">
</p>

## Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Usage](#usage)
- [Details on the Code](#details-on-the-code)
- [Notebooks](#notebooks)
  - [EDA (Handling Audio Data)](#1-exploratory-data-analysis-eda)
  - [Data Preparation](#2-data-preparation)
  - [Model Selection](#3-model-selection)
- [Conclusion](#conclusion)

## Overview

This project implements a system that predicts the genre of a given audio file using deep learning techniques. By leveraging the GTZAN dataset and extracting Mel-Frequency Cepstral Coefficients (MFCC) features, the application processes the input audio and classifies it into one of 10 genres: rock, classical, metal, disco, blues, reggae, country, hiphop, jazz, and pop. The project employs neural networks (NN) and convolutional neural networks (CNN) to achieve this goal.

The application is designed to be user-friendly and operates via a Flask API, which allows users to upload an audio file and receive the predicted genre in JSON format.

This project is presented as the Capstone project for the 2024/2025 [Machine Learning Zoomcamp](https://github.com/arsonor/machine-learning-zoomcamp).

## Problem Statement

### Context

Music genre classification is a challenging task in machine learning, as it requires analyzing complex audio data and identifying patterns unique to specific genres. With the rise of digital music platforms, automating this classification has become increasingly important for organizing, recommending, and retrieving music effectively.

### Dataset Description

This project utilizes the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), often referred to as the "MNIST of sounds", a popular benchmark dataset for music genre recognition.

It was created by George Tzanetakis in 2001 and consists of 1000 audio tracks, each 30 seconds long, evenly distributed across 10 different genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

The key features of this dataset include:

1. **'genres_original' folder:**  
 The 1,000 audio files across the 10 genres folders.  
 The recordings were collected from diverse sources (CDs, radio, microphone recordings) to represent various audio conditions.

2. **'images_original' folder:**  
Mel spectrogram images of the audio files.

3. **CSV Features:**  
Audio features derived from the dataset. Features are computed for both 30-second and 3-second audio segments.

These diverse data formats make the dataset a robust choice for exploring multiple classification approaches. However, in the purpose of this project, I chose to focus only on the audio files in the folder 'genres original'.

The audio files are processed to extract **MFCC features (this audio feature is explained in the [`eda.ipynb`](notebooks/eda.ipynb) notebook)** and segment the 30-second audio files into 3-second clips for model training.

### Practical Applications

- Automated music organization and tagging.

- Music recommendation systems for streaming platforms.

- Enhancing search and discovery features in digital music libraries.

- Assisting creators in identifying or categorizing their work.

- Audio Analysis Tools: Develop tools for musicologists and researchers to analyze genre trends.

- Educational Platforms: Aid in music education by categorizing and recommending tracks for learning.


## Installation

### 1. Clone the repository (or use a Github Codespace)

```sh
git clone https://github.com/arsonor/music-genre-classification-with-deep-learning
cd music-genre-classification-with-deep-learning
```

### 2. Create a Virtual Environment

#### 1. Create a virtual environment using Python:

```sh
python -m venv venv
```

#### 2. Activate the virtual environment:

- On Windows (Command Prompt):  
  ```sh
  venv\Scripts\activate
  ```

- On Windows (Git Bash):  
  ```sh
  source venv/Scripts/activate
  ```

- On macOS/Linux:  
  ```sh
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file (it may take a few minutes):
```sh
pip install -r flask/requirements.txt
```

### 4. Run Docker

Start the Flask API and Nginx server using Docker Compose:
```sh
docker-compose up --build
```


## Usage

**To test the app:**

1. Place your audio file in the appropriate directory.

2. Update the client.py file with the correct server URL and file path:
    ```sh
    URL = "http://127.0.0.1:5050/predict"
    FILE_PATH = "test/blues.00000.wav"
    ```

3. Run the client script:
    ```sh
    python client.py
    ```

4. The predicted genre will be returned as JSON:
    ```sh
    { "genre": "blues" }
    ```


**To test the training & model saving or the prediction service:**

Run the train script:
  ```sh
  cd classifier
  python train.py
  ```

Run the prediction service (the file let you test the predictions on an audio file for each genre in the test folder):
  ```sh
  cd ..
  cd flask
  python genre_prediction_service.py
  ```





## Details on the Code

### Folder and File Structure

#### Classifier Folder

- **Data File:** Extracted MFCC data stored as data_10.npz file for efficiency.

- [`train.py`](classifier/train.py): Contains the entire process for loading, preparing, and evaluating the model. The trained model is saved to the flask folder.

#### Flask Folder

- [`Dockerfile`](flask/Dockerfile): Defines the Flask application’s Docker image.

- [`requirements.txt`](flask/requirements.txt): Lists Python dependencies.

- `model.keras`: The trained model is saved here.

- [`server.py`](flask/server.py): Implements the Flask API, handling requests and returning predictions.

- [`genre_prediction_service.py`](flask/genre_prediction_service.py): A singleton class for loading the model, preprocessing audio, and predicting the genre.

#### Nginx Folder

- [`Dockerfile`](nginx/Dockerfile): Builds the Nginx server.

- [`nginx.conf`](nginx/nginx.conf): Configures Nginx to act as a reverse proxy for the Flask API.

Nginx is a reverse proxy server that enhances security, scalability, performance, and user experience:

- Scalability: If you expand the app to multiple instances or need load balancing, Nginx can handle it.
- Performance: By offloading tasks like static file serving, caching, and SSL termination, Nginx improves the overall performance of your application.
- Simplicity for Clients: Users interact with the app through Nginx on port 80, eliminating the need to know internal port configurations.

#### Test Folder

- **Example Audio Files:** One 30-second audio clip for each genre to test the app.

**[`client.py`](client.py):**

- Specifies the server endpoint and the audio file to be sent for prediction.

**[`docker-compose.yaml`](docker-compose.yaml):**

- Orchestrates the Flask and Nginx services.


## Notebooks

For experiments, I use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

### **1. Exploratory Data Analysis (EDA)**

[`eda.ipynb`](notebooks/eda.ipynb)
- Download the dataset from Kaggle.

- Analyze class distribution and dataset balance.

- Visualize audio waveforms and spectrograms.

- Compare and analyse MFCC features across genres.



### **2. Data Preparation**

[`data_preparation.ipynb`](notebooks/data_preparation.ipynb)

- Segment each 30-second audio file into 3-second clips.

- Extract MFCC features and export data as NPZ file.

### **3. Model Selection**

[`model_NN_classification.ipynb`](notebooks/model_NN_classification.ipynb)
- Experiment with NN and CNN architectures.

- Perform hyperparameter tuning.

- Evaluate model performance on validation and test sets.

- Save the best-performing CNN model for deployment.


## Conclusion

This project demonstrates the application of a Convolutional Neural Network (CNN) for music genre classification using the GTZAN dataset. The relatively high accuracy achieved (> 75% on validation and test sets) highlights the strength of CNNs in extracting and leveraging audio features like MFCCs for genre prediction.

However, while the GTZAN dataset has been foundational in advancing music genre classification, it's essential to be aware of its limitations (limited diversity, quality issues, overuse and overfitting). Researchers and developers often use additional or alternative datasets to achieve more robust and generalizable results.

Future improvements could focus on:

- Data Augmentation: Enhancing the dataset with techniques like pitch shifting, time stretching, or adding noise to increase diversity.
- Advanced Architectures: Exploring models like spectrogram-based transformers or hybrid CNN-RNN architectures for improved feature extraction and temporal modeling.
- Dataset Quality: Leveraging larger, more diverse, and well-labeled datasets to enhance robustness.
- Feature Engineering: Experimenting with additional audio features beyond MFCCs, such as chroma features or spectral contrast.

This project provides a solid foundation for music genre classification while acknowledging opportunities for refinement and further exploration.

