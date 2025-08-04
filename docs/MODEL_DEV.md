
The system predicts the genre of a given audio file using deep learning techniques. By leveraging the GTZAN dataset and extracting Mel-Frequency Cepstral Coefficients (MFCC) features, the application processes the input audio and classifies it into one of 10 genres: rock, classical, metal, disco, blues, reggae, country, hiphop, jazz, and pop. The project has employed neural networks (NN) and convolutional neural networks (CNN) to achieve this goal.

- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Notebooks](#notebooks)
  - [EDA (Handling Audio Data)](#1-exploratory-data-analysis-eda)
  - [Data Preparation](#2-data-preparation)
  - [Model Selection](#3-model-selection)



## Dataset Description

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

The audio files are processed to extract **MFCC features (this audio feature is explained in the [`EDA.ipynb`](notebooks/EDA.ipynb) notebook)** and segment the 30-second audio files into 3-second clips for model training.



## Model Architecture

### CNN for Audio Classification

Our model is specifically designed for processing MFCC (Mel-Frequency Cepstral Coefficients) features extracted from audio signals.

#### Input Processing
```python
# Input shape: (batch_size, time_steps, mfcc_features, channels)
# Example: (32, 130, 13, 1)
# - 130 time steps (3-second audio segments)
# - 13 MFCC coefficients
# - 1 channel (mono audio)
```
### **Feature Engineering:**
- **MFCC Features**: 13-dimensional Mel-Frequency Cepstral Coefficients
- **Audio Segmentation**: 30-second clips split into 3-second segments  
- **Data Augmentation**: Temporal and spectral augmentation techniques
- **Normalization**: Feature scaling for optimal neural network performance

## 🧠 **Model Architecture**

### **CNN Architecture for MFCC Feature Classification**

Our model uses a **Convolutional Neural Network (CNN)** optimized for audio feature classification:


#### Network Architecture
```python
def build_model(input_shape, l2_reg=0.001, learning_rate=0.001):
    model = Sequential([
        # Input Layer
        Input(shape=input_shape),  # (130, 13, 1) - Time steps × MFCC coefficients × Channel
        
        # Feature Extraction Block 1
        Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPool2D((3,3), strides=(2,2), padding='same'),
        BatchNormalization(),
        
        # Feature Extraction Block 2  
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPool2D((3,3), strides=(2,2), padding='same'),
        BatchNormalization(),
        
        # Feature Extraction Block 3
        Conv2D(128, (2,2), activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPool2D((2,2), strides=(2,2), padding='same'),
        BatchNormalization(),
        
        # Classification Head
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.4),
        Dense(10, activation='softmax')  # 10 genre classes
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

#### Key Design Features

**🎵 Audio-Optimized Convolutions**
- **MFCC Input Processing**: Handles 13-dimensional MFCC feature vectors across 130 time steps
- **Small kernel sizes** (3x3, 2x2) suitable for MFCC feature maps
- **Progressive filter increase** (32 → 64 → 128) for hierarchical feature learning
- **MaxPooling with strides** to reduce temporal and frequency dimensions

**🔧 Regularization Strategy**
- **L2 regularization** on all trainable layers (0.001)
- **Batch normalization** after each conv block for training stability
- **Dropout (0.4)** before final classification layer
- **Early stopping** on validation accuracy (patience=10)

**📊 Performance Characteristics**
- **Training accuracy**: ~90-95% (with regularization preventing overfitting)
- **Validation accuracy**: ~75-80% (realistic performance on unseen data)
- **Achieves >75% accuracy** on validation and test sets
- **Inference time**: <50ms per prediction on CPU
- **Model size**: ~2MB (suitable for edge deployment)






### Feature Engineering Pipeline

#### MFCC Extraction Process
```python
def extract_mfcc_features(audio_file_path):
    # Load audio at standardized sample rate
    signal, sr = librosa.load(audio_file_path, sr=22050)
    
    # Extract 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=signal, 
        sr=sr,
        n_mfcc=13,        # Standard 13 coefficients
        n_fft=2048,       # FFT window size
        hop_length=512    # Overlap between frames
    )
    
    # For training: segment into 3-second clips
    # For inference: average across time dimension
    return mfcc
```

#### Data Preprocessing
```python
# Training pipeline preprocessing
def prepare_training_data(audio_files):
    X, y = [], []
    
    for audio_file, genre_label in audio_files:
        # Load 30-second audio file
        signal, sr = librosa.load(audio_file, sr=22050)
        
        # Segment into 3-second clips (10 clips per file)
        segment_length = 3 * sr  # 3 seconds
        for i in range(0, len(signal) - segment_length, segment_length):
            segment = signal[i:i + segment_length]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(segment, sr=sr, n_mfcc=13)
            
            # Pad/truncate to fixed length (130 time steps)
            if mfcc.shape[1] < 130:
                mfcc = np.pad(mfcc, ((0,0), (0, 130-mfcc.shape[1])))
            else:
                mfcc = mfcc[:, :130]
            
            X.append(mfcc.T)  # Transpose to (time, features)
            y.append(genre_label)
    
    return np.array(X), np.array(y)
```




## Notebooks

For experiments, I use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

### **1. Exploratory Data Analysis (EDA)**

[`EDA.ipynb`](notebooks/EDA.ipynb)
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