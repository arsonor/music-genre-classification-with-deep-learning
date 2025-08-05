## 🔁 **Prediction Service (API + Nginx + MLflow)**

The prediction service provides a **production-ready API** for music genre classification with enterprise-grade features.

### **🏗 Service Architecture**

```mermaid
sequenceDiagram
    participant Client
    participant Nginx
    participant Flask
    participant MLflow
    participant Monitoring
    
    Client->>Nginx: POST /predict (audio file)
    Nginx->>Flask: Forward request
    Flask->>MLflow: Load latest model
    MLflow-->>Flask: Return model
    Flask->>Flask: Extract MFCC features
    Flask->>Flask: Predict genre
    Flask->>Monitoring: Log prediction data
    Flask-->>Nginx: JSON response
    Nginx-->>Client: {"predicted_genre": "blues"}
```

### **🎵 API Endpoints**

#### **Basic Command Structure**

The `client.py` script is a command-line tool for testing the audio genre prediction API. Here's how to use it:

**Required Parameter:**

- `--file`: Path to an audio file or folder containing audio files

**Optional Parameters:**

- `--url`: API endpoint URL (default: http://127.0.0.1:80/predict)
- `--genre`: Actual genre label for comparison (optional, it's detected if explicit in the file name)

#### **Command Examples**
1. Test a Single Audio File

```bash
python client.py --file test/blues.00000.wav
```

2. Test Multiple Files in a Folder
```bash
python client.py --file audio_files_test/

# Or with Makefile:
make run-client
```

3. Complete Example with All Parameters
```bash
python client.py --url http://127.0.0.1:80/predict --file audio_files_test/blues.00000.wav --genre blues
```

#### **Sample Output**
```
[INFO] Auto-detected genre for 'blues.00000.wav': 'blues'
[INFO] Sending file: test/blues.00000.wav
  [RESULT] Predicted genre: blues

=== Summary ===
blues.00000.wav | Actual: blues | Predicted: blues
```

The script is designed to work with the music genre classification API and supports both single-file testing and batch processing of entire directories.

### **🧠 Model Loading Strategy**

**MLflow Integration**:
```python
# Singleton pattern with MLflow model registry
def Genre_Prediction_Service():
    if not _instance:
        # Load latest model version from MLflow registry
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        model = mlflow.keras.load_model(model_uri)
    return _instance
```

**Key Features**:
- **Automatic versioning**: Always loads latest registered model
- **Singleton pattern**: Model loaded once, reused for all requests
- **Error handling**: Graceful degradation on model loading failures
- **MLflow integration**: Seamless model registry connectivity

### **🔊 Audio Processing Pipeline**

**MFCC Feature Extraction**:
```python
def extract_mean_mfcc(audio_file_path):
    # Load audio at 22050 Hz sample rate
    signal, sample_rate = librosa.load(audio_file_path, sr=22050)
    
    # Extract 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, 
        n_mfcc=13, n_fft=2048, hop_length=512
    )
    
    # Return mean across time dimension
    return np.mean(mfcc, axis=1)  # Shape: (13,)
```

**Processing Steps**:
1. **Audio normalization** to 22050 Hz sample rate
2. **MFCC extraction** with 13 coefficients  
3. **Temporal averaging** for fixed-size feature vector
4. **Model prediction** with confidence scores
5. **Genre mapping** to human-readable labels

### **🌐 Nginx Reverse Proxy**

**Production Benefits**:
- **Load balancing**: Multiple Flask instances support
- **SSL termination**: HTTPS certificate management
- **Static file serving**: Efficient asset delivery
- **Request buffering**: Improved performance under load
- **Security headers**: Additional protection layer

**Configuration Highlights**:
```nginx
upstream flask_app {
    server api:5050;  # Internal Docker network
}

server {
    listen 80;
    location /predict {
        proxy_pass http://flask_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### **📊 Data Logging for Monitoring**

**Prediction Logging**:
- **MFCC features**: 13-dimensional feature vectors
- **Predictions**: Model outputs with confidence
- **Ground truth**: Optional actual genres for validation
- **Timestamps**: Request timing for drift analysis
- **Storage**: Parquet format for efficient analysis

**Monitoring Integration**:
```python
# Log prediction data to monitoring/data/current.parquet
df_row = pd.DataFrame([mfcc_vector], columns=[f"mfcc_{i+1}" for i in range(13)])
df_row["predicted_genre"] = predicted_genre
df_row["actual_genre"] = actual_genre  # Optional
df_combined.to_parquet(CURRENT_PARQUET_PATH, index=False)
```

---