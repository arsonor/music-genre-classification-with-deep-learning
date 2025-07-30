import numpy as np
from classifier.pipeline.features import extract_features

def test_extract_features_shape():
    # Fake waveform data
    fake_waveform = np.random.randn(22050 * 5)
    features = extract_features(fake_waveform, sample_rate=22050)
    assert features.shape[0] > 0