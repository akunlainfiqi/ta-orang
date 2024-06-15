import joblib
import numpy as np
import librosa

# Load the model, scaler, and label encoder
model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

def extract_mfcc(data, sample_rate, n_mfcc=20):
    n_fft = min(1024, len(data))
    hop_length = n_fft // 4
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T, axis=0)
    return mfcc

# Fungsi untuk mendapatkan fitur dari file audio
def get_features(path):
    try:
        # Memuat data audio
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

        if len(data) == 0:
            raise ValueError("Zero-size array, possibly an empty or very short file.")

        # Mengekstrak MFCC dari data asli
        mfcc_original = extract_mfcc(data, sample_rate)

        # Pastikan jumlah fitur yang dihasilkan sesuai dengan yang diharapkan
        features = mfcc_original.reshape(1, -1)

        # Normalisasi fitur
        features_scaled = scaler.transform(features)

        return features_scaled
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def predict_alphabet(audio_path):
    features = get_features(audio_path)
    if features is None:
        return "Error processing audio file"
    
    predictions = model.predict(features)
    most_common_prediction = np.bincount(predictions).argmax()
    predicted_phonic = label_encoder.inverse_transform([most_common_prediction])[0]
    
    return predicted_phonic

# Ensure this file is not run directly, only imported
if __name__ == '__main__':
    print("This script is intended to be imported, not run directly.")
