import os
import uuid

# viz
from matplotlib import pyplot as plt
from tqdm import tqdm

# data
import numpy as np

# audio
import librosa
from audiomentations import (
    AddGaussianNoise, 
    AddGaussianSNR, 
    BandPassFilter, 
    Compose, 
    ClippingDistortion, 
    PolarityInversion, 
)

def get_melspectrogram_db(file_path, sr=48000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80, sec=5):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<sec*sr:
        wav=np.pad(wav,int(np.ceil((sec*sr-wav.shape[0])/2)),mode='reflect')
    
    wav=wav[:sec*sr]
    
    spec=librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft,
        hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax
    )
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    
    spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.float32)
    return spec_scaled

def get_spec(wav, sr=48000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80, sec=5):
    wav=wav[:sec*sr]
    
    spec=librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft,
        hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax
    )
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    
    return spec_db

def transform_audio(file_name, sr=48000):
    audio, _ = librosa.load(file_name, sr=sr)
    
    transform = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.5),
        AddGaussianSNR(p=0.5),
        ClippingDistortion(p=0.5),
        PolarityInversion(p=0.25),
        BandPassFilter(p=0.25)
    ])
    taudio = transform(audio, sample_rate=sr)
    
    return taudio

