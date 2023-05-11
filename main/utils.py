import os
import librosa
from matplotlib import pyplot as plt
import tqdm
import uuid

from preprocess import get_melspectrogram_db, transform_audio


def display_audio(filename, sr=48000):
  """display spectogram of a audio file in notebook"""
  s = get_melspectrogram_db(filename)
  plt.figure()
  plt.imshow(s)
  plt.show()

def generate_duplicate_wav(filepath, label, sr=48000, augmented=True, num_of_duplicates=300):
  """generate duplicates of input audio with the option to randomly augment/transform the audio"""
  for i in tqdm(range(num_of_duplicates), f"Generating wav duplicates of wavfile({filepath})..."):
    if augmented:
      audio = transform_audio(filepath)
    else:
      audio, _ = librosa.load(filepath, sr=sr)
    
    new_filename = f"{label}_{uuid.uuid4().hex}"
    new_filepath = os.path.join('data', label, new_filename)
    wavfile.write(new_filepath, sr, audio)