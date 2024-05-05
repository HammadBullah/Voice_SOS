import os
import numpy as np
from pydub import AudioSegment
from scipy.signal import spectrogram
import pyaudio
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import messagebox 


def preprocess_audio(audio_data):
    _, _, Sxx = spectrogram(audio_data, fs=44100)
    resized_spectrogram = resize_spectrogram(Sxx)
    spectrogram_gray = resized_spectrogram[..., np.newaxis]
    min_val = np.min(spectrogram_gray)
    max_val = np.max(spectrogram_gray)
    spectrogram_normalized = (spectrogram_gray - min_val) / (max_val - min_val)
    return spectrogram_normalized


def audio_callback(in_data, frame_count, time_info, status):
    global is_wake_word_detected, predictions
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    input_spectrogram = preprocess_audio(audio_data)
    prediction = model.predict(input_spectrogram[np.newaxis, ...])
    predictions.append(prediction[0, 0]) 
    activation_threshold = 0.00012
    if prediction[0, 0] > activation_threshold:
        is_wake_word_detected = True
    return (in_data, pyaudio.paContinue)


def on_wake_word_detected():
    print("Wake word detected! Activating function...")
    messagebox.showinfo("Help Detected", "Help word detected. Please be safe until help reaches you!")


def detect_wake_word():
    global is_wake_word_detected, predictions 
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024,
                    stream_callback=audio_callback)
    stream.start_stream() 
    while stream.is_active():
        if is_wake_word_detected:
            print("Help word detected!")
            on_wake_word_detected()
            break 
    stream.stop_stream()
    stream.close()
    p.terminate()


def resize_spectrogram(spectrogram, target_shape=(129, 129)):
    height_diff = target_shape[0] - spectrogram.shape[0]
    width_diff = target_shape[1] - spectrogram.shape[1]
    padded_spectrogram = np.pad(spectrogram, ((0, max(0, height_diff)), (0, max(0, width_diff))),
                                mode='constant', constant_values=0)
    cropped_spectrogram = padded_spectrogram[:target_shape[0], :target_shape[1]]
    return cropped_spectrogram

is_wake_word_detected = False
predictions = []

model = tf.keras.models.load_model(r'E:\MInor\Model\model_test1.keras')

detect_wake_word()


