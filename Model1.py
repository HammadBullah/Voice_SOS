import os
import numpy as np
from pydub import AudioSegment
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

HELP_DIR = r'E:\MInor\Model\data\positives'
NON_HELP_DIR = r'E:\MInor\Model\data\egatives'
NOISE_DIR = r'E:\MInor\Model\data\cckgrounds'

def load_and_convert_to_spectrograms(data_dir, label):
    spectrograms = []
    labels = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        audio = AudioSegment.from_file(file_path)
        audio_data = np.array(audio.get_array_of_samples())
        _, _, Sxx = spectrogram(audio_data, fs=audio.frame_rate)
        Sxx_resized = resize_spectrogram(Sxx)
        spectrograms.append(Sxx_resized)
        labels.append(label)
    return spectrograms, labels

def resize_spectrogram(spectrogram, target_shape=(129, 129)):
    height_diff = target_shape[0] - spectrogram.shape[0]
    width_diff = target_shape[1] - spectrogram.shape[1]
    
    padded_spectrogram = np.pad(spectrogram, ((0, max(0, height_diff)), (0, max(0, width_diff))),
                                mode='constant', constant_values=0)
    
    cropped_spectrogram = padded_spectrogram[:target_shape[0], :target_shape[1]]
    
    return cropped_spectrogram


help_spectrograms, help_labels = load_and_convert_to_spectrograms(HELP_DIR, 0) 
non_help_spectrograms, non_help_labels = load_and_convert_to_spectrograms(NON_HELP_DIR, 1)
noise_spectrograms, noise_labels = load_and_convert_to_spectrograms(NOISE_DIR, 2) 

all_spectrograms = np.concatenate([help_spectrograms, non_help_spectrograms, noise_spectrograms], axis=0)
all_labels = np.concatenate([help_labels, non_help_labels, noise_labels], axis=0)

min_val = np.min(all_spectrograms)
max_val = np.max(all_spectrograms)
all_spectrograms = (all_spectrograms - min_val) / (max_val - min_val)

X_train, X_test, y_train, y_test = train_test_split(all_spectrograms, all_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_gray = X_train[..., np.newaxis]
X_val_gray = X_val[..., np.newaxis]
X_test_gray = X_test[..., np.newaxis]

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') 
    ])
    return model

input_shape = (129, 129, 1)
model = create_model(input_shape)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_gray, y_train, epochs=60, validation_data=(X_val_gray, y_val))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

help_class_idx = 0
help_class_mask = y_val == help_class_idx
X_val_help = X_val_gray[help_class_mask]
y_val_help = y_val[help_class_mask]
help_loss, help_accuracy = model.evaluate(X_val_help, y_val_help)
print(f'Validation Accuracy for "help" class: {help_accuracy}')

test_loss, test_accuracy = model.evaluate(X_test_gray, y_test)
print(f'Test Accuracy: {test_accuracy}')

model.save(r'E:\MInor\Model\model_test2.keras')

