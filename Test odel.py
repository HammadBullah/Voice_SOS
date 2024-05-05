import os
import numpy as np
from pydub import AudioSegment
from scipy.signal import spectrogram
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to preprocess audio and convert to spectrogram
def preprocess_audio(audio_data):
    _, _, Sxx = spectrogram(audio_data, fs=44100)  # Adjust fs based on your audio input
    resized_spectrogram = resize_spectrogram(Sxx)
    spectrogram_gray = resized_spectrogram[..., np.newaxis]
    min_val = np.min(spectrogram_gray)
    max_val = np.max(spectrogram_gray)
    spectrogram_normalized = (spectrogram_gray - min_val) / (max_val - min_val)
    return spectrogram_normalized

# Function to resize spectrograms to a common shape
def resize_spectrogram(spectrogram, target_shape=(129, 129)):
    height_diff = target_shape[0] - spectrogram.shape[0]
    width_diff = target_shape[1] - spectrogram.shape[1]
    padded_spectrogram = np.pad(spectrogram, ((0, max(0, height_diff)), (0, max(0, width_diff))),
                                mode='constant', constant_values=0)
    cropped_spectrogram = padded_spectrogram[:target_shape[0], :target_shape[1]]
    return cropped_spectrogram

# Function to detect wake word in audio file
def detect_wake_word(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio_data = np.array(audio.get_array_of_samples())
    input_spectrogram = preprocess_audio(audio_data)
    prediction = model.predict(input_spectrogram[np.newaxis, ...])
    return prediction[0, 0]

# Load the trained model
model = tf.keras.models.load_model(r'E:\MInor\Model\model_test.keras')

# Directory containing audio files used for training
audio_dir = r'E:\MInor\Model\data'

# List to store results
results = []

# Iterate over audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        audio_file = os.path.join(audio_dir, filename)
        prediction = detect_wake_word(audio_file)
        results.append((filename, prediction))


test_spectrograms = []
y_test = []

# Iterate over test audio files
for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        # Load audio file
        audio = AudioSegment.from_file(os.path.join(audio_dir, filename))
        audio_data = np.array(audio.get_array_of_samples())
        
        # Preprocess audio and convert to spectrogram
        spectrogram = preprocess_audio(audio_data)
        test_spectrograms.append(spectrogram)
        
        # Assuming the label is 1 if the filename contains the word 'positive', otherwise 0
        label = 1 if 'positive' in filename else 0
        y_test.append(label)

# Convert lists to numpy arrays
X_test_gray = np.array(test_spectrograms)
y_test = np.array(y_test)

X_test_gray = np.expand_dims(X_test_gray, axis=-1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_gray, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Get predicted probabilities for the test set
y_pred_prob = model.predict(X_test_gray)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


