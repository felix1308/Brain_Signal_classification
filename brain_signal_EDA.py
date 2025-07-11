import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


brain_data = pd.read_csv('synthetic_brain_signal_multiclass.csv')
brain_data['label'] = brain_data['label'].astype('category')

classes = ['apple', 'banana', 'bottle', 'baseline']


# Basic data exploration using Pandas
print("Data info: ")
print(brain_data.info())

print("\n Missing values: ")
print(brain_data.isnull().sum())

print("\n Data shape: ")
print(brain_data.shape)

# Visualizing the distribution of classes
plt.figure(figsize=(10, 6))
sns.kdeplot(data=brain_data[brain_data['label'] == 'apple'], x='ch5_t150')
plt.title('Distribution of Brain Signal Classes')
plt.xlabel('Samples')
plt.ylabel('Count')
plt.show()

#time domain visualization
features = brain_data.drop(columns=['label'],axis=1)
labels = brain_data['label']

mean_signals = {}
for label in brain_data['label'].unique():
    mean_vector = features[labels==label].mean().values
    mean_signals[label] = mean_vector.reshape(16,512)
# --- Plot the average signal for a representative channel for each class ---
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Average Signal by Class on a Key Channel', fontsize=16)

# Channel selection based on where signatures were added
channel_map = {'apple': 5, 'banana': 7, 'bottle': 8, 'baseline': 4}
time = np.arange(512) / 256.0 # Create time axis

for i, label in enumerate(classes):
    channel_to_plot = channel_map[label] - 1 # Adjust for 0-based index
    axs[i].plot(time, mean_signals[label][channel_to_plot, :], label=f'Class: {label}')
    axs[i].set_title(f"Class: '{label}' on Channel {channel_map[label]}")
    axs[i].axvline(0.5, color='r', linestyle='--', label='Signature Start')
    axs[i].set_ylabel('Amplitude (µV)')
    axs[i].grid(True)

plt.xlabel('Time (s)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#frequency domain analysis
from scipy import signal

# --- Plot the PSD for the average signals ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
fig.suptitle('Power Spectral Density (PSD) by Class', fontsize=16)

for ax, label in zip(axs.flatten(), classes):
    channel_to_plot = channel_map[label] - 1
    # Calculate PSD
    freqs, psd = signal.welch(mean_signals[label][channel_to_plot, :], fs=256)
    ax.semilogy(freqs, psd) # Use a log scale for power
    ax.set_title(f"Class: '{label}' on Channel {channel_map[label]}")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.grid(True)
    ax.set_xlim(0, 50) # Focus on frequencies up to 50 Hz

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --- Prepare a subset of the data ---
n_samples_for_tsne = 1000
data_subset = brain_data.sample(n=n_samples_for_tsne, random_state=42)

# Separate features and labels for the subset
X_subset = data_subset.drop('label', axis=1)
y_subset = data_subset['label']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# --- Apply t-SNE ---
print("\nRunning t-SNE... (this may take a moment)")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(X_scaled)
print("t-SNE complete.")

# --- Plot the results ---
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    hue=y_subset,
    palette=sns.color_palette("hsv", n_colors=4),
    legend="full",
    alpha=0.8
)
plt.title('t-SNE Visualization of Brain Signal Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Classes')
plt.grid(True)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load and Prepare Data ---

# Load the dataset
df = pd.read_csv('synthetic_brain_signal_multiclass.csv')

# Select one single example from the 'apple' class
example_row = df[df['label'] == 'apple'].iloc[0]

# Separate the signal data (features) from the label
signal_features = example_row.drop('label').values

# Reshape the flattened data back to (channels, time_samples)
# The data has 16 channels and 512 time samples per channel
signal_2d = signal_features.reshape(16, 512)

# We'll visualize Channel 5, which is one of the channels with the 'apple' signature
channel_data = signal_2d[4, :] # Index 4 corresponds to Channel 5

# Define parameters
sampling_rate = 256  # Hz
time = np.arange(channel_data.size) / sampling_rate

# --- Generate Spectrogram ---

plt.figure(figsize=(10, 6))

# Create the spectrogram
Pxx, freqs, bins, im = plt.specgram(
    channel_data,
    NFFT=128,          # Number of data points used in each block for the FFT
    Fs=sampling_rate,  # Sampling frequency
    noverlap=64,       # Number of points of overlap between blocks
    cmap='viridis'     # Color map
)

plt.title("Spectrogram of a Single 'Apple' Class Trial (Channel 5)", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Frequency (Hz)", fontsize=12)
plt.colorbar(im, label='Intensity (dB)') # Add a color bar to show intensity
plt.ylim(0, 60) # Focus on frequencies up to 60 Hz
plt.show()