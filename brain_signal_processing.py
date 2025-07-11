import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (same as before) ---
n_channels = 16
sampling_rate = 256
duration = 2
n_samples = int(sampling_rate * duration)
time = np.arange(n_samples) / sampling_rate
classes = ['apple', 'banana', 'bottle', 'baseline']

# --- Function to generate a single trial ---
def generate_trial(label):
    trial_data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        alpha_wave = np.random.uniform(0.5, 1.5) * np.sin(2 * np.pi * np.random.uniform(8, 12) * time)
        beta_wave = np.random.uniform(0.2, 0.8) * np.sin(2 * np.pi * np.random.uniform(13, 25) * time)
        noise = np.random.normal(0, 0.5, n_samples)
        trial_data[ch, :] = alpha_wave + beta_wave + noise

    start_sample = int(0.5 * sampling_rate)

    if label == 'apple':
        burst_time = np.arange(int(0.2 * sampling_rate)) / sampling_rate
        burst_signal = 3 * np.sin(2 * np.pi * 40 * burst_time)
        for target_ch in [2, 5, 10]:
             trial_data[target_ch, start_sample:start_sample + len(burst_signal)] += burst_signal
    elif label == 'banana':
        burst_time = np.arange(int(0.3 * sampling_rate)) / sampling_rate
        burst_signal = 2.5 * np.sin(2 * np.pi * 25 * burst_time)
        for target_ch in [3, 7, 12]:
            trial_data[target_ch, start_sample:start_sample + len(burst_signal)] += burst_signal
    elif label == 'bottle':
        amp_window = np.zeros(n_samples)
        amp_window[start_sample:start_sample+int(0.5*sampling_rate)] = 1.5
        for target_ch in [1, 8, 14]:
            trial_data[target_ch, :] += trial_data[target_ch, :] * amp_window
    
    return trial_data

# --- Generate one trial for each class ---
apple_trial = generate_trial('apple')
banana_trial = generate_trial('banana')
bottle_trial = generate_trial('bottle')
baseline_trial = generate_trial('baseline')

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Spectrograms of Simulated Brain Signals by Class', fontsize=16)

# Plot for Apple (showing Channel 5)
axs[0, 0].specgram(apple_trial[4, :], Fs=sampling_rate, NFFT=128, noverlap=64, cmap='viridis')
axs[0, 0].set_title("Class: 'Apple' (Channel 5)")
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Frequency (Hz)')

# Plot for Banana (showing Channel 7)
axs[0, 1].specgram(banana_trial[6, :], Fs=sampling_rate, NFFT=128, noverlap=64, cmap='viridis')
axs[0, 1].set_title("Class: 'Banana' (Channel 7)")
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Frequency (Hz)')

# Plot for Bottle (showing Channel 8)
axs[1, 0].specgram(bottle_trial[7, :], Fs=sampling_rate, NFFT=128, noverlap=64, cmap='viridis')
axs[1, 0].set_title("Class: 'Bottle' (Channel 8)")
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Frequency (Hz)')

# Plot for Baseline (showing Channel 4)
axs[1, 1].specgram(baseline_trial[3, :], Fs=sampling_rate, NFFT=128, noverlap=64, cmap='viridis')
axs[1, 1].set_title("Class: 'Baseline' (Channel 4)")
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Frequency (Hz)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
