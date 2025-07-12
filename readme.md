# Brain Signal Classification: A Deep Learning Project

A simulation and classification project that explores the process of identifying distinct "thought patterns" from synthetic 16-channel EEG brain signal data using machine learning and deep learning techniques.

## 📖 Table of Contents

- [Motivation](#motivation)
- [Important Disclaimer](#important-disclaimer)
- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling & Results](#modeling--results)
- [Future Work](#future-work)
- [License](#license)

## 🎯 Motivation

The field of Brain-Computer Interfaces (BCIs) is one of the most exciting frontiers in technology. While true "thought-reading" remains science fiction, the ability to classify distinct mental states from EEG signals is a real and active area of research.

This project was created to:

- Simulate a realistic machine learning workflow for handling complex, time-series signal data
- Demonstrate key data science skills, including data generation, feature engineering, exploratory data analysis, and model building
- Explore the application of Deep Learning, particularly Convolutional Neural Networks (CNNs), for pattern recognition in noisy signal data

## ⚠️ Important Disclaimer

This project uses **synthetically generated data**. The "thought signatures" for classes like 'apple' or 'banana' are engineered patterns designed to be learnable by a model. This project does **not** represent actual thought decoding or mind-reading. Its purpose is to serve as a challenging and educational exercise in signal processing and machine learning.

## ✨ Project Features

The core of this project is a custom-generated dataset that mimics the challenges of real-world EEG data:

- **Multi-Class Classification**: The goal is to classify signals into one of five distinct categories: apple, banana, bottle, baseline (resting state), and noise (artifacts)
- **16-Channel Data**: Simulates data from a 16-electrode EEG cap
- **Engineered Signatures**: Each class has a unique, subtle signature embedded in the frequency domain
- **Class Imbalance**: The dataset is intentionally imbalanced to mimic real-world scenarios where some events are rarer than others
- **Noise Artifacts**: A dedicated noise class simulates junk signals, forcing the model to be robust

## 📂 Project Structure

```
brain-signal-classification/
│
├── data/
│   └── synthetic_brain_signal_imbalanced.csv    # The generated dataset
│
├── notebooks/
│   ├── 01_Data_Generation.ipynb                 # Script to generate the dataset
│   └── 02_EDA.ipynb                            # Notebook for Exploratory Data Analysis
│
├── src/
│   ├── train.py                                # Script to train the classification model
│   └── model.py                                # Model architecture (e.g., CNN)
│
├── README.md                                   # You are here!
└── requirements.txt                            # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip and virtualenv

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/brain-signal-classification.git
   cd brain-signal-classification
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Generate the data (Optional):** To generate a new version of the dataset, run the data generation script/notebook.

2. **Explore the data:** Open and run the `02_EDA.ipynb` notebook to see the full exploratory data analysis.

3. **Train the model:**
   ```bash
   python src/train.py
   ```

## 📊 Exploratory Data Analysis (EDA)

Before modeling, a thorough EDA was conducted to understand the dataset's structure and verify the engineered signatures.

### 1. Time-Domain Visualization
The average signal for each class shows the engineered patterns clearly. The 'apple' and 'banana' classes have distinct bursts at 0.5s, while 'bottle' shows a sustained amplitude shift.

### 2. Frequency-Domain Analysis (PSD)
Power Spectral Density plots confirm the unique frequency peaks for each class. 'Apple' has a sharp peak at 40 Hz, and 'banana' has one at 25 Hz.

### 3. Dimensionality Reduction (t-SNE)
A t-SNE plot visualizes the high-dimensional data in 2D. The clear separation between clusters confirms that the classes are distinct and a machine learning model should be able to classify them effectively.

## 🧠 Modeling & Results
### (Future Development)

*(This section would be filled in after building the model)*

A 1D Convolutional Neural Network (CNN) was chosen for this task due to its effectiveness in finding local patterns in sequence data.

### Model Architecture:
- **Input Layer**: (None, 8192, 1)
- **Conv1D Layer** (Filters: 32, Kernel: 3) -> ReLU -> MaxPool
- **Conv1D Layer** (Filters: 64, Kernel: 3) -> ReLU -> MaxPool
- **Flatten Layer**
- **Dense Layer** (128 units, ReLU)
- **Output Layer** (5 units, Softmax)

### Performance:
- The model achieved an overall accuracy of **XX.X%** on the test set
- **Confusion Matrix**: *Coming soon*
- The confusion matrix shows that the model performs well across all classes, with minor confusion between the baseline and bottle classes

## 🔮 Future Work

This project provides a strong foundation that can be extended in several exciting ways:

- **Use a Real-World Dataset**: Adapt the pipeline to a public EEG dataset, such as a motor imagery dataset from PhysioNet, to classify real-world brain states
- **Build an Interactive Demo**: Create a simple web application using Streamlit or Gradio to visualize random trials and see the model's live predictions
- **Experiment with Architectures**: Implement and compare other models, such as Recurrent Neural Networks (RNNs) or Transformers, to see if they can achieve better performance

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note**: Remember to replace `YOUR_USERNAME` in the clone URL with your actual GitHub username, and update the performance metrics once you have actual results from your model training.
