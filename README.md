# ML4SCI Test 2f/2g — Jet Physics Regression using CNN

## 📌 Overview
This notebook builds a deep learning pipeline for predicting **jet mass** from high-energy physics data using **Convolutional Neural Networks (CNNs)**.

The dataset consists of **particle jet features stored in Parquet format**, requiring efficient large-scale data handling.

---

## 🎯 Objectives
- Predict jet mass from structured particle data
- Handle large-scale datasets efficiently
- Analyze feature importance using interpretability methods

---

## 🧠 Methodology

### 1. Dataset
- Source: CERN Open Data (Jet Physics)
- Format: Parquet files
- Structure:
  - `X_jet`: multi-channel feature representation
  - `m`: target jet mass

---

### 2. Data Processing Pipeline

#### Chunked Data Loading
- Uses `pyarrow` to read row groups
- Avoids loading full dataset into memory

#### Preprocessing
- Converts raw jet data into fixed-size tensors
- Handles variable-length particle features
- Applies transformations to standardize input

#### Caching System
- Preprocessed chunks saved to disk
- Enables faster training without repeated parsing

---

### 3. Dataset Handling
- Custom `Dataset` class loads cached chunks
- Data split:
  - Training set
  - Validation set

---

### 4. Model Architecture

#### CNN-Based Regression Model
- Input: multi-channel jet representation (e.g., 8 channels)
- Convolutional layers:
  - Extract spatial/feature patterns
- Fully connected layers:
  - Map features → jet mass prediction

---

### 5. Training Pipeline
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch training with DataLoader
- GPU acceleration supported

---

### 6. Evaluation Metrics
- Mean Absolute Error (MAE)
- Relative MAE
- Residual distribution
- Prediction vs Ground Truth plots

---

## 🔍 Interpretability

### 1. Channel Ablation
- Removes one input channel at a time
- Measures impact on prediction accuracy
- Identifies most important physics features

---

### 2. Gradient-Based Analysis
- Computes gradients w.r.t. input
- Highlights regions influencing predictions

---

## 📊 Visualization
- Loss curves
- Error histograms
- Residual plots
- Feature importance maps

---

## 📤 Model Export
- Model saved as `.pth`
- Exported to **ONNX format** for deployment

---

## ⚙️ Performance Considerations
- Efficient I/O via chunked reading
- Caching reduces redundant computation
- Scalable for large datasets

---

## 📦 Outputs
- Trained CNN model
- Cached dataset chunks
- Evaluation metrics and plots
- ONNX model

---

## 🚀 Key Insights
- CNNs effectively learn structured jet representations
- Data pipeline efficiency is critical for large datasets
- Certain input channels dominate prediction performance

---

## 🛠 Requirements
- Python 3.8+
- PyTorch
- pyarrow / fastparquet
- numpy
- matplotlib
- tqdm
- onnx

---

## 📌 Future Work
- Try Graph Neural Networks (GNNs)
- Incorporate attention mechanisms
- Use physics-informed constraints

---