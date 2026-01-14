# VHD Audio Classification

This project implements an audio classification system for Valvular Heart Disease (VHD) detection using Deep Learning (RNN-LSTM).

## Dataset Preprocessing (`preprocessing.ipynb`)

The preprocessing script handles the reorganization of the dataset into training and testing sets.

- **Source Directory**: `dataset/train`
- **Target Directory**: `dataset/test`
- **Split Ratio**: 10% of the training data is moved to the test set for validation.
- **Classes Processed**:
  - AS (Aortic Stenosis)
  - MR (Mitral Regurgitation)
  - MS (Mitral Stenosis)
  - MVP (Mitral Valve Prolapse)
  - N (Normal)

## Class Mapping

The model uses the following mapping for prediction outputs:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | AS | Aortic Stenosis |
| 1 | MR | Mitral Regurgitation |
| 2 | MS | Mitral Stenosis |
| 3 | MVP | Mitral Valve Prolapse |
| 4 | N | Normal |

## Model Architecture (`vhd_model.ipynb`)

The model utilizes Mel-Frequency Cepstral Coefficients (MFCC) features extracted from audio files to train a Recurrent Neural Network (RNN).

### Feature Extraction

- **Library**: Librosa
- **Features**: MFCC (13 coefficients)
- **Input Shape**: (100, 13) - padded/truncated to 100 time steps.

### Neural Network Structure

- **Type**: Sequential LSTM
- **Layers**:
  1. LSTM (128 units)
  2. Dropout (0.3)
  3. Dense (64 units, ReLU)
  4. Dropout (0.3)
  5. Output Dense (5 units, Softmax)

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32

## Model Conversion

The trained model is saved as `model.h5` and converted to TensorFlow.js format for web deployment using `tensorflowjs_converter`.

**Note**: Ensure compatible versions of `numpy` and `tensorflowjs` are installed to avoid `AttributeError: module 'numpy' has no attribute 'bool'` during conversion.

## Technologies Used

- **Core**: Python 3.x
- **ML Framework**: TensorFlow / Keras
- **Audio Processing**: Librosa
- **Web Deployment**: TensorFlow.js
