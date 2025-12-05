# Isolated Sign Language Recognition

A deep learning system for real-time American Sign Language (ASL) recognition using Transformer architecture and MediaPipe landmark detection.

## Overview

This project implements an end-to-end pipeline for recognizing isolated ASL signs from video input. The system extracts body, hand, and face landmarks using MediaPipe, processes them through a Transformer-based neural network, and outputs the predicted sign in real-time.

Key features:
- Recognition of 250 ASL signs
- Real-time inference on edge devices
- Docker-based deployment with MQTT messaging
- Transformer architecture with multi-head attention

## Project Structure

```
├── Edge Device/           # Edge deployment components
│   ├── Dockerize/         # Docker configuration and containerized app
│   ├── ML_Model.py        # Main inference script
│   ├── signer_*.py        # Sign capture utilities
│   └── v1_model.tflite    # TensorFlow Lite model
├── modeling/              # Model training and development
│   ├── config/            # Training configuration files
│   ├── layers/            # Custom Keras layers
│   ├── tf_models/         # Saved TensorFlow models
│   ├── tflite_models/     # Converted TFLite models
│   ├── utils/             # Training utilities
│   ├── train.py           # Training script with hyperparameter tuning
│   └── *.ipynb            # Training notebooks
├── model_final/           # Final model notebooks
├── legacy/                # Legacy training and data preparation
└── resources/             # Additional resources
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- MediaPipe
- OpenCV
- NumPy
- Pandas
- paho-mqtt (for edge deployment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd isolated-sign-language-recognition
```

2. Install dependencies:
```bash
pip install tensorflow opencv-python mediapipe numpy pandas paho-mqtt
```

## Usage

### Training

To train the model with hyperparameter tuning:

```bash
cd modeling
python train.py config/config.json
```

Training configuration can be modified in `modeling/config/config.json`:

```json
{
    "N_EPOCHS": 150,
    "TRAIN_BATCH_SIZE": 512,
    "LEARNING_RATE": 0.0001,
    "NUM_CLASSES": 250
}
```

### Edge Deployment

1. Build the Docker container:
```bash
cd "Edge Device/Dockerize"
./runDocker.sh
```

2. Or run directly:
```bash
docker build -t asl-recognition .
docker run -it asl-recognition
```

The edge deployment uses MQTT for communication between the camera capture module and the ML inference module.

### Inference

For standalone inference:

```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter("Edge Device/v1_model.tflite")
prediction_fn = interpreter.get_signature_runner("serving_default")

# Run inference
output = prediction_fn(inputs=preprocessed_data)
predicted_sign = output['outputs'].argmax()
```

## Model Architecture

The model uses a Transformer-based architecture:

- **Input Processing**: Landmarks from face (468 points), pose (33 points), and hands (21 points each) are extracted using MediaPipe
- **Embedding Layer**: Separate dense layers for each landmark type, combined into a unified embedding
- **Transformer Encoder**: Multiple attention blocks with multi-head self-attention
- **Classification Head**: Dense layer with softmax activation for 250 sign classes

Model parameters:
- Transformer blocks: 2-6 (configurable)
- Attention heads: 8
- Embedding dimension: 384-512
- Dropout rate: 0.1-0.2

## Data Processing

The input data consists of sequences of landmark coordinates:
- 543 total landmarks per frame:
  - Face: 468 landmarks
  - Pose: 33 landmarks
  - Left hand: 21 landmarks
  - Right hand: 21 landmarks
- x, y coordinates for each landmark
- Variable-length sequences padded/truncated to fixed input size

## License

This project is available under the MIT License.

## Acknowledgments

- Based on the Google Isolated Sign Language Recognition challenge
- Uses MediaPipe for landmark detection
- Built with TensorFlow and Keras
