# MNIST-Digit-Classification
Dual-framework (PyTorch + TensorFlow) 2-layer MLP achieving **97.6% accuracy** on handwritten digit recognition.

## Features
- PyTorch `nn.Module` + `DataLoader` (manual training loop)
- TensorFlow/Keras `Sequential` + `tf.data` (`model.fit()`)
- Adam optimizer, batch_size=64, 5 epochs
- End-to-end pipeline: data → model → train → evaluate → save

| Framework | Lines | Paradigm     | Training Time* | Accuracy |
|-----------|-------|--------------|----------------|----------|
| PyTorch   | 82    | Imperative   | ~45s (CPU)     | 97.5%    |
| TensorFlow| 52    | Declarative  | ~35s (CPU)     | 97.6%    |

*5 epochs, i7 CPU, no GPU [web:2]

## Quick Start
```bash
pip install -r requirements.txt
python mnist_pytorch.py    # PyTorch version
python mnist_tensorflow.py # TensorFlow version