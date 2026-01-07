# MNIST-Digit-Classification
Dual-framework (PyTorch + TensorFlow) 2-layer MLP achieving **97.6% accuracy** on handwritten digit recognition.

## Features
- PyTorch `nn.Module` + `DataLoader` (manual training loop)
- TensorFlow/Keras `Sequential` + `tf.data` (`model.fit()`)
- Adam optimizer, batch_size=64, 5 epochs
- End-to-end pipeline: data → model → train → evaluate → save

## Quick Start
```bash
pip install -r requirements.txt
python pytorch_demo.py    # PyTorch version
python tensorflow_demo.py # TensorFlow version
