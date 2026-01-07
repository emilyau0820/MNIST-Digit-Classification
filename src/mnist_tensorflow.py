# Project: MNIST Digit Classification
# File Name: mnist_tensorflow.py
# Author: Emily Au

# Dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow: {tf.__version__}")

# 1. Load MNIST dataset (auto-downloads ~10MB)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess (normalize + flatten for Dense layers)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(60000, 28*28)  # Flatten to (60000, 784)
x_test = x_test.reshape(10000, 28*28)

# 3. Create tf.data datasets (for compile/fit)
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)

# 4. Build model
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(10)  # Logits (no softmax needed)
])

# 5. Compile + fit
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset, 
    epochs=5, 
    validation_data=val_dataset,
    verbose=1  # Progress bars
)

# 6. Results
test_loss, test_acc = model.evaluate(val_dataset)
test_acc *= 100
print(f"Final Test Accuracy: {test_acc:.2f}%")