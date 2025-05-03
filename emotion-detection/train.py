import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

total = 31002
input = 478*3

X_PATH = './dataset/processed-data/X.dat'
Y_PATH = './dataset/processed-data/label_Y.dat'
MODEL_PATH = './models/label_model.keras'

X = np.memmap(X_PATH, dtype='float32', mode='r', shape=(total, input))
Y = np.memmap(Y_PATH, dtype='int32', mode='r', shape=(total,))

model = keras.Sequential([
    keras.layers.Input(shape=(input,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(8, activation='softmax')  # For 8 emotion classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callback to save best model (entire model, not just weights)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    save_best_only=True,
    save_weights_only=False,  # Save full model
    monitor="val_loss",
    mode="min",               # Save the model with the lowest val_loss
    verbose=1
)

history = model.fit(
    x=X,
    y=Y,
    batch_size=32,
    validation_split=0.2,  # Use 20% of the data for validation
    shuffle=True,          # Shuffle the data before splitting
    verbose=1,
    callbacks=[checkpoint_callback],  # Add the checkpoint callback here
    epochs=20
)

# Assuming you have the `history` object from model.fit(...)
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
