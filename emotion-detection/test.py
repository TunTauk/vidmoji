import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

total = 31002
input = 478*3

X_PATH = './dataset/processed-data/X.dat'
Y_PATH = './dataset/processed-data/label_Y.dat'
MODEL_PATH = './models/label_model.keras'

X = np.memmap(X_PATH, dtype='float32', mode='r', shape=(total, input))
Y = np.memmap(Y_PATH, dtype='int32', mode='r', shape=(total,))

model = keras.models.load_model(MODEL_PATH)

label_map = {
    0: 'anger',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprise'
}

y_predict = model.predict(X)
y_predict = np.argmax(y_predict, axis=1)

cm = confusion_matrix(Y, y_predict, labels=list(label_map.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

