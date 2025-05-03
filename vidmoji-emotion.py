import cv2
import time
import random
import os
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult as TFaceLandmarkerResult
from tensorflow import keras

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Config
DETECTION_TIME = 0.3  # seconds to wait before showing emoji
MODEL_PATH = "./models/emotion_detection_model.keras"

model = keras.models.load_model(MODEL_PATH)

last_proccess_time = 0

# Function to handle gesture recognition results
def result_cb(result: TFaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print("result_cb")
    print(len(result.face_landmarks))
    if len(result.face_landmarks) == 0:
      return
    face_landmarks = result.face_landmarks[0]
    data = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks]).flatten()
    print("data: ", data.shape)
    y_predict = model.predict(data.reshape(1, -1))
    print("y_predict: ", y_predict)
    y_predict = np.argmax(y_predict, axis=1)
    print("y_predict: ", y_predict)

# Webcam
cap = cv2.VideoCapture(0) 

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./models/face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_cb
)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
        current_time = time.time()
        end_y = int(frame.shape[0])
        frame = cv2.flip(frame, 1)
        
        if current_time - last_proccess_time > DETECTION_TIME:
            last_proccess_time = current_time
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, timestamp)
        
        cv2.imshow("Vidmoji", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
