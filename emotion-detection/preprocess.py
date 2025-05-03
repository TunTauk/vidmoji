import os
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DATASET_PATH = './dataset/data'
PROCESSED_DATA_PATH = './dataset/processed-data'
FACE_LANDMARKER_MODEL_PATH = './../models/face_landmarker.task'

total = 31002
input = 478*3

map_y = {
  'anger': 0,
  'contempt': 1,
  'disgust': 2,
  'fear': 3,
  'happy': 4,
  'neutral': 5,
  'sad': 6,
  'surprise': 7
}

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)

X = np.memmap(
  os.path.join(PROCESSED_DATA_PATH, 'X.dat'), 
  dtype='float32', 
  mode='w+', 
  shape=(total, input)
)

Y = np.memmap(
  os.path.join(PROCESSED_DATA_PATH, '8_class_Y.dat'), 
  dtype='int32',
  mode='w+', 
  shape=(total,)
)

landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

try:
  count = 0
  for label in os.listdir(DATASET_PATH):
    print("label: ", label)
    for entry in os.listdir(os.path.join(DATASET_PATH, label)):
      print("entry: ", entry)
      
      Y[count] = map_y[label]
      img_path = os.path.join(DATASET_PATH, label, entry)
      image = mp.Image.create_from_file(img_path)
      result = landmarker.detect(image)
      if len(result.face_landmarks) == 0:
        continue
      face_landmarks = result.face_landmarks[0]
      data = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks]).flatten()
      X[count] = data
      count += 1

finally:
  landmarker.close()
  X.flush()
  Y.flush()
    
