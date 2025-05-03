import os
import numpy as np
import csv

DATASET_PATH = './dataset/data'
PROCESSED_DATA_PATH = './dataset/processed-data'
LABEL_FILE_PATH = './dataset/labels.csv'

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

LABEL_Y = np.memmap(
  os.path.join(PROCESSED_DATA_PATH, 'label_Y.dat'), 
  dtype='int32',
  mode='w+', 
  shape=(total,)
)

f = open(LABEL_FILE_PATH, 'r')
reader = csv.reader(f)

dict = {}

try:
  for index, row in enumerate(reader):
    if index == 0:
      continue
    dict[row[0]] = row[1]
  count = 0
  for label in os.listdir(DATASET_PATH):
    print("label: ", label)
    for entry in os.listdir(os.path.join(DATASET_PATH, label)):
      print("entry: ", entry)
      LABEL_Y[count] = map_y[dict[label+"/"+entry]]
      count += 1
finally:
  f.close()
  LABEL_Y.flush()