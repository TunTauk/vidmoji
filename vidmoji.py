import cv2
import time
import random
import os
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult as TGestureRecognizerResult

# Config
DETECTION_TIME = 0.3  # seconds to wait before showing emoji
NUM_HANDS = 2  # number of hands to detect
EMOJI_DURATION = 2.0  # seconds to show
EMOJI_SIZE = 50 # size of emoji
EMOJI_PATH = "./emojis/" # path to emojis folder

#Init mediapipe
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Load emojis
gesture_to_emoji = {}
for filename in os.listdir(EMOJI_PATH):
    img_path = os.path.join(EMOJI_PATH, filename)
    img = Image.open(img_path).convert("RGBA").resize((EMOJI_SIZE, EMOJI_SIZE))
    gesture_to_emoji.update({filename.split(".")[0]: img})

# Global variables
active_emojis = []
last_proccess_time = 0

# Function to overlay image
def overlay_image(bg_frame, overlay_pil, pos, alpha=1.0):
    """Overlay RGBA image on OpenCV frame at given position."""
    bg_pil = Image.fromarray(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Adjust emoji alpha
    overlay = overlay_pil.copy()
    alpha_mask = overlay.split()[3].point(lambda p: int(p * alpha))
    overlay.putalpha(alpha_mask)

    # Overlay emoji on temp transparent canvas
    temp = Image.new("RGBA", bg_pil.size)
    temp.paste(overlay, pos, overlay)

    # Composite and convert back to OpenCV format
    combined = Image.alpha_composite(bg_pil, temp)
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGBA2BGR)

# Function to handle gesture recognition results
def result_cb(result: TGestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global active_emojis
    current_time = time.time()
    half_width = int(output_image.width / 2)
    for i in range(len(result.gestures)):
        if len(result.gestures[i]) == 0 or len(result.handedness[i]) == 0:
            continue
        gesture = result.gestures[i][0].category_name
        handedness = result.handedness[i][0].category_name
        # we do invert because we flip the image
        handedness = "Left" if handedness == "Right" else "Right"
        key = f"{handedness}+{gesture}".lower()
        if key not in gesture_to_emoji:
            continue
        print(f"Detected {gesture} with {handedness}")
        from_x = 0 if handedness == "Left" else half_width
        to_x = half_width if handedness == "Left" else int(output_image.width)
        active_emojis.append({
            "start_time": current_time,
            "emoji": gesture_to_emoji.get(key),
            "x": random.randint(from_x, to_x - EMOJI_SIZE), 
        })

# Webcam
cap = cv2.VideoCapture(0) 

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./models/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_cb,
    num_hands=NUM_HANDS,
)

with GestureRecognizer.create_from_options(options) as recognizer:
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
            recognizer.recognize_async(mp_image, timestamp)
        
        new_active_emojis = []
        for emoji in active_emojis:
            elapsed = time.time() - emoji["start_time"]
            if elapsed < EMOJI_DURATION:
                progress = elapsed / EMOJI_DURATION
                y = int(end_y - 80 - progress * 200)
                alpha = 1.0 if progress < 0.5 else 1.0 - ((progress - 0.5) * 2)
                frame = overlay_image(frame, emoji["emoji"], (emoji["x"], y), alpha)
                new_active_emojis.append(emoji)
        active_emojis = new_active_emojis
        
        cv2.imshow("Vidmoji", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
