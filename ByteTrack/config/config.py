import os
from pathlib import Path

# Project root (one level above this 'config' directory)
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = str(ROOT_DIR)
# print(f"ROOT_DIR: {ROOT_DIR}")

# --- MODEL ---
MODEL_PERSON_PATH = os.path.join(BASE_DIR, "models", "best_17Sep.pt")
MODEL_PERSON_CLASS_TARGET = [0]
MODEL_PERSON_CONFIDENCE_THRESHOLD = 0.8
MODEL_FACE_PATH = os.path.join(BASE_DIR, "models", "yolo12n-face.pt")
MODEL_OBJECT_PATH = os.path.join(BASE_DIR, "models", "object.pt")
MODEL_OBJECT_CONFIDENCE_THRESHOLD = 0.7


# --- DATA ---
VIDEO_NAME = "many"
SOURCE_VIDEO_PATH = f"./videos/{VIDEO_NAME}.avi"
TARGET_VIDEO_PATH = f"./results/videos/{VIDEO_NAME}.avi"
TARGET_METADATA_PATH = f"./results/metadata/{VIDEO_NAME}.txt"

# --- IMAGE SIZE ---
IMG_WIDTH_SIZE = 640
IMG_HEIGHT_SIZE = 640
MAX_FRAMES = 200  # Set to None to process the entire video

# BYTES_TRACK
TRACK_THRESHOLD = 0.8
TRACK_BUFFER = 200
MATCH_THRESH = 0.8
ASPECT_RATIO_THRESHOLD = 3.0
MIN_BOX_AREA = 1.0
MOT20 = False

PROGRESS = False  # Show progress bar
