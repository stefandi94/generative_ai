import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ANIMAL_FACES_DIR = os.path.join(DATA_DIR, "animal_faces")
ANIMAL_FACES_TRAIN_DIR = os.path.join(ANIMAL_FACES_DIR, "train")
ANIMAL_FACES_VAL_DIR = os.path.join(ANIMAL_FACES_DIR, "val")

ANIMAL_FACES_LABELS = {"cat": 0, "dog": 1, "wild": 2}