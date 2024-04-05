import os

current_script_path = os.path.abspath(__file__)
SRC = os.path.dirname(os.path.dirname(current_script_path))
DATA_DIR = os.path.join(SRC, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "processed/synthetic")
