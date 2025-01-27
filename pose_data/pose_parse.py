import numpy as np
import pickle
with open('processed_quart_val.pkl', 'rb') as f:
    pose_data = pickle.load(f)
    daw = 0