import cv2
import numpy as np

def lab_to_rgb(lab_image):
    # OpenCV expects the range [0, 255] for color images
    lab = (lab_image * 255).astype(np.uint8)  # Scale to [0, 255]
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # Convert LAB to BGR
    return rgb