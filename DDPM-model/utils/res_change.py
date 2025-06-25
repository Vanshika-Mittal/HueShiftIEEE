import cv2
import numpy as np
import os

def colorize_image(high_res_gray_path, low_res_color_path, output_path):
    # Load images
    high_res_gray = cv2.imread(high_res_gray_path, cv2.IMREAD_GRAYSCALE)
    low_res_color = cv2.imread(low_res_color_path)

    # Resize low-res color to match high-res grayscale
    low_res_resized = cv2.resize(low_res_color, (high_res_gray.shape[1], high_res_gray.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Convert images to LAB color space
    low_res_lab = cv2.cvtColor(low_res_resized, cv2.COLOR_BGR2LAB)
    high_res_lab = cv2.merge([high_res_gray, low_res_lab[..., 1], low_res_lab[..., 2]])

    # Convert back to BGR
    colorized_image = cv2.cvtColor(high_res_lab, cv2.COLOR_LAB2BGR)

    # Save output
    cv2.imwrite(output_path, colorized_image)
    print(f"Colorized image saved as {output_path}")
