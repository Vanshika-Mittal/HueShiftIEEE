# HueShift: Breathing Life into Every Frame 🖼️
Hueshift is a video colorization pipeline designed for frame-wise colorization of grayscale videos using a diffusion-based model, with a post-processing step to ensure temporal consistency. It operates on videos converted into individual frames and leverages keyframe detection and deflickering techniques to enhance both quality and coherence.

# Methodology
## Dataset Preparation
Source: [UCF101 action recognition dataset](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) <br>
Videos were converted into frames.
Keyframes were selected using a video-keyframe-detector to focus computation on representative content.

## Denoising Diffusion Probabalistic Model
Diffusion models are a class of generative models that work by iteratively adding noise to an input signal (like an image, text, or audio) and then learning to denoise from the noisy signal to generate new samples. Traditionally, these images are corrupted by Gaussian noise: the next image in the forward diffusion process is calculated by adding noise, sampled from a Gaussian distribution q, to the current image. 

In this approach, we used images in the LAB color space, a 3-channel alternative to the RGB color space. The “L” (Lightness) channel in this space is equivalent to a grayscale image: it represents the luminous intensity of each pixel. The two other channels are used to represent the color of each pixel. 
By using the LAB color space, we can corrupt only the color channels of an image and learn how to denoise the color channels using a grayscale image as conditioning. The model uses a traditional UNet to denoise the A and B channels of LAB images. 

The Model is essentially a UNet that takes a 3-channel LAB input image (the ground-truth greyscale channel with noised AB channels) along with the positionally embedded timestep and outputs a 2-channel prediction of the color noise.

## Post Processing
Due to low compute availability, we ran the model at 64x64 resolution; simply resizing the model outputs wouldn't cut it.
We used a Python script to convert the DDPM model’s low-resolution color outputs into high-resolution colorized frames. Specifically, the script loads the high-resolution grayscale (L channel) frame and the corresponding low-resolution colorized output from the model (which contains AB channels). It resizes the low-res color image to match the high-res grayscale frame using bicubic interpolation. Both images are then converted to the LAB color space, and the resized A and B channels are merged with the original high-res L channel to reconstruct a full-resolution LAB image. This LAB image is finally converted back to BGR and saved as a high-quality colorized frame. 

To address issues in temporal consistency during video colorization, we referred to the paper "Blind Video Deflickering by Neural Filtering with a Flawed Atlas", which proposes a zero-shot method to remove flickering artifacts without needing prior knowledge of the flicker type. Integrating insights from this approach and applying them on the high-resolution BGR images generated during post-processing helped us achieve more stable and visually coherent results across frames.

# Results
![drumming](https://github.com/user-attachments/assets/b7cb2bc2-8282-4708-96a1-daaac416444f)
![makeup](https://github.com/user-attachments/assets/efb25b6e-9a66-456a-8225-5a9d4e791aa1)

# Usage
1. **Clone the repository:**

    ```bash
    git clone https://github.com/Kazedaa/Hueshift-Video-Coloring
    cd Hueshift-Video-Coloring
    ```

2. **Create and activate a virtual environment:**

    ```bash
    conda create -n hueshift -f environment.yml
    conda activate hueshift
    ```

    Run `train_ddpm.ipynb` to train and `infer_ddpm.ipynb` to run inference.

3. **Increase resolution**
Use `utils/res_change.py:colorize_image(high_res_gray_path, low_res_color_path, output_path)`<br>
where <br>
`high_res_gray_path`: path to the frame at the original resolution<br>
`low_res_color_path`: path to the model output at low resolution<br>
`output_path`: output path<br>
4. [All in One Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker)
# Acknowledgments ❤️
 - [Key Frame Detector](https://github.com/joelibaceta/video-keyframe-detector) <br>
 - [All in One Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker)
