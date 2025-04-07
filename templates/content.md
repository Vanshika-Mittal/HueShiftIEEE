# Hue Shift: Bringing Color to Black and White Media

## Project Overview
Hue Shift is an innovative deep learning project that transforms black and white media into vibrant, colored content using a Denoising Diffusion Probabilistic Model (DDPM) approach. Our implementation focuses on maintaining color consistency and natural appearance across both images and videos.

## The DDPM Implementation
Our DDPM implementation works by learning to reverse a carefully calibrated noise-addition process. The model learns to predict and remove noise from the color channels while preserving the original grayscale information.

### Technical Architecture
Our implementation uses a UNet backbone with the following specifications:
- Channel progression: [256, 384, 512, 768] for downsampling path
- Multi-head self-attention (16 heads) at multiple scales
- Sinusoidal time embeddings (dimension 512) for noise step conditioning
- Group normalization (32 channels) for training stability
- Skip connections between corresponding layers

### Color Space Implementation
We operate in the LAB color space, separating our input into:
- L channel: Represents the input grayscale image (luminance)
- A and B channels: Target color information to be predicted

### Noise Schedule
We implement a linear noise schedule with:
- 1000 diffusion timesteps
- Beta values ranging from 1E-4 to 0.02
- Separate forward and backward processes for training and inference

## Colorization Process

### Training Pipeline
Our training process begins with image preparation in the LAB color space. Each input image is resized to 16x16 pixels and split into its L channel (luminance) and AB channels (color information). The L channel serves as our conditioning input, while the AB channels are our target for color prediction.

During the forward diffusion process, we keep the L channel unchanged while progressively adding noise to the A and B channels. This noise addition follows a carefully calibrated linear schedule with beta values ranging from 1E-4 to 0.02 across 1000 timesteps. At each step, we track the noise level using sinusoidal time embeddings of dimension 512.

The model then learns through reverse diffusion. At each timestep, it predicts the noise that was added to the AB channels, using the L channel as a guide. We train using Mean Squared Error (MSE) loss between the predicted and actual noise. This approach allows the model to learn the relationship between structural information in the L channel and the corresponding color information in the AB channels.

### Inference Pipeline
1. **Input Preparation**:
   - Extract L channel from grayscale input
   - Initialize random noise for A and B channels

2. **Denoising Process**:
   - Gradually denoise A and B channels over 1000 steps
   - Each step uses model's noise prediction
   - L channel guides the colorization process

3. **Output Generation**:
   - Combine denoised A and B channels with original L channel
   - Convert from LAB back to RGB color space

## Video Processing
Our video implementation processes frames sequentially through the same pipeline:
## Results Gallery
[Note: Add your result images here with captions]
1. G2C Forward Diffusion Process
2. G2C Reverse Diffusion Results
3. C2G Model Outputs
4. Video Colorization Examples
5. Saliency Maps and Color Propagation
6. Before/After Comparisons
### Frame Processing
1. **Input Handling**:
   - Load video frames individually
   - Convert each frame to LAB color space
   - Extract L channel as conditioning input

### Color Consistency
Color consistency in videos is achieved through:
1. **Architectural Features**:
   - Stable noise schedule across frames
   - Consistent processing pipeline for each frame
   - Fixed random seed for noise initialization

2. **Channel Management**:
   - L channel preservation ensures structural consistency
   - Independent A and B channel prediction maintains color stability

### Output Generation
- Sequential frame processing in original order
- Direct LAB to RGB conversion for each frame
- No additional post-processing required

## Technical Specifications
- Model Parameters: ~400MB
- Training Configuration:
  - Batch Size: 2
  - Image Size: 16x16
  - Training Epochs: 50
  - Optimizer: Adam (learning rate: 1E-5)
- Hardware Requirements: NVIDIA T4/A100 GPUs

[Gallery section to be updated with actual results]


