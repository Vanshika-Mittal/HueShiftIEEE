# HueShift: Breathing Life into Every Frame

## Project Overview
HueShift is an advanced video colorization project that transforms grayscale videos into vibrant color using two different deep learning approaches: Diffusion Models and Generative Adversarial Networks (GANs). Our goal was to develop and compare these two methods, with a special focus on ensuring temporal consistency and realistic coloration.

## Why Video Colorization?
Colorization enhances visual quality and usability of grayscale media, opening new possibilities for:
- Historical film restoration
- Artistic enhancement
- Media production and editing
- Educational content improvement

The key challenge in video colorization is maintaining temporal consistency - ensuring colors remain stable across frames to avoid flickering, while also producing naturally vibrant and realistic colors.

## Our Dual Approach

### 1. Diffusion-Based Approach
![Diffusion Model Process](images/diffusion_process.jpg)

Diffusion models are a class of generative models that work by iteratively adding and then removing noise. Our specific implementation uses:

#### LAB Color Space
- L channel: Luminance (grayscale information)
- A & B channels: Color information

Unlike traditional RGB colorization, we operate in LAB space where the L channel directly corresponds to the grayscale input, allowing us to focus on generating only the color channels (A and B).

#### Model Architecture
Our diffusion model uses a U-Net architecture with:
- Encoder: Extracts features from grayscale input through multiple downsampling blocks
- Bottleneck: Processes the most abstract features
- Decoder: Generates color information through upsampling blocks
- Skip connections: Preserve spatial detail between encoder and decoder

#### Temporal Consistency
To ensure consistent coloration across frames, we implemented a deflickering process using neural filtering with an atlas-based approach:
1. Atlas generation: Maps pixels from different frames to a shared 2D space
2. Neural filtering: Refines the frame-to-frame consistency while preserving details

### 2. GAN-Based Approach
![GAN Model Process](images/gan_process.jpg)

Our GAN implementation uses a saliency map-guided approach for more focused colorization of important regions.

#### Generator Architecture
- Encoder: Processes grayscale input through convolutional layers
- Feature Integration: Combines learned features with pre-trained VGG16 features
- Decoder: Generates color output and saliency maps

#### Dual Discriminator Setup
We employ two discriminators:
1. **Color Discriminator**: Evaluates the realism of generated colors
2. **Attention-based Discriminator**: Focuses on salient regions using generated saliency maps

#### Training Methodology
The model is trained with a combined loss function including:
- Adversarial loss: Improves realism of the generated colors
- L1 loss: Ensures color accuracy
- Saliency loss: Helps focus colorization on important areas

## Results

### Diffusion Model Outputs
![Diffusion Results](images/diffusion_results.jpg)

Our diffusion model showcases:
- Consistent coloration across frames
- High-fidelity detail preservation
- Natural color palette generation

**Processing Workflow:**
1. Low-resolution colorization using diffusion model
2. High-resolution upscaling while preserving original grayscale details
3. Temporal consistency enhancement using neural deflickering

### GAN Model Outputs
![GAN Results](images/gan_results.jpg)

The GAN-based approach demonstrates:
- Excellent performance on simple scenes
- Adaptive coloration guided by saliency
- Efficient processing pipeline

**Processing Workflow:**
1. Frame extraction from input video
2. Grayscale conversion and feature extraction
3. Colorization using the saliency-guided generator
4. Frame reassembly with temporal smoothing

## Comparison: Diffusion vs. GAN

### Strengths of Diffusion Models
- More natural color distribution
- Better handling of complex scenes
- Higher quality results with sufficient training

### Strengths of GAN Models
- Faster inference time
- More efficient training
- Better performance on limited compute resources

## Technical Implementation

### Dataset Processing
We worked with the UCF101 Action Recognition Dataset, using a custom keyframe extraction algorithm that:
1. Analyzes frame-to-frame differences in grayscale intensity
2. Applies Gaussian blur to reduce noise
3. Uses threshold-based peak detection to identify significant changes
4. Saves corresponding original color frames for training

### Model Training
Both models were trained on vast.ai GPUs with the following approaches:
- **Diffusion Model**: Trained using progressive denoising with conditioning on grayscale input
- **GAN Model**: Trained adversarially with dual discriminators and saliency guidance

### Post-Processing
For high-quality output, we implemented specialized post-processing:
1. **Diffusion Model**: 
   - Upscaling low-resolution outputs to match original video resolution
   - Merging high-res grayscale with generated color channels
   - Neural deflickering for temporal consistency

2. **GAN Model**:
   - Direct high-resolution output
   - Optional optical flow-based frame smoothing

## Future Directions
We see several promising avenues for future work:
1. Higher resolution model training
2. More efficient architectures for real-time processing
3. Integration with video editing software
4. Style-conditioned colorization for artistic control

## Team
**Mentors**:
- Aditya Ubaradka
- Aishini Bhattacharjee
- Hemang Jamadagni
- Sree Dakshinya

**Mentees**:
- Akhil Sakhtieswaran
- Swaraj Singh
- Vanshika Mittal

## References
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Blind Video Deflickering by Neural Filtering with a Flawed Atlas](https://arxiv.org/abs/2303.08120)
- [SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network](https://arxiv.org/abs/2011.05108)
- [Pyramid Feature Attention Network for Saliency detection](https://arxiv.org/abs/1903.00179)
- [UnDIVE: Generalized Underwater Video Enhancement Using Generative Priors](https://arxiv.org/abs/2311.12131)

## Code Repositories
- [Diffusion Approach](https://github.com/Kazedaa/Hueshift-Video-Coloring)
- [GANs Approach](https://github.com/SreeDakshinya/HueShift-Video-Coloring/tree/main)
- [Project Website](https://github.com/Vanshika-Mittal/HueShift-IEEE)
