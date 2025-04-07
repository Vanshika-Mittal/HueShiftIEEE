HueShift: Breathing Life into Every Frame

[Thumbnail photo- 64x64]

Very short summary (1-2 lines)
This project explores automatic video colorization using deep learning, comparing GAN-based and diffusion-based approaches to ensure smooth and realistic color transitions across frames.
Aim
To develop and compare deep learning models for grayscale video colorization using two independent approaches: saliency-map guided GANs and denoising diffusion probabilistic models, with an emphasis on achieving temporally coherent and visually realistic outputs.
Introduction
Colorization involves adding natural-looking color to grayscale images or videos, enhancing their visual quality and usability. In videos, maintaining temporal consistency is crucial to avoid flickering and ensure realism. With advancements in deep learning, particularly in generative models like GANs and diffusion models, automated video colorization has become increasingly effective and relevant across domains such as film restoration, media production, and AI-powered editing tools.
Literature survey and technologies used
Papers Referred to:
Diffusion Approach:
Denoising Diffusion Probabilistic Models 
Blind Video Deflickering by Neural Filtering with a Flawed Atlas 
GANs Approach:
Pyramid Feature Attention Network for Saliency detection 
SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network 
UnDIVE: Generalized Underwater Video Enhancement Using Generative Priors
Methodology ( includes research, theorems, procedure, etc.)
Diffusion Approach:
Diffusion models are a class of generative models that work by iteratively adding noise to an input signal (like an image, text, or audio) and then learning to denoise from the noisy signal to generate new samples. 

[image: forward pass and generative backward pass of a diffusion model]
Traditionally, these images are corrupted by gaussian noise: the next image in the forward diffusion process is calculated by adding noise, sampled from a gaussian distribution q, to the current image. 

In this approach, we used images in the LAB color space, a 3 channel alternative to the RGB color space. The “L” (Lightness) channel in this space is equivalent to a grayscale image: it represents the luminous intensity of each pixel. The two other channels are used to represent the color of each pixel. 
By using the LAB color space, we can corrupt only the color channels of an image, and learn how to denoise the color channels using a grayscale image as conditioning. The model uses a traditional UNet to denoise the A and B channels of LAB images. 



Our Model is essentially a UNet that takes a 3-channel LAB input image (the ground-truth greyscale channel with noised AB channels) along with the positionally embedded timestep and outputs a 2-channel prediction of the color noise. This model was trained on vast.ai GPUs using keyframes generated from videos in the UCF101 Action Recognition Dataset.



Dataset Preprocessing and Keyframe Generation
We generated keyframes of the UCF101 dataset using a python script that identified keyframes in a video by analyzing frame-to-frame differences in grayscale intensity after applying Gaussian blur. It used a threshold-based peak detection method to locate significant changes, then saved the corresponding original color frames.
Data Post-processing
We used a python script to convert the DDPM model’s low-resolution color outputs into high-resolution colorized frames. Specifically, the script loads the high-resolution grayscale (L channel) frame and the corresponding low-resolution colorized output from the model (which contains AB channels). It resizes the low-res color image to match the high-res grayscale frame using bicubic interpolation. Both images are then converted to the LAB color space, and the resized A and B channels are merged with the original high-res L channel to reconstruct a full-resolution LAB image. This LAB image is finally converted back to BGR and saved as the high-quality colorized frame. 
Fixing temporal consistency
To address issues in temporal consistency during video colorization, we referred to the paper "Blind Video Deflickering by Neural Filtering with a Flawed Atlas", which proposes a zero-shot method to remove flickering artifacts without needing prior knowledge of the flicker type. Integrating insights from this approach and applying them on the high-resolution BGR images generated during post-processing helped us achieve more stable and visually coherent results across frames.
GANs Approach:
In our second approach, we attempted Saliency Map-guided Colorization with Generative Adversarial Networks (GANs).
We adopted the generator architecture from the SCGAN framework as the foundational skeleton for our model. However, as an optimization, we deviated from the original design by reducing the number of channels as the network depth increased—contrary to SCGAN’s approach of increasing them—to improve efficiency and reduce overfitting.


[GENERATOR]
The model uses two discriminators based on the 70×70 PatchGAN architecture to enhance local detail and high-frequency accuracy. The first discriminator compares the generated colorized image with the ground truth, while the second—an attention-based discriminator—evaluates saliency-weighted versions of the images to better focus on important regions. This dual-discriminator setup improves realism and visual quality in the colorization results.

[DISCRIMINATORS]
We also experimented with WGAN loss, as suggested in the original paper, but observed that it did not significantly contribute to the model’s learning. As a result, we decided to exclude it from the final implementation.

[Final model architecture]
Optical Flow
Optical flow is a technique used in computer vision to estimate the motion of objects or pixels between two consecutive frames of a video. It assumes that the intensity of a point in the image remains constant as it moves from one frame to the next. By analyzing how pixels shift in position over time, optical flow generates a vector field that represents the direction and speed of movement in the scene.
We had planned to adopt the methodology from the UnDIVE paper to enhance temporal consistency in our video colorization pipeline. Specifically, we intended to use optical flow on the colorized outputs generated by our SCGAN-based model for keyframe and next-frame pairs identified during pre-processing. Using FastFlowNet, a real-time and efficient optical flow estimator, we aimed to compute the flow between the colorized keyframe (after warping) and the colorized next frame. This motion information would help align objects across frames more accurately. By applying a loss function between the warped keyframe and the next colorized frame, we sought to encourage smoother motion, consistent color transitions, and improved structural coherence throughout the video.
Results

Conclusions/Future Scope
Despite being constrained by limited computational resources, both our proposed approaches—saliency-map-guided GANs and denoising diffusion probabilistic models—demonstrated satisfactory temporal consistency and visual quality in video colorization. 
Looking ahead, with access to more powerful compute, we can scale our models to handle more diverse datasets and improve their generalizability. 
References/links to GitHub repo
Github Links:
Diffusion Approach: https://github.com/Kazedaa/Hueshift-Video-Coloring
GANs Approach: https://github.com/SreeDakshinya/HueShift-Video-Coloring/tree/main 
Flask Website: https://github.com/Vanshika-Mittal/HueShift-IEEE 
References:
Keyframe Detection: https://github.com/joelibaceta/video-keyframe-detector 
All-In-One Deflicker: https://github.com/ChenyangLEI/All-In-One-Deflicker
Mentors mentees details
Mentors
Aditya Ubaradka
Aishini Bhattacharjee
Hemang Jamadagni
Sree Dakshinya
Mentees
Akhil Sakhtieswaran
Swaraj Singh
Vanshika Mittal
