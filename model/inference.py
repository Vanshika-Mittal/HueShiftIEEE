import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from pathlib import Path
import os
import yaml
from model.unet import UNet
import logging
import traceback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
NUM_TIMESTEPS = 1000

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LinearNoiseSchedule:
    def __init__(self, T):
        super().__init__()
        beta_start = 1e-4
        beta_end = 0.02

        self.beta = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # for sampling process
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.one_by_sqrt_one_minus_alpha_hat = 1.0 / torch.sqrt(1 - self.alpha_hat)
        self.sqrt_beta = torch.sqrt(self.beta)

    def backward(self, xt, noise_pred, t):
        l, ct = torch.split(xt, [1, 2], dim=1)

        one_by_sqrt_alpha = self.one_by_sqrt_alpha.to(xt.device)[t]
        beta = self.beta.to(xt.device)[t]
        one_by_sqrt_one_minus_alpha_hat = self.one_by_sqrt_one_minus_alpha_hat.to(
            xt.device
        )[t]
        sqrt_beta = self.sqrt_beta.to(xt.device)[t]

        for _ in range(len(xt.shape) - 1):
            one_by_sqrt_alpha = one_by_sqrt_alpha.unsqueeze(-1)
            one_by_sqrt_one_minus_alpha_hat = one_by_sqrt_one_minus_alpha_hat.unsqueeze(
                -1
            )
            beta = beta.unsqueeze(-1)

        mean = one_by_sqrt_alpha * (
            ct - beta * one_by_sqrt_one_minus_alpha_hat * noise_pred
        )
        std_dev = sqrt_beta

        if t == 0:
            out = torch.cat([l, mean], dim=1)
        else:
            z = torch.randn_like(ct).to(xt.device)
            out = torch.cat([l, mean + std_dev * z], dim=1)
        return out


def lab_to_rgb(lab_image):
    lab = (lab_image * 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


def preprocess_image(image_path):
    """Preprocess the input image with logging"""
    logger.info(f"Loading image from {image_path}")
    try:
        image = Image.open(image_path)
        logger.info(f"Original image size: {image.size}, mode: {image.mode}")

        # Convert to grayscale if not already
        if image.mode != "L":
            image = image.convert("L")
            logger.info("Converted image to grayscale")

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        logger.info(
            f"Normalized array shape: {img_array.shape}, range: [{img_array.min()}, {img_array.max()}]"
        )

        # Add batch and channel dimensions
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        logger.info(f"Input tensor shape: {img_tensor.shape}")

        return img_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def sample(L, model, scheduler):
    """Generate colorized image from L channel"""
    num_samples = L.shape[0]
    out_shape = (num_samples, 2, IMG_SIZE, IMG_SIZE)
    xt = torch.cat([L, torch.randn(out_shape).to(DEVICE)], dim=1).to(DEVICE)

    for t in reversed(range(NUM_TIMESTEPS)):
        timestep = torch.ones(num_samples, dtype=torch.long, device=DEVICE) * t
        noise_pred = model(xt, timestep)
        xt = scheduler.backward(xt, noise_pred, t)

    # Convert output to RGB
    save_output = xt.cpu()
    save_output = save_output.permute(0, 2, 3, 1).numpy()
    save_output = np.array([lab_to_rgb(img) for img in save_output])
    save_output = torch.tensor(save_output).permute(0, 3, 1, 2).float() / 255.0

    return save_output


def load_model(model_path, config_path):
    """Load the model with detailed logging"""
    logger.info("Loading model configuration")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config: {config}")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

    logger.info("Initializing model")
    try:
        model = UNet(
            input_channels=config["model"]["input_channels"],
            output_channels=config["model"]["output_channels"],
            hidden_channels=config["model"]["hidden_channels"],
            num_blocks=config["model"]["num_blocks"],
        )
        logger.info(f"Model architecture: {model}")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

    logger.info(f"Loading model weights from {model_path}")
    try:
        # Add weights_only=True to address the FutureWarning
        model.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model weights: {str(e)}")
        raise

    model.eval()
    return model


def postprocess_image(output_tensor, output_path):
    """Postprocess the output tensor with logging"""
    logger.info("Postprocessing output tensor")
    try:
        # Convert to numpy and scale to [0, 255]
        output_array = output_tensor.squeeze().cpu().numpy()
        logger.info(
            f"Output array shape: {output_array.shape}, range: [{output_array.min()}, {output_array.max()}]"
        )

        output_array = (output_array * 255).clip(0, 255).astype(np.uint8)

        # Save the image
        output_image = Image.fromarray(output_array)
        logger.info(f"Saving output image to {output_path}")
        output_image.save(output_path)
        logger.info("Output image saved successfully")

        return output_path
    except Exception as e:
        logger.error(f"Error postprocessing image: {str(e)}")
        raise


def run_inference(image_path, model_path, output_path, config_path=None):
    """Run inference with comprehensive logging"""
    logger.info(f"Starting inference process for {image_path}")
    try:
        # Load model
        if config_path is None:
            config_path = model_path.replace(".pth", ".yaml")
        logger.info(f"Using config path: {config_path}")

        # Verify paths exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at {image_path}")

        model = load_model(model_path, config_path)

        # Preprocess
        input_tensor = preprocess_image(image_path)

        # Run inference
        logger.info("Running model inference")
        with torch.no_grad():
            output_tensor = model(input_tensor)
        logger.info(f"Model output shape: {output_tensor.shape}")

        # Postprocess and save
        output_path = postprocess_image(output_tensor, output_path)
        logger.info("Inference completed successfully")

        return output_path
    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # Example usage
    image_path = "input.jpg"
    model_path = "model/ddpm.pth"
    output_path = "output.jpg"
    run_inference(image_path, model_path, output_path)
