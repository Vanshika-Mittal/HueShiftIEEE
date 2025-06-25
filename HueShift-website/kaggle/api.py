from flask import Flask, request, jsonify, send_file
from flask_ngrok import run_with_ngrok
import os
from pathlib import Path
import sys
import uuid
from PIL import Image
import torch
import yaml
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add model directory to path
model_dir = "/kaggle/working/model"
sys.path.append(model_dir)
logger.info(f"Added {model_dir} to sys.path")
from inference import run_inference

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Configure paths
UPLOAD_FOLDER = "/kaggle/working/storage/temp"
RESULTS_FOLDER = "/kaggle/working/storage/processed/images"
MODEL_PATH = os.path.join(model_dir, "ddpm.pth")
CONFIG_PATH = os.path.join(model_dir, "ddpm.yaml")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

logger.info(f"Initialized with paths:")
logger.info(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"RESULTS_FOLDER: {RESULTS_FOLDER}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"CONFIG_PATH: {CONFIG_PATH}")

# Verify model files exist
logger.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")
logger.info(f"Config path exists: {os.path.exists(CONFIG_PATH)}")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    logger.info("Health check requested")
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": os.path.exists(MODEL_PATH),
            "config_loaded": os.path.exists(CONFIG_PATH),
        }
    )


@app.route("/api/process", methods=["POST"])
def process_image():
    """Handle image processing requests"""
    logger.info("Processing request received")

    if "image" not in request.files:
        logger.error("No image in request.files")
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        logger.error("Empty filename received")
        return jsonify({"error": "No selected file"}), 400

    # Generate unique filename
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.jpg")
    output_path = os.path.join(RESULTS_FOLDER, f"{unique_id}_output.jpg")

    logger.info(f"Generated paths - Input: {input_path}, Output: {output_path}")

    # Save uploaded file
    try:
        file.save(input_path)
        logger.info(f"File saved successfully to {input_path}")

        # Verify image was saved and can be opened
        with Image.open(input_path) as img:
            logger.info(f"Image details - Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        logger.error(f"Error saving or verifying image: {str(e)}")
        return jsonify({"error": f"Error saving image: {str(e)}"}), 500

    try:
        logger.info("Starting inference process")
        logger.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")
        logger.info(f"Config path exists: {os.path.exists(CONFIG_PATH)}")
        logger.info(f"Input path exists: {os.path.exists(input_path)}")

        # Run inference with explicit config path
        output_path = run_inference(
            input_path, MODEL_PATH, output_path, config_path=CONFIG_PATH
        )
        logger.info(f"Inference completed successfully, output saved to {output_path}")

        # Verify output was created
        if os.path.exists(output_path):
            with Image.open(output_path) as img:
                logger.info(
                    f"Output image details - Size: {img.size}, Mode: {img.mode}"
                )
        else:
            logger.warning(f"Output file not found at {output_path}")

        # Return result path and ID
        response = {"success": True, "output_path": output_path, "id": unique_id}
        logger.info(f"Returning success response: {response}")
        return jsonify(response)
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"Cleaned up input file: {input_path}")
        except Exception as e:
            logger.error(f"Error cleaning up input file: {str(e)}")


@app.route("/api/results/<image_id>", methods=["GET"])
def get_result(image_id):
    """Serve processed images"""
    logger.info(f"Result requested for image_id: {image_id}")

    output_path = os.path.join(RESULTS_FOLDER, f"{image_id}_output.jpg")
    logger.info(f"Looking for result at: {output_path}")

    if not os.path.exists(output_path):
        logger.error(f"Result not found at {output_path}")
        return jsonify({"error": "Image not found"}), 404

    try:
        # Verify image can be opened before sending
        with Image.open(output_path) as img:
            logger.info(f"Result image details - Size: {img.size}, Mode: {img.mode}")
        logger.info("Sending result file")
        return send_file(output_path, mimetype="image/jpeg")
    except Exception as e:
        error_msg = f"Error sending result: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run()
