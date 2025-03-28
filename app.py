import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
    send_file,
)
from werkzeug.utils import secure_filename
import uuid
import requests
from pathlib import Path
import tempfile
import shutil
import json
from PIL import Image
import numpy as np

# Config
UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = "results"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "wmv"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
# This should be a much smaller value, kept it at 16MB for now

# Kaggle notebook endpoint (ngrok)
KAGGLE_ENDPOINT = os.getenv("KAGGLE_ENDPOINT")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "dev-secret-key"  # This is not needed in development env

UPLOAD_FOLDER.mkdir(exist_ok=True)
os.makedirs(os.path.join(app.root_path, RESULT_FOLDER), exist_ok=True)


def allowed_image_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def allowed_video_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def generate_unique_filename(filename):
    """Generate a unique filename while preserving the original extension."""
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return f"{uuid.uuid4().hex}.{ext}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    print("=== Starting upload_file function ===")

    if "image" not in request.files:
        print("No image in request.files")
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        print("Empty filename")
        return jsonify({"error": "No selected file"}), 400

    # Generate unique ID and save locally
    unique_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}_{filename}")
    print(f"Saving file locally to: {local_path}")
    file.save(local_path)

    try:
        # Send to Kaggle notebook API
        print(f"Opening file for sending to Kaggle API: {local_path}")
        with open(local_path, "rb") as f:
            files = {"image": (f"{unique_id}_input.jpg", f, "image/jpeg")}
            print(f"Sending POST request to: {KAGGLE_ENDPOINT}/api/process")
            print(f"Files dictionary: {files}")

            response = requests.post(f"{KAGGLE_ENDPOINT}/api/process", files=files)
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response content: {response.content}")

        if response.status_code != 200:
            print(f"Processing failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            return jsonify(
                {"error": "Processing failed", "details": response.text}
            ), 500

        try:
            result = response.json()
            print(f"Parsed JSON response: {result}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response content: {response.content}")
            return jsonify({"error": "Invalid JSON response from server"}), 500

        if not result.get("success"):
            print(f"Processing failed: {result}")
            return jsonify(
                {
                    "error": "Processing failed",
                    "details": result.get("error", "Unknown error"),
                }
            ), 500

        result_id = result.get("id")
        if not result_id:
            print("No result ID in response")
            return jsonify({"error": "No result ID returned"}), 500

        print(f"Got result ID: {result_id}")

        # Download result from Kaggle
        print(f"Requesting result from: {KAGGLE_ENDPOINT}/api/results/{result_id}")
        result_response = requests.get(f"{KAGGLE_ENDPOINT}/api/results/{result_id}")
        print(f"Result response status: {result_response.status_code}")

        if result_response.status_code == 200:
            result_path = os.path.join(
                app.config["RESULT_FOLDER"], f"{unique_id}_result.jpg"
            )
            print(f"Saving result to: {result_path}")
            with open(result_path, "wb") as f:
                f.write(result_response.content)

            response_data = {
                "success": True,
                "result_id": unique_id,
                "original_path": f"/uploads/{unique_id}_{filename}",
                "result_path": f"/results/{unique_id}_result.jpg",
            }
            print(f"Returning success response: {response_data}")
            return jsonify(response_data)
        else:
            print(f"Failed to get result: {result_response.text}")
            return jsonify({"error": "Failed to get result"}), 500

    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    finally:
        print("=== Ending upload_file function ===")


@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_video_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = generate_unique_filename(filename)
            file_path = UPLOAD_FOLDER / unique_filename
            file.save(file_path)

            # Here we will process the video with colorization model
            # For now, it just return success message
            flash(
                "Video uploaded successfully. Processing will be implemented in future phases."
            )

            return redirect(url_for("upload_video"))
        else:
            flash(
                "Invalid file format. Please upload a video file (mp4, avi, mov, wmv)."
            )
            return redirect(request.url)

    return render_template("upload_video.html")


@app.route("/gallery")
def gallery():
    return render_template("gallery.html")


@app.route("/status/<process_id>")
def check_status(process_id):
    try:
        with open(os.path.join(UPLOAD_FOLDER, f"{process_id}.json"), "r") as f:
            process_info = json.load(f)
        return jsonify(process_info)
    except FileNotFoundError:
        return jsonify({"status": "processing", "message": "Still processing"})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route("/colorize/image", methods=["POST"])
def colorize_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_image_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Generate unique process ID and filenames
        process_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        base_name, ext = os.path.splitext(filename)

        original_path = os.path.join(UPLOAD_FOLDER, f"{process_id}_original{ext}")
        file.save(original_path)

        with Image.open(original_path) as img:
            if img.mode != "L":  # If not already grayscale
                gray = img.convert("L")
                gray.save(original_path)

        with open(original_path, "rb") as f:
            files = {"image": (f"{process_id}_input.jpg", f, "image/jpeg")}
            response = requests.post(f"{KAGGLE_ENDPOINT}/api/process", files=files)

            if response.status_code != 200:
                print(f"Processing failed: {response.text}")  # Debug log
                raise Exception(f"Failed to process image: {response.text}")

            result = response.json()
            if not result.get("success"):
                raise Exception(
                    f"Processing failed: {result.get('error', 'Unknown error')}"
                )

            result_id = result.get("id")
            if not result_id:
                raise Exception("No result ID returned from processing")

            result_response = requests.get(f"{KAGGLE_ENDPOINT}/api/results/{result_id}")
            if result_response.status_code != 200:
                raise Exception("Failed to download processed image")

            colorized_path = os.path.join(UPLOAD_FOLDER, f"{process_id}_colorized{ext}")
            with open(colorized_path, "wb") as f:
                f.write(result_response.content)

        # Store process info
        process_info = {
            "status": "completed",
            "result": {
                "original_url": f"/uploads/{process_id}_original{ext}",
                "colorized_url": f"/processed/{process_id}_colorized{ext}",
            },
        }

        with open(os.path.join(UPLOAD_FOLDER, f"{process_id}.json"), "w") as f:
            json.dump(process_info, f)

        return jsonify(
            {
                "process_id": process_id,
                "message": "Image processing completed",
                "original_url": f"/uploads/{process_id}_original{ext}",
                "colorized_url": f"/processed/{process_id}_colorized{ext}",
            }
        )

    except Exception as e:
        print(f"Error in colorize_image: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500


@app.route("/gallery")
def get_gallery():
    try:
        response = requests.get(f"{KAGGLE_ENDPOINT}/api/gallery", params=request.args)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/result/image/<image_id>")
def get_result(image_id):
    try:
        response = requests.get(f"{KAGGLE_ENDPOINT}/api/results/{image_id}")
        if response.status_code != 200:
            return jsonify(response.json()), response.status_code

        # Save the image temporarily
        temp_path = UPLOAD_FOLDER / f"{image_id}.png"
        with open(temp_path, "wb") as f:
            f.write(response.content)

        return send_file(temp_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@app.route("/thumbnail/image/<image_id>")
def get_thumbnail(image_id):
    try:
        response = requests.get(f"{KAGGLE_ENDPOINT}/api/thumbnails/{image_id}")
        if response.status_code != 200:
            return jsonify(response.json()), response.status_code

        # Save the thumbnail temporarily
        temp_path = UPLOAD_FOLDER / f"{image_id}_thumb.jpg"
        with open(temp_path, "wb") as f:
            f.write(response.content)

        return send_file(temp_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/results/<result_id>")
def get_result_from_id(result_id):
    result_path = os.path.join(RESULT_FOLDER, f"{result_id}_result.jpg")
    if not os.path.exists(result_path):
        return jsonify({"error": "Result not found"}), 404
    return send_from_directory(RESULT_FOLDER, f"{result_id}_result.jpg")


@app.errorhandler(413)
def too_large(e):
    flash("File is too large. Maximum size is 16MB.")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
