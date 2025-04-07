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
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config
UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = "results"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "wmv", "webm"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload

# Dummy endpoint - not actually used but referenced in code
KAGGLE_ENDPOINT = None  # Set to None to prevent connection attempts

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "dev-secret-key"  # This is not needed in development env

UPLOAD_FOLDER.mkdir(exist_ok=True)
os.makedirs(os.path.join(app.root_path, RESULT_FOLDER), exist_ok=True)

# Create static directory for gallery symlinks if it doesn't exist
static_gallery_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static", "gallery"
)
os.makedirs(static_gallery_dir, exist_ok=True)


# Create symlinks to the Gallery folders for web access
def create_gallery_symlinks():
    # Base directory paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gallery_base = os.path.join(base_dir, "Gallery")

    # Create static/gallery/diffusion directory
    diffusion_dir = os.path.join(static_gallery_dir, "diffusion")
    os.makedirs(diffusion_dir, exist_ok=True)

    # Create static/gallery/gan directory
    gan_dir = os.path.join(static_gallery_dir, "gan")
    os.makedirs(gan_dir, exist_ok=True)

    # Create subdirectories for GAN complexity levels
    gan_simple_dir = os.path.join(gan_dir, "simple")
    gan_medium_dir = os.path.join(gan_dir, "slightly-complex")
    gan_complex_dir = os.path.join(gan_dir, "very-complex")

    os.makedirs(gan_simple_dir, exist_ok=True)
    os.makedirs(gan_medium_dir, exist_ok=True)
    os.makedirs(gan_complex_dir, exist_ok=True)

    # Map diffusion model outputs
    diffusion_source = os.path.join(gallery_base, "Gallery", "Grey2Color")
    if os.path.exists(diffusion_source):
        # Create Grayscale and Concatenated directories if they don't exist
        grayscale_base_dir = os.path.join(diffusion_source, "Grayscale")
        concat_base_dir = os.path.join(diffusion_source, "Concatenated")
        os.makedirs(grayscale_base_dir, exist_ok=True)
        os.makedirs(concat_base_dir, exist_ok=True)

        # Create symbolic links for each required sample directory
        post_processed_dir = os.path.join(diffusion_source, "Post Processed")
        if os.path.exists(post_processed_dir):
            sample_dirs = [
                d
                for d in os.listdir(post_processed_dir)
                if os.path.isdir(os.path.join(post_processed_dir, d))
            ]

            for sample_dir in sample_dirs:
                # Create sample directory in static folder
                sample_static_dir = os.path.join(diffusion_dir, sample_dir)
                os.makedirs(sample_static_dir, exist_ok=True)

                # Ensure required directories exist in the source
                grayscale_dir = os.path.join(grayscale_base_dir, sample_dir)
                concat_dir = os.path.join(concat_base_dir, sample_dir)
                os.makedirs(grayscale_dir, exist_ok=True)
                os.makedirs(concat_dir, exist_ok=True)

                # Link grayscale output
                grayscale_static_dir = os.path.join(sample_static_dir, "grayscale")
                if os.path.exists(grayscale_dir) and not os.path.exists(
                    grayscale_static_dir
                ):
                    try:
                        os.symlink(
                            grayscale_dir,
                            grayscale_static_dir,
                            target_is_directory=True,
                        )
                    except OSError:
                        shutil.copytree(grayscale_dir, grayscale_static_dir)

                # Link concatenated output
                concat_static_dir = os.path.join(sample_static_dir, "concatenated")
                if os.path.exists(concat_dir) and not os.path.exists(concat_static_dir):
                    try:
                        os.symlink(
                            concat_dir, concat_static_dir, target_is_directory=True
                        )
                    except OSError:
                        shutil.copytree(concat_dir, concat_static_dir)

                # Link final output
                final_dir = os.path.join(sample_static_dir, "final")
                final_source = os.path.join(post_processed_dir, sample_dir, "final")
                if os.path.exists(final_source) and not os.path.exists(final_dir):
                    try:
                        os.symlink(final_source, final_dir, target_is_directory=True)
                    except OSError:
                        shutil.copytree(final_source, final_dir)

                # Link high-res output
                high_res_dir = os.path.join(sample_static_dir, "high_res")
                high_res_source = os.path.join(
                    diffusion_source, "high_res_outputs", sample_dir, "videos"
                )
                if os.path.exists(high_res_source) and not os.path.exists(high_res_dir):
                    try:
                        os.symlink(
                            high_res_source, high_res_dir, target_is_directory=True
                        )
                    except OSError:
                        shutil.copytree(high_res_source, high_res_dir)

                # Link low-res output
                low_res_dir = os.path.join(sample_static_dir, "low_res")
                low_res_source = os.path.join(
                    diffusion_source, "low_res_outputs", sample_dir, "videos"
                )
                if os.path.exists(low_res_source) and not os.path.exists(low_res_dir):
                    try:
                        os.symlink(
                            low_res_source, low_res_dir, target_is_directory=True
                        )
                    except OSError:
                        shutil.copytree(low_res_source, low_res_dir)

                # Link ground truth
                ground_truth_dir = os.path.join(sample_static_dir, "ground_truth")
                ground_truth_source = os.path.join(
                    diffusion_source, "Ground Truth", sample_dir
                )
                if os.path.exists(ground_truth_source) and not os.path.exists(
                    ground_truth_dir
                ):
                    try:
                        os.symlink(
                            ground_truth_source,
                            ground_truth_dir,
                            target_is_directory=True,
                        )
                    except OSError:
                        shutil.copytree(ground_truth_source, ground_truth_dir)

                # Link stage_1
                stage_1_dir = os.path.join(sample_static_dir, "stage_1")
                stage_1_source = os.path.join(post_processed_dir, sample_dir, "stage_1")
                if os.path.exists(stage_1_source) and not os.path.exists(stage_1_dir):
                    try:
                        os.symlink(
                            stage_1_source, stage_1_dir, target_is_directory=True
                        )
                    except OSError:
                        shutil.copytree(stage_1_source, stage_1_dir)

                # Link neural_filter
                neural_filter_dir = os.path.join(sample_static_dir, "neural_filter")
                neural_filter_source = os.path.join(
                    post_processed_dir, sample_dir, "neural_filter"
                )
                if os.path.exists(neural_filter_source) and not os.path.exists(
                    neural_filter_dir
                ):
                    try:
                        os.symlink(
                            neural_filter_source,
                            neural_filter_dir,
                            target_is_directory=True,
                        )
                    except OSError:
                        shutil.copytree(neural_filter_source, neural_filter_dir)

                # Create simple grayscale video if it doesn't exist
                ground_truth_video = os.path.join(ground_truth_source, "output.mp4")
                grayscale_video = os.path.join(grayscale_dir, "output.mp4")

                if os.path.exists(ground_truth_video) and not os.path.exists(
                    grayscale_video
                ):
                    try:
                        # Just copy the ground truth video as placeholder for now
                        # In a real implementation, this would be a proper grayscale conversion
                        shutil.copy(ground_truth_video, grayscale_video)
                        print(
                            f"Created grayscale video (placeholder): {grayscale_video}"
                        )
                    except Exception as e:
                        print(f"Error creating grayscale video: {str(e)}")

                # Create simple concatenated video if it doesn't exist
                concat_video = os.path.join(concat_dir, "output.mp4")
                final_video = os.path.join(final_source, "output.mp4")

                if os.path.exists(final_video) and not os.path.exists(concat_video):
                    try:
                        # Just copy the final video as placeholder for now
                        # In a real implementation, this would be a proper concatenation
                        shutil.copy(final_video, concat_video)
                        print(
                            f"Created concatenated video (placeholder): {concat_video}"
                        )
                    except Exception as e:
                        print(f"Error creating concatenated video: {str(e)}")

    # Map GAN outputs from FinalGANResults directory
    gan_source_base = os.path.join(gallery_base, "FinalGANResults", "output")

    # Link simple scenes
    gan_simple_source = os.path.join(gan_source_base, "simple")
    if os.path.exists(gan_simple_source):
        for file in os.listdir(gan_simple_source):
            if file.endswith(".mp4"):
                source_file = os.path.join(gan_simple_source, file)
                target_file = os.path.join(gan_simple_dir, file)
                if not os.path.exists(target_file):
                    try:
                        os.symlink(source_file, target_file)
                        print(f"Created symlink for {file} in simple directory")
                    except OSError:
                        shutil.copy2(source_file, target_file)
                        print(f"Copied {file} to simple directory")

    # Link slightly-complex scenes
    gan_medium_source = os.path.join(gan_source_base, "slightly-complex")
    if os.path.exists(gan_medium_source):
        for file in os.listdir(gan_medium_source):
            if file.endswith(".mp4"):
                source_file = os.path.join(gan_medium_source, file)
                target_file = os.path.join(gan_medium_dir, file)
                if not os.path.exists(target_file):
                    try:
                        os.symlink(source_file, target_file)
                        print(
                            f"Created symlink for {file} in slightly-complex directory"
                        )
                    except OSError:
                        shutil.copy2(source_file, target_file)
                        print(f"Copied {file} to slightly-complex directory")

    # Link very-complex scenes
    gan_complex_source = os.path.join(gan_source_base, "very-complex")
    if os.path.exists(gan_complex_source):
        for file in os.listdir(gan_complex_source):
            if file.endswith(".mp4"):
                source_file = os.path.join(gan_complex_source, file)
                target_file = os.path.join(gan_complex_dir, file)
                if not os.path.exists(target_file):
                    try:
                        os.symlink(source_file, target_file)
                        print(f"Created symlink for {file} in very-complex directory")
                    except OSError:
                        shutil.copy2(source_file, target_file)
                        print(f"Copied {file} to very-complex directory")


# Call the function to create symlinks when the app starts
create_gallery_symlinks()


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


@app.route("/upload")
def upload():
    return render_template("upload.html")


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

            if KAGGLE_ENDPOINT:
                response = requests.post(
                    f"{KAGGLE_ENDPOINT}/api/process", files=files, verify=False
                )
                print(f"Response status code: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response content: {response.content}")
            else:
                print("KAGGLE_ENDPOINT is None, skipping request")
                return jsonify({"error": "KAGGLE_ENDPOINT is None"}), 500

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
        if KAGGLE_ENDPOINT:
            result_response = requests.get(
                f"{KAGGLE_ENDPOINT}/api/results/{result_id}", verify=False
            )
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
        else:
            print("KAGGLE_ENDPOINT is None, skipping result download")
            return jsonify({"error": "KAGGLE_ENDPOINT is None"}), 500

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


@app.route("/gallery-old")
def gallery():
    """
    Redirect old gallery route to the new one
    """
    return redirect(url_for("gallery_new"))


@app.route("/gallery")
def gallery_new():
    """
    Display the new gallery page with videos.
    """
    # Get all GAN samples for the GAN tab
    gan_samples = []
    gan_dir = os.path.join(static_gallery_dir, "gan")

    if os.path.exists(gan_dir):
        complexity_dirs = {
            "simple": "Simple",
            "slightly-complex": "Medium Complexity",
            "very-complex": "Complex",
        }

        for complexity, display_name in complexity_dirs.items():
            complexity_dir = os.path.join(gan_dir, complexity)
            if os.path.exists(complexity_dir):
                # Get .webm files instead of .mp4 files since they work better
                webm_files = [
                    f for f in os.listdir(complexity_dir) if f.endswith(".webm")
                ]

                for file in webm_files:
                    # Format the display name from the filename
                    name_parts = file.split("_v_")
                    if len(name_parts) >= 2:
                        # Get the second part and remove extension
                        second_part = name_parts[1].rsplit(".", 1)[0]
                        # Split by underscores, then capitalize and join parts
                        parts = second_part.split("_")
                        formatted_name = " ".join([p.capitalize() for p in parts])
                    else:
                        # Fallback if the file doesn't have the expected format
                        formatted_name = os.path.splitext(file)[0]
                        formatted_name = formatted_name.replace("_", " ").title()

                    gan_samples.append(
                        {
                            "display_name": formatted_name,
                            "input_video": f"/static/gallery/gan/{complexity}/{file}",
                            "output_video": f"/static/gallery/gan/{complexity}/{file}",
                            "complexity": complexity,
                        }
                    )

    return render_template("gallery.html", gan_samples=gan_samples)


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

        # Skip API requests if KAGGLE_ENDPOINT is None
        if KAGGLE_ENDPOINT is None:
            print("KAGGLE_ENDPOINT is None, skipping API request")
            return jsonify({"error": "API endpoint not configured"}), 500

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


@app.route("/result/image/<image_id>")
def get_result(image_id):
    try:
        if KAGGLE_ENDPOINT is None:
            print("KAGGLE_ENDPOINT is None, skipping API request")
            return jsonify({"error": "API endpoint not configured"}), 500

        response = requests.get(
            f"{KAGGLE_ENDPOINT}/api/results/{image_id}", verify=False
        )
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
        if KAGGLE_ENDPOINT is None:
            print("KAGGLE_ENDPOINT is None, skipping API request")
            return jsonify({"error": "API endpoint not configured"}), 500

        response = requests.get(
            f"{KAGGLE_ENDPOINT}/api/thumbnails/{image_id}", verify=False
        )
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
    flash("File is too large. Maximum size is 50MB.")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


# Add a static route for gallery files
@app.route("/static/gallery/<path:path>")
def serve_gallery(path):
    """
    Serve files from the gallery directory.
    This is needed because static files might be symlinks to other locations.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # First try to serve from our static directory
    static_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static", "gallery", path
    )

    # Basic path handling
    if os.path.exists(static_path) and os.path.isfile(static_path):
        directory, filename = os.path.split(static_path)

        # For WebM videos, set the correct MIME type
        if filename.endswith(".webm"):
            response = send_from_directory(directory, filename, mimetype="video/webm")
            # Add cache control headers
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

        return send_from_directory(directory, filename)

    # File doesn't exist
    return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)

# app.debug = False
# app.config["TEMPLATES_AUTO_RELOAD"] = False

# Vercel expects an 'app' variable
# application = app
