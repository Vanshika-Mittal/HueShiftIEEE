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
KAGGLE_ENDPOINT = "http://localhost:8000"

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

    # Map GAN model outputs
    gan_source = os.path.join(gallery_base, "FinalGANResults", "output")
    if os.path.exists(gan_source):
        complexity_dirs = [
            d
            for d in os.listdir(gan_source)
            if os.path.isdir(os.path.join(gan_source, d))
        ]

        for complexity in complexity_dirs:
            complexity_source = os.path.join(gan_source, complexity)
            complexity_dir = os.path.join(gan_dir, complexity)

            if not os.path.exists(complexity_dir):
                try:
                    os.symlink(
                        complexity_source, complexity_dir, target_is_directory=True
                    )
                except OSError:
                    shutil.copytree(complexity_source, complexity_dir)


# Create gallery symlinks when the app starts
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

            response = requests.post(
                f"{KAGGLE_ENDPOINT}/api/process", files=files, verify=False
            )
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
    # Sample name mapping for more user-friendly display
    sample_name_mapping = {
        "v_ApplyLipstick_g05_c04": "Sample 1 - Apply Lipstick",
        "v_BlowingCandles_g06_c02": "Sample 2 - Blowing Candles",
        "v_CricketShot_g04_c04": "Sample 3 - Cricket Shot",
        "v_Drumming_g06_c01": "Sample 4 - Drumming",
        "v_Fencing_g04_c04": "Sample 5 - Fencing",
        "v_FrontCrawl_g02_c04": "Sample 6 - Front Crawl",
        "v_HandstandWalking_g21_c04": "Sample 7 - Handstand Walking",
        "v_HorseRace_g07_c01": "Sample 8 - Horse Race",
        "v_HorseRiding_g03_c04": "Sample 9 - Horse Riding",
        "v_IceDancing_g03_c02": "Sample 10 - Ice Dancing",
    }

    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Base directory: {base_dir}")

    # Get all diffusion model samples
    diffusion_samples = []
    diffusion_dir = os.path.join(
        base_dir,
        "Gallery",
        "Gallery",
        "Grey2Color",
    )

    print(f"Looking for diffusion samples in: {diffusion_dir}")

    # Get all sample directories from Post Processed
    post_processed_dir = os.path.join(diffusion_dir, "Post Processed")
    print(f"Post processed directory: {post_processed_dir}")
    print(f"Post processed directory exists: {os.path.exists(post_processed_dir)}")

    if os.path.exists(post_processed_dir):
        sample_dirs = [
            d
            for d in os.listdir(post_processed_dir)
            if os.path.isdir(os.path.join(post_processed_dir, d))
        ]

        print(f"Found sample directories: {sample_dirs}")

        for sample_dir in sample_dirs:
            sample_name = sample_dir
            display_name = sample_name_mapping.get(sample_name, sample_name)

            sample_data = {
                "name": sample_name,
                "display_name": display_name,
                "final_output": None,
                "high_res_output": None,
                "low_res_output": None,
                "ground_truth": None,
                "grayscale": None,
                "stage_1": None,
                "neural_filter": None,
                "global_info": None,
                "concatenated": None,
            }

            print(f"Processing sample: {sample_name}")

            # Check for final output
            final_dir = os.path.join(post_processed_dir, sample_dir, "final")
            final_video_path = os.path.join(final_dir, "output.mp4")
            print(f"Final output path: {final_video_path}")
            print(
                f"Final output exists: {os.path.exists(final_video_path) if os.path.exists(final_dir) else 'Directory not found'}"
            )

            if os.path.exists(final_dir) and os.path.exists(final_video_path):
                sample_data["final_output"] = url_for(
                    "static",
                    filename=f"gallery/diffusion/{sample_name}/final/output.mp4",
                )
                print(f"Final output URL: {sample_data['final_output']}")

            # Check for high-res output
            high_res_dir = os.path.join(diffusion_dir, "high_res_outputs", sample_dir)
            high_res_video_path = os.path.join(high_res_dir, "videos", "output.mp4")
            print(f"High-res output path: {high_res_video_path}")
            print(
                f"High-res output exists: {os.path.exists(high_res_video_path) if os.path.exists(high_res_dir) else 'Directory not found'}"
            )

            if os.path.exists(high_res_dir) and os.path.exists(high_res_video_path):
                sample_data["high_res_output"] = url_for(
                    "static",
                    filename=f"gallery/diffusion/{sample_name}/high_res/output.mp4",
                )
                print(f"High-res output URL: {sample_data['high_res_output']}")

            # Check for low-res output
            low_res_dir = os.path.join(diffusion_dir, "low_res_outputs", sample_dir)
            low_res_video_path = os.path.join(low_res_dir, "videos", "output.mp4")
            print(f"Low-res output path: {low_res_video_path}")
            print(
                f"Low-res output exists: {os.path.exists(low_res_video_path) if os.path.exists(low_res_dir) else 'Directory not found'}"
            )

            if os.path.exists(low_res_dir) and os.path.exists(low_res_video_path):
                sample_data["low_res_output"] = url_for(
                    "static",
                    filename=f"gallery/diffusion/{sample_name}/low_res/output.mp4",
                )
                print(f"Low-res output URL: {sample_data['low_res_output']}")

            # Check for ground truth
            ground_truth_dir = os.path.join(diffusion_dir, "Ground Truth", sample_dir)
            ground_truth_video_path = os.path.join(ground_truth_dir, "output.mp4")
            print(f"Ground truth path: {ground_truth_video_path}")
            print(
                f"Ground truth exists: {os.path.exists(ground_truth_video_path) if os.path.exists(ground_truth_dir) else 'Directory not found'}"
            )

            if os.path.exists(ground_truth_dir) and os.path.exists(
                ground_truth_video_path
            ):
                sample_data["ground_truth"] = url_for(
                    "static",
                    filename=f"gallery/diffusion/{sample_name}/ground_truth/output.mp4",
                )
                print(f"Ground truth URL: {sample_data['ground_truth']}")

            # Generate grayscale video if not exists
            grayscale_dir = os.path.join(diffusion_dir, "Grayscale", sample_dir)
            grayscale_video_path = os.path.join(grayscale_dir, "output.mp4")
            print(f"Grayscale path: {grayscale_video_path}")
            print(
                f"Grayscale exists: {os.path.exists(grayscale_video_path) if os.path.exists(grayscale_dir) else 'Directory not found'}"
            )

            if os.path.exists(ground_truth_dir) and os.path.exists(
                ground_truth_video_path
            ):
                # Ensure the grayscale directory exists
                os.makedirs(grayscale_dir, exist_ok=True)

                # If grayscale video doesn't exist, generate it from ground truth
                if not os.path.exists(grayscale_video_path):
                    try:
                        # This is a placeholder for the conversion code
                        # In a real implementation, you would use opencv or ffmpeg to convert
                        shutil.copy(ground_truth_video_path, grayscale_video_path)
                        print(
                            f"Created grayscale video (placeholder): {grayscale_video_path}"
                        )
                        # TODO: Implement actual grayscale conversion
                    except Exception as e:
                        print(f"Error creating grayscale video: {str(e)}")

                if os.path.exists(grayscale_video_path):
                    sample_data["grayscale"] = url_for(
                        "static",
                        filename=f"gallery/diffusion/{sample_name}/grayscale/output.mp4",
                    )
                    print(f"Grayscale URL: {sample_data['grayscale']}")

            # Check for stage_1
            stage_1_dir = os.path.join(post_processed_dir, sample_dir, "stage_1")
            stage_1_reconstruction_path = (
                os.path.join(stage_1_dir, "010000", f"reconstruction_{sample_name}.mp4")
                if os.path.exists(stage_1_dir)
                else None
            )
            print(f"Stage 1 path: {stage_1_reconstruction_path}")
            print(
                f"Stage 1 exists: {os.path.exists(stage_1_reconstruction_path) if stage_1_reconstruction_path else 'Directory not found'}"
            )

            if os.path.exists(stage_1_dir):
                global_info_path = os.path.join(
                    stage_1_dir, "010000", f"global_info_{sample_name}.mp4"
                )
                if os.path.exists(global_info_path):
                    sample_data["global_info"] = url_for(
                        "static",
                        filename=f"gallery/diffusion/{sample_name}/stage_1/010000/global_info_{sample_name}.mp4",
                    )
                    print(f"Global info URL: {sample_data['global_info']}")

                reconstruction_path = os.path.join(
                    stage_1_dir, "010000", f"reconstruction_{sample_name}.mp4"
                )
                if os.path.exists(reconstruction_path):
                    sample_data["stage_1"] = url_for(
                        "static",
                        filename=f"gallery/diffusion/{sample_name}/stage_1/010000/reconstruction_{sample_name}.mp4",
                    )
                    print(f"Stage 1 URL: {sample_data['stage_1']}")

            # Check for neural_filter
            neural_filter_dir = os.path.join(
                post_processed_dir, sample_dir, "neural_filter"
            )
            neural_filter_video_path = os.path.join(neural_filter_dir, "output.mp4")
            print(f"Neural filter path: {neural_filter_video_path}")
            print(
                f"Neural filter exists: {os.path.exists(neural_filter_video_path) if os.path.exists(neural_filter_dir) else 'Directory not found'}"
            )

            if os.path.exists(neural_filter_dir) and os.path.exists(
                neural_filter_video_path
            ):
                sample_data["neural_filter"] = url_for(
                    "static",
                    filename=f"gallery/diffusion/{sample_name}/neural_filter/output.mp4",
                )
                print(f"Neural filter URL: {sample_data['neural_filter']}")

            # Generate concatenated video with all outputs
            concat_dir = os.path.join(diffusion_dir, "Concatenated", sample_dir)
            concat_video_path = os.path.join(concat_dir, "output.mp4")

            # Ensure the concat directory exists
            os.makedirs(concat_dir, exist_ok=True)

            # If concatenated video doesn't exist, generate it (in a real implementation)
            if not os.path.exists(concat_video_path):
                # This is a placeholder - in a real impl, you would use ffmpeg to concatenate
                if sample_data["final_output"]:
                    try:
                        # Just copy the final output as a placeholder
                        shutil.copy(final_video_path, concat_video_path)
                        print(
                            f"Created concatenated video (placeholder): {concat_video_path}"
                        )
                        # TODO: Implement actual video concatenation
                    except Exception as e:
                        print(f"Error creating concatenated video: {str(e)}")

            if os.path.exists(concat_video_path):
                sample_data["concatenated"] = url_for(
                    "static",
                    filename=f"gallery/diffusion/{sample_name}/concatenated/output.mp4",
                )
                print(f"Concatenated URL: {sample_data['concatenated']}")

            diffusion_samples.append(sample_data)

    # Get all GAN model samples
    gan_samples = []
    gan_dir = os.path.join(
        base_dir,
        "Gallery",
        "FinalGANResults",
        "output",
    )

    print(f"Looking for GAN samples in: {gan_dir}")
    print(f"GAN directory exists: {os.path.exists(gan_dir)}")

    # GAN name mapping
    gan_name_mapping = {
        # Add mappings as needed for GAN filenames
    }

    if os.path.exists(gan_dir):
        complexity_dirs = [
            d for d in os.listdir(gan_dir) if os.path.isdir(os.path.join(gan_dir, d))
        ]

        print(f"Found complexity directories: {complexity_dirs}")

        sample_counter = 1

        for complexity in complexity_dirs:
            complexity_dir = os.path.join(gan_dir, complexity)

            # Get video files in this directory
            video_files = [f for f in os.listdir(complexity_dir) if f.endswith(".mp4")]

            print(f"Found video files in {complexity}: {video_files}")

            for video_file in video_files:
                original_name = video_file.split(".")[0]

                # Create a display name based on counter
                display_name = gan_name_mapping.get(original_name)
                if not display_name:
                    display_name = f"Sample {sample_counter} - {original_name.replace('_', ' ').title()}"
                    sample_counter += 1

                video_path = os.path.join(complexity_dir, video_file)
                print(f"GAN video path: {video_path}")
                print(f"GAN video exists: {os.path.exists(video_path)}")

                # Ensure static directory exists for GAN videos
                static_gan_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "static",
                    "gallery",
                    "gan",
                    complexity,
                )
                os.makedirs(static_gan_dir, exist_ok=True)

                # Create a symlink or copy the video to the static directory
                static_video_path = os.path.join(static_gan_dir, video_file)
                if not os.path.exists(static_video_path):
                    try:
                        os.symlink(video_path, static_video_path)
                    except OSError:
                        shutil.copy(video_path, static_video_path)
                    print(f"Created link for GAN video: {static_video_path}")

                url = url_for(
                    "static", filename=f"gallery/gan/{complexity}/{video_file}"
                )
                print(f"Generated GAN video URL: {url}")

                sample_data = {
                    "name": original_name,
                    "display_name": display_name,
                    "output": url,
                    "complexity": complexity,
                }

                gan_samples.append(sample_data)

    # Sort samples by their display names
    diffusion_samples.sort(key=lambda x: x["display_name"])
    gan_samples.sort(key=lambda x: x["display_name"])

    return render_template(
        "gallery.html", diffusion_samples=diffusion_samples, gan_samples=gan_samples
    )


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
        response = requests.get(
            f"{KAGGLE_ENDPOINT}/api/gallery", params=request.args, verify=False
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/result/image/<image_id>")
def get_result(image_id):
    try:
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
    print(f"Requested gallery path: {path}")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Base directory: {base_dir}")

    # First try to serve from our static directory (faster and more reliable)
    static_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static", "gallery", path
    )
    print(f"Checking static path: {static_path}")
    if os.path.exists(static_path) and os.path.isfile(static_path):
        print(f"Found file in static directory")
        directory, filename = os.path.split(static_path)
        return send_from_directory(directory, filename)

    # If not in static dir, try the actual Gallery folders
    if path.startswith("diffusion/"):
        # Extract sample name from path
        parts = path.split("/")
        sample_name = parts[1]
        file_type = parts[2]

        print(f"Serving diffusion sample: {sample_name}, file type: {file_type}")

        if file_type == "final":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "Post Processed",
                sample_name,
                "final",
                "output.mp4",
            )
            print(f"Final video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Post Processed",
                        sample_name,
                        "final",
                    ),
                    "output.mp4",
                )
        elif file_type == "high_res":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "high_res_outputs",
                sample_name,
                "videos",
                "output.mp4",
            )
            print(f"High-res video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "high_res_outputs",
                        sample_name,
                        "videos",
                    ),
                    "output.mp4",
                )
        elif file_type == "low_res":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "low_res_outputs",
                sample_name,
                "videos",
                "output.mp4",
            )
            print(f"Low-res video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "low_res_outputs",
                        sample_name,
                        "videos",
                    ),
                    "output.mp4",
                )
        elif file_type == "ground_truth":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "Ground Truth",
                sample_name,
                "output.mp4",
            )
            print(f"Ground truth video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Ground Truth",
                        sample_name,
                    ),
                    "output.mp4",
                )
        elif file_type == "grayscale":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "Grayscale",
                sample_name,
                "output.mp4",
            )
            print(f"Grayscale video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Grayscale",
                        sample_name,
                    ),
                    "output.mp4",
                )
        elif file_type == "concatenated":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "Concatenated",
                sample_name,
                "output.mp4",
            )
            print(f"Concatenated video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Concatenated",
                        sample_name,
                    ),
                    "output.mp4",
                )
        elif file_type == "stage_1":
            if len(parts) > 3:
                final_path = "/".join(parts[3:])
                print(f"Stage 1 path detail: {final_path}")

                if "global_info" in final_path:
                    file_path = os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Post Processed",
                        sample_name,
                        "stage_1",
                        "010000",
                        f"global_info_{sample_name}.mp4",
                    )
                    print(f"Global info video path: {file_path}")
                    print(f"File exists: {os.path.exists(file_path)}")
                    if os.path.exists(file_path):
                        return send_from_directory(
                            os.path.join(
                                base_dir,
                                "Gallery",
                                "Gallery",
                                "Grey2Color",
                                "Post Processed",
                                sample_name,
                                "stage_1",
                                "010000",
                            ),
                            f"global_info_{sample_name}.mp4",
                        )
                elif "reconstruction" in final_path:
                    file_path = os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Post Processed",
                        sample_name,
                        "stage_1",
                        "010000",
                        f"reconstruction_{sample_name}.mp4",
                    )
                    print(f"Reconstruction video path: {file_path}")
                    print(f"File exists: {os.path.exists(file_path)}")
                    if os.path.exists(file_path):
                        return send_from_directory(
                            os.path.join(
                                base_dir,
                                "Gallery",
                                "Gallery",
                                "Grey2Color",
                                "Post Processed",
                                sample_name,
                                "stage_1",
                                "010000",
                            ),
                            f"reconstruction_{sample_name}.mp4",
                        )
        elif file_type == "neural_filter":
            file_path = os.path.join(
                base_dir,
                "Gallery",
                "Gallery",
                "Grey2Color",
                "Post Processed",
                sample_name,
                "neural_filter",
                "output.mp4",
            )
            print(f"Neural filter video path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                return send_from_directory(
                    os.path.join(
                        base_dir,
                        "Gallery",
                        "Gallery",
                        "Grey2Color",
                        "Post Processed",
                        sample_name,
                        "neural_filter",
                    ),
                    "output.mp4",
                )

    elif path.startswith("gan/"):
        # Extract complexity and filename
        parts = path.split("/")
        complexity = parts[1]
        filename = parts[2]

        file_path = os.path.join(
            base_dir, "Gallery", "FinalGANResults", "output", complexity, filename
        )
        print(f"GAN video path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

        if os.path.exists(file_path):
            return send_from_directory(
                os.path.join(
                    base_dir, "Gallery", "FinalGANResults", "output", complexity
                ),
                filename,
            )

    print("File not found for path:", path)
    return "File not found", 404


# For Vercel serverless deployment
app.debug = False
app.config["TEMPLATES_AUTO_RELOAD"] = False

# Vercel expects an 'app' variable
application = app
