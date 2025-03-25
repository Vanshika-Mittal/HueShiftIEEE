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

# Config
UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = "results"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "wmv"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
# This should be a much smaller value, kept it at 16MB for now


KAGGLE_API_URL = ()

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


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_image_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = generate_unique_filename(filename)
            file_path = UPLOAD_FOLDER / unique_filename
            file.save(file_path)

            # Here we will process the image with colorization model
            # For now, it just returns success message
            flash(
                "Image uploaded successfully. Processing will be implemented in future phases."
            )

            return redirect(url_for("upload_image"))
        else:
            flash(
                "Invalid file format. Please upload an image file (png, jpg, jpeg, gif)."
            )
            return redirect(request.url)

    return render_template("upload_image.html")


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


@app.route("/colorize/image", methods=["POST"])
def colorize_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save file temporarily
        temp_path = UPLOAD_FOLDER / file.filename
        file.save(temp_path)

        # Forward to Kaggle API
        with open(temp_path, "rb") as f:
            files = {"file": (file.filename, f, "image/jpeg")}
            response = requests.post(f"{KAGGLE_API_URL}/colorize/image", files=files)

        # Cleanup
        temp_path.unlink()

        if response.status_code != 200:
            return jsonify(response.json()), response.status_code

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gallery")
def get_gallery():
    try:
        response = requests.get(f"{KAGGLE_API_URL}/gallery", params=request.args)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/result/image/<image_id>")
def get_result(image_id):
    try:
        response = requests.get(f"{KAGGLE_API_URL}/result/image/{image_id}")
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
        response = requests.get(f"{KAGGLE_API_URL}/thumbnail/image/{image_id}")
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


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.errorhandler(413)
def too_large(e):
    flash("File is too large. Maximum size is 16MB.")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
