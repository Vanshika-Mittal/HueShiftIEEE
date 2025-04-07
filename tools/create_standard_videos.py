import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import glob
import argparse


def get_base_dirs():
    """Get the base directories for the project"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    website_dir = os.path.dirname(script_dir)
    base_dir = os.path.dirname(website_dir)

    gallery_dir = os.path.join(base_dir, "Gallery", "Gallery", "Grey2Color")
    static_dir = os.path.join(website_dir, "static", "gallery", "diffusion")

    print(f"Gallery directory: {gallery_dir}")
    print(f"Static directory: {static_dir}")

    return gallery_dir, static_dir


def get_sample_dirs(gallery_dir):
    """Get all sample directories in the Post Processed folder"""
    post_processed_dir = os.path.join(gallery_dir, "Post Processed")
    return [
        d
        for d in os.listdir(post_processed_dir)
        if os.path.isdir(os.path.join(post_processed_dir, d))
    ]


def convert_to_grayscale(img):
    """Convert an image to grayscale"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def create_video_from_frames(frames, output_path, fps=10, size=None):
    """Create a video from frames"""
    if not frames:
        print(f"No frames to create video at {output_path}")
        return False

    # Get size from first frame if not specified
    if size is None:
        size = (frames[0].shape[1], frames[0].shape[0])

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size, True)

    # Write frames
    for frame in frames:
        # Convert grayscale frames to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Resize if necessary
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    print(f"Created video: {output_path}")
    return True


def get_reference_video_info(gallery_dir, sample_name):
    """Get information about a reference video for consistent settings"""
    stage_1_dir = os.path.join(
        gallery_dir, "Post Processed", sample_name, "stage_1", "010000"
    )

    # Look for reference videos
    for file in os.listdir(stage_1_dir):
        if file.endswith(".mp4"):
            video_path = os.path.join(stage_1_dir, file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 10  # Default if cannot be determined

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"Reference video: {video_path}")
            print(f"Size: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
            }

    # Default values if no reference video found
    return {"width": 320, "height": 240, "fps": 10, "total_frames": 30}


def read_frames_from_directory(directory, name_pattern="frame_*.jpg"):
    """Read frames from a directory based on name pattern"""
    frame_files = sorted(glob.glob(os.path.join(directory, name_pattern)))
    frames = []

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            frames.append(frame)

    print(f"Read {len(frames)} frames from {directory}")
    return frames


def process_sample(gallery_dir, static_dir, sample_name):
    """Process a single sample directory"""
    print(f"\nProcessing sample: {sample_name}")

    # Create sample directory in static
    sample_static_dir = os.path.join(static_dir, sample_name)
    os.makedirs(sample_static_dir, exist_ok=True)

    # Get reference video info for consistent settings
    video_info = get_reference_video_info(gallery_dir, sample_name)
    video_size = (video_info["width"], video_info["height"])
    fps = video_info["fps"]

    # Define source and destination directories
    ground_truth_dir = os.path.join(gallery_dir, "Ground Truth", sample_name)
    low_res_dir = os.path.join(gallery_dir, "low_res_outputs", sample_name, "videos")
    high_res_dir = os.path.join(gallery_dir, "high_res_outputs", sample_name, "videos")
    final_dir = os.path.join(gallery_dir, "Post Processed", sample_name, "final")

    # Create destination directories in static
    ground_truth_static_dir = os.path.join(sample_static_dir, "ground_truth")
    grayscale_static_dir = os.path.join(sample_static_dir, "grayscale")
    low_res_static_dir = os.path.join(sample_static_dir, "low_res")
    high_res_static_dir = os.path.join(sample_static_dir, "high_res")
    final_static_dir = os.path.join(sample_static_dir, "final")

    os.makedirs(ground_truth_static_dir, exist_ok=True)
    os.makedirs(grayscale_static_dir, exist_ok=True)
    os.makedirs(low_res_static_dir, exist_ok=True)
    os.makedirs(high_res_static_dir, exist_ok=True)
    os.makedirs(final_static_dir, exist_ok=True)

    # Process Ground Truth video
    ground_truth_frames = read_frames_from_directory(ground_truth_dir)
    if ground_truth_frames:
        # Create standard ground truth video
        ground_truth_output = os.path.join(
            ground_truth_static_dir, f"{sample_name}_standardized.mp4"
        )
        create_video_from_frames(
            ground_truth_frames, ground_truth_output, fps=fps, size=video_size
        )

        # Create grayscale video
        grayscale_frames = [
            convert_to_grayscale(frame) for frame in ground_truth_frames
        ]
        grayscale_output = os.path.join(
            grayscale_static_dir, f"{sample_name}_standardized.mp4"
        )
        create_video_from_frames(
            grayscale_frames, grayscale_output, fps=fps, size=video_size
        )

    # Process Low-res video
    low_res_mp4 = os.path.join(low_res_dir, "output.mp4")
    if os.path.exists(low_res_mp4):
        # Read existing video and create standardized version
        cap = cv2.VideoCapture(low_res_mp4)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if frames:
            low_res_output = os.path.join(
                low_res_static_dir, f"{sample_name}_standardized.mp4"
            )
            create_video_from_frames(frames, low_res_output, fps=fps, size=video_size)

    # Process High-res video
    high_res_mp4 = os.path.join(high_res_dir, "output.mp4")
    if os.path.exists(high_res_mp4):
        # Read existing video and create standardized version
        cap = cv2.VideoCapture(high_res_mp4)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if frames:
            high_res_output = os.path.join(
                high_res_static_dir, f"{sample_name}_standardized.mp4"
            )
            create_video_from_frames(frames, high_res_output, fps=fps, size=video_size)

    # Process Final video
    final_mp4 = os.path.join(final_dir, "output.mp4")
    if os.path.exists(final_mp4):
        # Read existing video and create standardized version
        cap = cv2.VideoCapture(final_mp4)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if frames:
            final_output = os.path.join(
                final_static_dir, f"{sample_name}_standardized.mp4"
            )
            create_video_from_frames(frames, final_output, fps=fps, size=video_size)

    return {
        "ground_truth": os.path.join(
            ground_truth_static_dir, f"{sample_name}_standardized.mp4"
        )
        if ground_truth_frames
        else None,
        "grayscale": os.path.join(
            grayscale_static_dir, f"{sample_name}_standardized.mp4"
        )
        if ground_truth_frames
        else None,
        "low_res": os.path.join(low_res_static_dir, f"{sample_name}_standardized.mp4")
        if os.path.exists(low_res_mp4)
        else None,
        "high_res": os.path.join(high_res_static_dir, f"{sample_name}_standardized.mp4")
        if os.path.exists(high_res_mp4)
        else None,
        "final": os.path.join(final_static_dir, f"{sample_name}_standardized.mp4")
        if os.path.exists(final_mp4)
        else None,
    }


def update_app_paths(sample_name, video_paths):
    """Update the app.py to use the standardized videos"""
    website_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(website_dir, "app.py")

    with open(app_path, "r") as f:
        app_content = f.read()

    # Update paths for each video type
    for video_type, path in video_paths.items():
        if path:
            # Extract the relative path from the static directory
            rel_path = os.path.relpath(path, os.path.join(website_dir, "static"))
            url_path = f"gallery/diffusion/{sample_name}/{video_type}/{sample_name}_standardized.mp4"

            # Replace the URL patterns in app.py
            pattern = (
                f'filename=f"gallery/diffusion/{sample_name}/{video_type}/output.mp4"'
            )
            replacement = f'filename="{url_path}"'
            app_content = app_content.replace(pattern, replacement)

    # Write the updated content back
    with open(app_path, "w") as f:
        f.write(app_content)

    print(f"Updated app.py with standardized video paths for {sample_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create standardized videos from gallery frames"
    )
    parser.add_argument("--sample", help="Process only a specific sample")
    args = parser.parse_args()

    gallery_dir, static_dir = get_base_dirs()

    # Get sample directories
    if args.sample:
        sample_dirs = [args.sample]
    else:
        sample_dirs = get_sample_dirs(gallery_dir)

    print(f"Found {len(sample_dirs)} sample directories")

    # Process each sample
    for sample_name in sample_dirs:
        video_paths = process_sample(gallery_dir, static_dir, sample_name)
        # update_app_paths(sample_name, video_paths)

    print("\nCompleted processing all samples")


if __name__ == "__main__":
    main()
