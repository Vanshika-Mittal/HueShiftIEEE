import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path


def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path


def generate_video_from_frames(frames_dir, output_path, fps=30):
    """Generate an MP4 video from a sequence of image frames"""
    print(f"Generating video from frames in {frames_dir} to {output_path}")

    # Check if output already exists
    if os.path.exists(output_path):
        print(f"Video already exists: {output_path}")
        return True

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if frames exist
    frames = sorted(
        [
            f
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not frames:
        print(f"No frames found in {frames_dir}")
        return False

    print(f"Found {len(frames)} frames in {frames_dir}")

    # Get first frame to determine dimensions
    first_frame_path = os.path.join(frames_dir, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error reading first frame: {first_frame_path}")
        return False

    height, width, _ = first_frame.shape
    print(f"Frame dimensions: {width}x{height}")

    # Create video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video.isOpened():
            print(f"Failed to create video writer for {output_path}")
            return False

        # Add frames to video
        for i, frame_name in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video.write(frame)
                if i % 10 == 0:
                    print(f"Added frame {i + 1}/{len(frames)} to video: {frame_name}")
            else:
                print(f"Error reading frame: {frame_path}")

        # Release video writer
        video.release()

        # Verify the output file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Video generated successfully: {output_path}")
            print(
                f"Video file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB"
            )
            return True
        else:
            print(f"Video file not created or empty: {output_path}")
            return False
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        return False


def generate_grayscale_video(color_video_path, output_path, fps=30):
    """Generate a grayscale version of a color video

    Args:
        color_video_path: Path to the input color video
        output_path: Path to save the grayscale output video
        fps: Frames per second (default: 30, but will use source video fps)

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Generating grayscale video from {color_video_path} to {output_path}")

    # Check if output already exists
    if os.path.exists(output_path):
        print(f"Grayscale video already exists: {output_path}")
        return True

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if input video exists
    if not os.path.exists(color_video_path):
        print(f"Input video not found: {color_video_path}")
        return False

    # Open video file
    cap = cv2.VideoCapture(color_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {color_video_path}")
        return False

    try:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)

        # IMPORTANT: Always use the source video's framerate to ensure sync
        if input_fps > 0:
            fps = input_fps
            print(f"Using source video framerate: {fps} FPS")
        else:
            print(f"Source video framerate unavailable, using default: {fps} FPS")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video dimensions: {width}x{height}, FPS: {fps}, Frames: {frame_count}")

        # Create video writer for grayscale output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Failed to create output video: {output_path}")
            cap.release()
            return False

        # Process frames with progress tracking
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            if current_frame % 30 == 0 or current_frame == frame_count:
                print(
                    f"Processing frame {current_frame}/{frame_count} ({current_frame / frame_count * 100:.1f}%)"
                )

            # Convert to grayscale and back to BGR (since VideoWriter expects BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Write the grayscale frame
            out.write(gray_bgr)

        # Release resources
        cap.release()
        out.release()

        # Verify the output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Grayscale video generated successfully: {output_path}")
            print(
                f"Output file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB"
            )

            # Verify we can read back the video to confirm it's valid
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_cap.release()
                print(
                    f"Verified output video: {test_fps} FPS, {test_frame_count} frames"
                )
            else:
                print(f"Warning: Could not verify output video: {output_path}")

            return True
        else:
            print(f"Error: Output file is empty or not created: {output_path}")
            return False

    except Exception as e:
        print(f"Error generating grayscale video: {str(e)}")
        if "cap" in locals() and cap.isOpened():
            cap.release()
        if "out" in locals() and out.isOpened():
            out.release()
        return False


def process_ground_truth_frames():
    """Process Ground Truth frames to create videos"""
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    gallery_dir = os.path.join(base_dir, "Gallery", "Gallery", "Grey2Color")

    print(f"Base gallery directory: {gallery_dir}")
    print(f"Gallery directory exists: {os.path.exists(gallery_dir)}")

    # Create the necessary directories
    ground_truth_dir = os.path.join(gallery_dir, "Ground Truth")
    grayscale_dir = os.path.join(gallery_dir, "Grayscale")
    os.makedirs(grayscale_dir, exist_ok=True)

    print(f"Ground Truth directory: {ground_truth_dir}")
    print(f"Ground Truth directory exists: {os.path.exists(ground_truth_dir)}")

    # Process each sample directory
    if os.path.exists(ground_truth_dir):
        sample_dirs = [
            d
            for d in os.listdir(ground_truth_dir)
            if os.path.isdir(os.path.join(ground_truth_dir, d))
        ]

        print(f"Found {len(sample_dirs)} sample directories")

        for sample_dir in sample_dirs:
            sample_ground_truth_dir = os.path.join(ground_truth_dir, sample_dir)
            print(f"\nProcessing sample directory: {sample_dir}")
            print(f"Full path: {sample_ground_truth_dir}")

            # Create ground truth video if it doesn't exist
            ground_truth_video = os.path.join(sample_ground_truth_dir, "output.mp4")

            # Check for frames in the directory
            frame_files = [
                f
                for f in os.listdir(sample_ground_truth_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            frames_exist = len(frame_files) > 0

            print(f"Sample has {len(frame_files)} frame files")
            print(f"Ground truth video exists: {os.path.exists(ground_truth_video)}")

            # Generate video from frames
            if frames_exist and not os.path.exists(ground_truth_video):
                print(f"Creating ground truth video for {sample_dir}")
                result = generate_video_from_frames(
                    sample_ground_truth_dir, ground_truth_video
                )
                print(f"Video generation result: {'Success' if result else 'Failed'}")

            # Always recreate grayscale video from ground truth video to ensure correct framerate
            if os.path.exists(ground_truth_video):
                # Ensure grayscale directory exists
                sample_grayscale_dir = os.path.join(grayscale_dir, sample_dir)
                os.makedirs(sample_grayscale_dir, exist_ok=True)

                grayscale_video = os.path.join(sample_grayscale_dir, "output.mp4")
                print(f"Grayscale video path: {grayscale_video}")
                print(f"Grayscale video exists: {os.path.exists(grayscale_video)}")

                # Delete existing grayscale video to force recreation
                if os.path.exists(grayscale_video):
                    print(
                        f"Removing existing grayscale video to regenerate with correct framerate"
                    )
                    try:
                        os.remove(grayscale_video)
                    except Exception as e:
                        print(f"Error removing existing grayscale video: {str(e)}")

                print(f"Creating grayscale video for {sample_dir}")
                result = generate_grayscale_video(ground_truth_video, grayscale_video)
                print(
                    f"Grayscale video generation result: {'Success' if result else 'Failed'}"
                )
            else:
                print(
                    f"Ground truth video doesn't exist, can't create grayscale version"
                )
    else:
        print(f"Ground Truth directory not found: {ground_truth_dir}")

        # Try to find where the frames might be located
        print("Searching for possible frame directories...")
        for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(base_dir))):
            if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
                print(f"Found directory with frames: {root}")
                if (
                    len(files) > 5
                ):  # Assuming more than 5 files indicates a video frame sequence
                    print(
                        f"  Contains {len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])} image files"
                    )


def ensure_all_video_formats():
    """Ensure that all video formats are available by copying from existing formats if needed"""
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    gallery_dir = os.path.join(base_dir, "Gallery", "Gallery", "Grey2Color")

    # Define the directories
    ground_truth_dir = os.path.join(gallery_dir, "Ground Truth")
    grayscale_dir = os.path.join(gallery_dir, "Grayscale")
    post_processed_dir = os.path.join(gallery_dir, "Post Processed")
    low_res_dir = os.path.join(gallery_dir, "low_res_outputs")
    high_res_dir = os.path.join(gallery_dir, "high_res_outputs")
    concatenated_dir = os.path.join(gallery_dir, "Concatenated")

    # Create directories if they don't exist
    os.makedirs(grayscale_dir, exist_ok=True)
    os.makedirs(concatenated_dir, exist_ok=True)

    # Get sample directories
    if os.path.exists(post_processed_dir):
        sample_dirs = [
            d
            for d in os.listdir(post_processed_dir)
            if os.path.isdir(os.path.join(post_processed_dir, d))
        ]

        print(f"Processing {len(sample_dirs)} samples")

        for sample_dir in sample_dirs:
            print(f"\nProcessing sample: {sample_dir}")

            # Define paths for all video types
            ground_truth_video = os.path.join(
                ground_truth_dir, sample_dir, "output.mp4"
            )
            grayscale_video_dir = os.path.join(grayscale_dir, sample_dir)
            grayscale_video = os.path.join(grayscale_video_dir, "output.mp4")
            low_res_video = os.path.join(
                low_res_dir, sample_dir, "videos", "output.mp4"
            )
            high_res_video = os.path.join(
                high_res_dir, sample_dir, "videos", "output.mp4"
            )
            final_video = os.path.join(
                post_processed_dir, sample_dir, "final", "output.mp4"
            )
            concat_video_dir = os.path.join(concatenated_dir, sample_dir)
            concat_video = os.path.join(concat_video_dir, "output.mp4")

            # Create directories if they don't exist
            os.makedirs(grayscale_video_dir, exist_ok=True)
            os.makedirs(concat_video_dir, exist_ok=True)

            # Check and create grayscale video if needed
            if os.path.exists(ground_truth_video) and not os.path.exists(
                grayscale_video
            ):
                print(f"Creating grayscale video for {sample_dir}")
                generate_grayscale_video(ground_truth_video, grayscale_video)

            # Check and create concatenated video if needed
            if not os.path.exists(concat_video):
                print(f"Creating concatenated video for {sample_dir}")

                # Use final video as fallback if it exists
                if os.path.exists(final_video):
                    print(f"Using final video as placeholder for concatenated video")
                    shutil.copy(final_video, concat_video)
                    print(f"Concatenated video created (placeholder): {concat_video}")
                else:
                    print(f"No source video found for concatenated video")

            # Report on video availability
            print(
                f"Ground Truth video: {'✓' if os.path.exists(ground_truth_video) else '✗'}"
            )
            print(f"Grayscale video: {'✓' if os.path.exists(grayscale_video) else '✗'}")
            print(f"Low-res video: {'✓' if os.path.exists(low_res_video) else '✗'}")
            print(f"High-res video: {'✓' if os.path.exists(high_res_video) else '✗'}")
            print(f"Final video: {'✓' if os.path.exists(final_video) else '✗'}")
            print(f"Concatenated video: {'✓' if os.path.exists(concat_video) else '✗'}")


def link_static_directories():
    """Link or copy Gallery directories to static directory for web serving"""
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    gallery_dir = os.path.join(base_dir, "Gallery", "Gallery", "Grey2Color")
    static_gallery_dir = os.path.join(
        base_dir, "website", "static", "gallery", "diffusion"
    )

    print(f"Linking gallery directories to static folder for web serving")
    print(f"Source gallery: {gallery_dir}")
    print(f"Target static directory: {static_gallery_dir}")

    os.makedirs(static_gallery_dir, exist_ok=True)

    # Define the directories to link
    source_dirs = ["Ground Truth", "Grayscale", "Post Processed", "Concatenated"]

    for source_dir_name in source_dirs:
        source_dir = os.path.join(gallery_dir, source_dir_name)
        if os.path.exists(source_dir):
            print(f"Processing {source_dir_name} directory")

            # Get all sample directories
            sample_dirs = [
                d
                for d in os.listdir(source_dir)
                if os.path.isdir(os.path.join(source_dir, d))
            ]

            for sample_dir in sample_dirs:
                source_sample_dir = os.path.join(source_dir, sample_dir)
                target_sample_dir = os.path.join(
                    static_gallery_dir,
                    sample_dir,
                    source_dir_name.lower().replace(" ", "_"),
                )

                # Create target directory
                os.makedirs(target_sample_dir, exist_ok=True)

                # Copy video files to target directory
                video_file = os.path.join(source_sample_dir, "output.mp4")
                if os.path.exists(video_file):
                    target_video = os.path.join(target_sample_dir, "output.mp4")

                    # Only copy if target doesn't exist or source is newer
                    if not os.path.exists(target_video) or os.path.getmtime(
                        video_file
                    ) > os.path.getmtime(target_video):
                        print(f"Copying {video_file} to {target_video}")
                        try:
                            shutil.copy2(video_file, target_video)
                            print(
                                f"Successfully copied video file: {os.path.exists(target_video)}"
                            )
                        except Exception as e:
                            print(f"Error copying video file: {str(e)}")
                else:
                    print(f"Source video file not found: {video_file}")
        else:
            print(f"Source directory not found: {source_dir}")


if __name__ == "__main__":
    print("Starting video processing...")

    # Process ground truth frames to create videos
    process_ground_truth_frames()

    # Ensure all video formats are available
    ensure_all_video_formats()

    # Link or copy gallery directories to static directory for web serving
    link_static_directories()

    print("Video processing completed!")
