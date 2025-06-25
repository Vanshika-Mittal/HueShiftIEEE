import os
import re
from pathlib import Path
import argparse


def get_base_dirs():
    """Get the base directories for the project"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    website_dir = os.path.dirname(script_dir)

    return website_dir


def update_app_py(website_dir, sample_name=None):
    """Update app.py to serve standardized videos"""
    app_path = os.path.join(website_dir, "app.py")
    print(f"Updating {app_path}")

    with open(app_path, "r") as f:
        app_content = f.read()

    # Update the serve_gallery function to use the standardized videos
    # Original pattern for video URLs
    pattern = r'filename=f"gallery/diffusion/{sample_name}/([^/]+)/output.mp4"'

    # New pattern using standardized videos
    if sample_name:
        # For a specific sample
        replacement = f'filename=f"gallery/diffusion/{sample_name}/\\1/{sample_name}_standardized.mp4"'
        app_content = re.sub(
            pattern.replace("{sample_name}", sample_name), replacement, app_content
        )
    else:
        # For all samples
        replacement = r'filename=f"gallery/diffusion/{sample_name}/\1/{sample_name}_standardized.mp4"'
        app_content = re.sub(pattern, replacement, app_content)

    # Also update the serve_gallery route to look for new file names
    serve_gallery_pattern = r'# Map the URL path to the actual file path in the Gallery folder(.+?)return "File not found", 404'

    def serve_gallery_replacement(match):
        # Keep the original content but add logic to check for standardized files
        original_content = match.group(1)

        # Replace output.mp4 with sample_name_standardized.mp4 in file paths
        modified_content = re.sub(
            r'(os\.path\.join\([^)]+\), ")output.mp4(")',
            r"\1{sample_name}_standardized.mp4\2",
            original_content,
        )

        # Replace hardcoded instances of output.mp4
        modified_content = re.sub(
            r'"output.mp4"', r'f"{sample_name}_standardized.mp4"', modified_content
        )

        return f'# Map the URL path to the actual file path in the Gallery folder{modified_content}return "File not found", 404'

    app_content = re.sub(
        serve_gallery_pattern, serve_gallery_replacement, app_content, flags=re.DOTALL
    )

    # Update handling for stage_1 videos
    stage1_pattern = r'(f"(?:global_info|reconstruction)_)({sample_name})(\.mp4")'
    if sample_name:
        app_content = re.sub(
            stage1_pattern.replace("{sample_name}", sample_name),
            r"\1\2_standardized\3",
            app_content,
        )
    else:
        app_content = re.sub(
            stage1_pattern, r"\1{sample_name}_standardized\3", app_content
        )

    # Write the updated content back
    with open(app_path, "w") as f:
        f.write(app_content)

    print(f"Updated app.py to use standardized videos")


def update_gallery_html(website_dir):
    """Update gallery.html to use standardized videos"""
    gallery_path = os.path.join(website_dir, "templates", "gallery.html")
    print(f"Updating {gallery_path}")

    with open(gallery_path, "r") as f:
        gallery_content = f.read()

    # Update the video playback parameters
    # Remove autoplay attribute as it doesn't work consistently
    gallery_content = re.sub(
        r'<video class="sync-video"([^>]*) autoplay([^>]*)>',
        r'<video class="sync-video"\1\2>',
        gallery_content,
    )

    # Ensure all videos have consistent attributes
    gallery_content = re.sub(
        r"<video ([^>]*)>", r"<video \1 playsinline>", gallery_content
    )

    # Replace duplicate playsinline attributes if any
    gallery_content = gallery_content.replace("playsinline playsinline", "playsinline")

    # Add a note about standardized videos
    note = """
    <div class="alert alert-info mt-3 mb-3">
        <strong>Note:</strong> All videos have been standardized to ensure consistent framerate and synchronization.
    </div>
    """

    # Insert the note after the gallery heading
    gallery_content = re.sub(
        r'(<h1 class="text-center mb-4">HueShift Gallery</h1>)',
        r"\1" + note,
        gallery_content,
    )

    # Write the updated content back
    with open(gallery_path, "w") as f:
        f.write(gallery_content)

    print(f"Updated gallery.html for improved video playback")


def main():
    parser = argparse.ArgumentParser(
        description="Update app.py and gallery.html to use standardized videos"
    )
    parser.add_argument("--sample", help="Update only a specific sample")
    args = parser.parse_args()

    website_dir = get_base_dirs()

    # Update app.py
    update_app_py(website_dir, args.sample)

    # Update gallery.html
    update_gallery_html(website_dir)

    print("\nCompleted updating templates for standardized videos")


if __name__ == "__main__":
    main()
