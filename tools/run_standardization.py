#!/usr/bin/env python
import os
import subprocess
import argparse
import sys


def get_base_dirs():
    """Get the base directories for the project"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    website_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(website_dir)

    return project_dir, website_dir


def run_command(cmd, cwd=None):
    """Run a shell command and print output"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the video standardization process"
    )
    parser.add_argument("--sample", help="Process only a specific sample")
    parser.add_argument(
        "--skip-video-creation",
        action="store_true",
        help="Skip the video creation step",
    )
    args = parser.parse_args()

    project_dir, website_dir = get_base_dirs()

    print(f"Project directory: {project_dir}")
    print(f"Website directory: {website_dir}")

    # Step 1: Create standardized videos
    if not args.skip_video_creation:
        print("\n=== Step 1: Creating standardized videos ===")
        cmd = [
            sys.executable,
            os.path.join(website_dir, "tools", "create_standard_videos.py"),
        ]

        if args.sample:
            cmd.extend(["--sample", args.sample])

        if not run_command(cmd, cwd=website_dir):
            print("Error creating standardized videos. Aborting.")
            return False

    # Step 2: Update templates to use the standardized videos
    print("\n=== Step 2: Updating templates ===")
    cmd = [
        sys.executable,
        os.path.join(website_dir, "tools", "update_gallery_templates.py"),
    ]

    if args.sample:
        cmd.extend(["--sample", args.sample])

    if not run_command(cmd, cwd=website_dir):
        print("Error updating templates. Aborting.")
        return False

    print("\n=== Standardization complete! ===")
    print("You can now run the web server to see the changes.")
    print(f"cd {website_dir} && python app.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
