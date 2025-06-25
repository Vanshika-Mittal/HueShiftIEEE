# HueShift Video Standardization Tools

This directory contains scripts to standardize and improve the video playback experience in the HueShift Gallery.

## Tools Overview

1. **create_standard_videos.py** - Creates standardized videos from frames in the Gallery directory
2. **update_gallery_templates.py** - Updates app.py and gallery.html to use the standardized videos
3. **run_standardization.py** - Main script to run the entire standardization process

## Synchronization JS

The `sync_videos.js` file in the `/static/js/` directory handles synchronized video playback across all videos in the gallery, ensuring:

- All videos play/pause simultaneously
- Videos maintain the same timeline position
- Keyboard controls for playback (space to play/pause, arrows to seek, R to restart)
- Fixed timing issues with video loading and playback

## Running the Standardization Process

To standardize all videos in the Gallery:

```bash
python tools/run_standardization.py
```

To standardize a specific sample:

```bash
python tools/run_standardization.py --sample v_ApplyLipstick_g05_c04
```

To update only the templates without recreating videos:

```bash
python tools/run_standardization.py --skip-video-creation
```

## Features

The standardization process:

1. Creates uniform videos with consistent framerate (10 FPS by default)
2. Ensures all videos have the same length and timing
3. Converts ground truth frames to grayscale for better comparison
4. Uses standardized naming conventions
5. Improves video playback in the browser
6. Ensures synchronized playback across all videos

## Troubleshooting

If videos aren't playing synchronously:
- Check browser console for errors
- Make sure all videos have the same number of frames
- Verify that all videos have the `.sync-video` CSS class 