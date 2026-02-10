import sys
import os
from pathlib import Path

try:
    # Try v2 import first
    from moviepy import VideoFileClip
except ImportError:
    # Fallback for v1
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("Error: moviepy library is not installed.")
        print("Please install it by running: pip install moviepy")
        sys.exit(1)

def convert_mp4_to_gif(input_path, output_path, resize_width=640, fps=10):
    """
    Converts an MP4 video to GIF.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Converting: {input_path}")
    print(f"Output:     {output_path}")
    print("-" * 30)

    try:
        # Load video clip
        clip = VideoFileClip(str(input_path))
        
        # Resize to reduce file size (optional)
        if resize_width:
            try:
                clip = clip.resize(width=resize_width)
            except AttributeError:
                print("Warning: .resize() method not found (moviepy 2.x). Skipping resize to ensure continued operation.")
                # Future improvement: implement vfx.resize for v2.x
        
        # Write GIF
        clip.write_gif(str(output_path), fps=fps)
        print("-" * 30)
        print("Success! GIF saved.")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    # Define paths relative to this script: D:/HavenNet/scripts/ -> D:/HavenNet/output.mp4
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    # Default input/output
    INPUT_VIDEO = PROJECT_ROOT / "output.mp4"
    OUTPUT_GIF = PROJECT_ROOT / "output.gif"

    convert_mp4_to_gif(INPUT_VIDEO, OUTPUT_GIF)
