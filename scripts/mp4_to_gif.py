import sys
import os
from pathlib import Path

# Attempt imports
try:
    # Try v2
    from moviepy import VideoFileClip
except ImportError:
    try:
        # Try v1
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
            if hasattr(clip, 'resize'):
                # moviepy 1.x
                print(f"Resizing to width={resize_width} (v1 style)...")
                clip = clip.resize(width=resize_width)
            elif hasattr(clip, 'resized'):
                # moviepy 2.x (some versions)
                print(f"Resizing to width={resize_width} (v2 .resized execution)...")
                clip = clip.resized(width=resize_width)
            else:
                print("Warning: .resize() method not found. Attempting skip.")
                # We could try importing vfx but paths vary. 
                # Proceeding with full size if resize fails is better than crashing.
        
        # Write GIF
        print("Writing GIF (this may take a while)...")
        clip.write_gif(str(output_path), fps=fps, logger='bar')
        print("\n" + "-" * 30)
        print("Success! GIF saved.")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Define paths relative to this script: D:/HavenNet/scripts/ -> D:/HavenNet/output.mp4
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    # Default input/output
    INPUT_VIDEO = PROJECT_ROOT / "output.mp4"
    OUTPUT_GIF = PROJECT_ROOT / "output.gif"

    convert_mp4_to_gif(INPUT_VIDEO, OUTPUT_GIF)
