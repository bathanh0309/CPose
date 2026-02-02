import cv2
import os

def compress_video(input_path, output_path, target_size_mb=40):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    cap = cv2.VideoCapture(input_path)
    
    # Get original properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: Empty video")
        return

    print(f"Original: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Resize to 540p for better compression
    target_height = 540
    if height > target_height:
        scale = target_height / height
        new_width = int(width * scale)
        new_height = target_height
    else:
        new_width = width
        new_height = height
        
    print(f"Target: {new_width}x{new_height}")

    # Calculate bitrate
    duration = total_frames / fps
    # target_size_bits = target_size_mb * 8 * 1024 * 1024
    # bitrate = target_size_bits / duration
    # Since OpenCV VideoWriter doesn't easily allow setting strict bitrate without ffmpeg, 
    # we rely on resizing and reducing FPS if needed.
    # Let's reduce FPS to 20 if it's 30+ to save space
    
    target_fps = min(fps, 24)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))
    
    count = 0
    frame_interval = int(fps / target_fps) if fps > target_fps else 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            resized = cv2.resize(frame, (new_width, new_height))
            out.write(resized)
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total_frames} frames", end='\r')

    cap.release()
    out.release()
    print("\nCompression complete!")
    
    # Check size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"New size: {size_mb:.2f} MB")

if __name__ == "__main__":
    compress_video("D:/HAVEN/output.mp4", "D:/HAVEN/demo.mp4")
