import os
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from tqdm import tqdm


def extract_frames(video_path, output_folder, frame_interval=30):
    try:
        # Load the video file
        video = VideoFileClip(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate over all frames with specified interval
        for i, frame in tqdm(enumerate(video.iter_frames())):
            if i % frame_interval == 0:
                # Convert NumPy array to Image object
                image = Image.fromarray(frame)

                # Save frame as PNG file
                frame_path = os.path.join(output_folder, f"{video_name}_{i:04d}.png")
                image.save(frame_path)

        # Close the video file
        video.close()

    except Exception as e:
        print(f"Error extracting frames: {str(e)}")


if __name__ == "__main__":
    video_folder = "video"
    output_folder = "video/save/"
    for filename in tqdm(os.listdir(video_folder)):
        if filename.endswith(".mp4"):
            video_file = f"video/{filename}"
            extract_frames(video_file, output_folder, frame_interval=30)