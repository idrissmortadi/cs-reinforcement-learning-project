import argparse
import math
import os

import cv2
import numpy as np


def video_to_grid(
    video_path, output_path, num_frames=16, spacing=5, font_scale=0.5, font_thickness=1
):
    """
    Reads a video, selects frames evenly spaced throughout, arranges them
    in a grid with spacing and frame numbers, and saves the grid as an image.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output grid image.
        num_frames (int): The total number of frames to include in the grid.
        spacing (int): Pixels of spacing between frames in the grid.
        font_scale (float): Font scale for the frame number text.
        font_thickness (int): Thickness of the frame number text.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or frame_width == 0 or frame_height == 0:
        print(f"Error: Video file {video_path} seems to be empty or corrupted.")
        cap.release()
        return

    # Ensure num_frames is not greater than total_frames
    num_frames = min(num_frames, total_frames)
    if num_frames <= 0:
        print("Error: Number of frames must be positive.")
        cap.release()
        return

    # Calculate frame indices to capture
    if num_frames == 1:
        indices = [total_frames // 2]  # Capture the middle frame if only one
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    frame_indices_captured = []  # Store the actual indices captured
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_indices_captured.append(i)  # Store successful index
        else:
            # If reading fails for some reason, try the previous frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_indices_captured.append(i - 1)  # Store successful index
            else:
                print(f"Warning: Could not read frame at or near index {i}")

    cap.release()

    if not frames:
        print("Error: No frames were captured.")
        return

    # Determine grid size (try to make it squarish)
    grid_cols = math.ceil(math.sqrt(len(frames)))
    grid_rows = math.ceil(len(frames) / grid_cols)

    # Calculate canvas size with spacing
    canvas_height = grid_rows * frame_height + (grid_rows + 1) * spacing
    canvas_width = grid_cols * frame_width + (grid_cols + 1) * spacing

    # Create blank canvas (e.g., white background)
    grid_image = np.full(
        (canvas_height, canvas_width, 3), 255, dtype=np.uint8
    )  # White background

    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 0)  # Black text

    # Place frames onto the grid
    for idx, frame in enumerate(frames):
        # Get the original frame index
        original_frame_index = frame_indices_captured[idx]

        # Add frame number text to the frame copy
        frame_copy = frame.copy()
        text = f"F: {original_frame_index}"
        # Get text size to position it
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        # Position text at bottom-left corner (with small padding)
        text_x = 5
        text_y = frame_height - 5
        # Add a small white background rectangle for better readability (optional)
        cv2.rectangle(
            frame_copy,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame_copy,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        row = idx // grid_cols
        col = idx % grid_cols

        # Calculate top-left corner coordinates with spacing
        y_start = spacing + row * (frame_height + spacing)
        x_start = spacing + col * (frame_width + spacing)

        y_end = y_start + frame_height
        x_end = x_start + frame_width

        # Place the frame with text onto the grid
        grid_image[y_start:y_end, x_start:x_end] = frame_copy

    # Save the grid image
    try:
        cv2.imwrite(output_path, grid_image)
        print(f"Grid image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert video sequence to a grid image."
    )
    parser.add_argument("video_path", help="Path to the input video file (e.g., .mp4)")
    parser.add_argument(
        "output_path", help="Path to save the output grid image (e.g., .png, .jpg)"
    )
    parser.add_argument(
        "-n",
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to include in the grid (default: 16)",
    )
    parser.add_argument(
        "-s",
        "--spacing",
        type=int,
        default=5,
        help="Spacing between frames in pixels (default: 5)",
    )
    # Add arguments for font scale and thickness if desired, or keep them fixed

    args = parser.parse_args()

    # Pass spacing to the function
    video_to_grid(args.video_path, args.output_path, args.num_frames, args.spacing)
