import cv2
import numpy as np

# Function to resize emoji to fit within a bounding box
def resize_emoji(emoji, box_width, box_height):
    return cv2.resize(emoji, (box_width, box_height))

# Function to overlay emoji on the frame
def overlay_emoji(frame, emoji, x, y, w, h):
    emoji_resized = resize_emoji(emoji, w, h)

    # Ensure alpha channel exists (transparency support)
    if emoji_resized.shape[2] != 4:
        b, g, r = cv2.split(emoji_resized)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        emoji_resized = cv2.merge((b, g, r, alpha))

    # Region of interest (ROI) in the frame
    roi = frame[y:y+h, x:x+w]

    # Blend the emoji and frame using the alpha channel
    alpha_emoji = emoji_resized[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_emoji

    for c in range(3):  # Loop through RGB channels
        roi[:, :, c] = (alpha_emoji * emoji_resized[:, :, c] + alpha_frame * roi[:, :, c])

    frame[y:y+h, x:x+w] = roi
    return frame
