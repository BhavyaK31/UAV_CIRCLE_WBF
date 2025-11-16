import cv2
import numpy as np
import random
import os

# =============================================================
# 1. MOTION BLUR
# =============================================================
def apply_motion_blur(frame, kernel_size=15, angle=0):
    if kernel_size < 3:
        return frame
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(frame, -1, kernel)

# =============================================================
# 2. GAUSSIAN BLUR
# =============================================================
def apply_gaussian_blur(frame, ksize=5):
    ksize = max(3, ksize // 2 * 2 + 1)  # ensure odd kernel size
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

# =============================================================
# 3. BRIGHTNESS & CONTRAST VARIATION
# =============================================================
def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    beta = brightness  # Brightness offset (-40 to +40)
    alpha = 1 + (contrast / 100.0)  # Contrast multiplier (0.6x to 1.4x)
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# =============================================================
# MAIN FUNCTION
# =============================================================
def degrade_video(input_path, output_path,
                  motion_blur=True, gaussian_blur=True, lighting=True):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"🎬 {os.path.basename(input_path)} - {frame_count} frames")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply motion blur
        if motion_blur and random.random() < 0.9:
            angle = random.uniform(0, 360)
            frame = apply_motion_blur(frame, kernel_size=random.randint(5, 15), angle=angle)

        # Apply gaussian blur
        if gaussian_blur and random.random() < 0.7:
            frame = apply_gaussian_blur(frame, ksize=random.randint(3, 7))

        # Apply brightness/contrast variation
        if lighting and random.random() < 0.8:
            frame = adjust_brightness_contrast(
                frame,
                brightness=random.randint(-40, 40),
                contrast=random.randint(-40, 40)
            )

        out.write(frame)
        frame_idx += 1

        # Print progress every 100 frames
        if frame_idx % 100 == 0:
            print(f"   → Processed {frame_idx}/{frame_count} frames", end="\r")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved blurred video: {output_path}\n")