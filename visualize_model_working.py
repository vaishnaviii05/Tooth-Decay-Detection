# visualize_model_working.py
import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Import from your training script
from train_from_archive_updated import (
    DATA_ROOT,
    IMAGES_DIR,
    preprocess_image_rgb,
    imread_unicode,
    bce_dice_loss,
    dice_coef
)

# Path to the trained model
MODEL_PATH = os.path.join(DATA_ROOT, "best_model.keras")

# Load model with custom loss + metric
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef}
)

print("Loaded model from:", MODEL_PATH)

def predict_single_image(img_path, threshold=0.5, min_area=100):
    """
    1) Load original X-ray patch
    2) Preprocess (CLAHE, resize, stack to 3 channels)
    3) Run model -> probability map
    4) Threshold + morphology
    5) Find contours + bounding boxes
    6) Return all intermediate results for visualization
    """
    # ----- step 1: load original image -----
    orig = imread_unicode(img_path)
    if orig is None:
        raise FileNotFoundError(img_path)
    if len(orig.shape) == 2:
        orig_gray = orig.copy()
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = orig.copy()
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    h0, w0 = orig_gray.shape[:2]

    # ----- step 2: preprocessing -----
    # This is exactly what you used during training
    model_input = preprocess_image_rgb(orig)  # shape (IMAGE_SIZE, IMAGE_SIZE, 3), float32 [0,1]
    print("Original shape:", orig_gray.shape, "-> model input shape:", model_input.shape)

    # ----- step 3: model prediction -----
    # Add batch dimension: (1, H, W, 3)
    prob = model.predict(np.expand_dims(model_input, axis=0))[0, ..., 0]  # shape (H,W)
    print("Raw prediction (prob map) shape:", prob.shape,
          "min:", prob.min(), "max:", prob.max(), "mean:", prob.mean())

    # ----- step 4: resize back to original + threshold + morphology -----
    prob_resized = cv2.resize((prob * 255).astype(np.uint8), (w0, h0),
                              interpolation=cv2.INTER_LINEAR)
    _, mask_bin = cv2.threshold(prob_resized,
                                int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ----- step 5: contours + bounding boxes -----
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # copy original for drawing
    overlay = orig_bgr.copy()
    vis = orig_bgr.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # ignore tiny blobs / noise

        x, y, w, h = cv2.boundingRect(cnt)
        # draw bounding box (in red)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # draw filled contour on overlay (in green)
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)

    # blend mask overlay with original
    blended = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    return orig_gray, model_input, prob, mask_clean, blended

def show_example(num_examples=3, threshold=0.5, min_area=100):
    # pick random sample images from patch folder
    img_paths = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if len(img_paths) == 0:
        raise RuntimeError(f"No .png files found in {IMAGES_DIR}")
    chosen = random.sample(img_paths, min(num_examples, len(img_paths)))

    for path in chosen:
        print("\n=== Example:", os.path.basename(path), "===")
        orig_gray, model_inp, prob_map, mask_clean, blended = predict_single_image(
            path, threshold=threshold, min_area=min_area
        )

        # ---- Plot the full story ----
        plt.figure(figsize=(14, 6))

        # original
        plt.subplot(1, 4, 1)
        plt.imshow(orig_gray, cmap="gray")
        plt.title("Original patch")
        plt.axis("off")

        # model input (CLAHE + resize)
        plt.subplot(1, 4, 2)
        plt.imshow(model_inp[..., 0], cmap="gray")
        plt.title("Preprocessed (to model)")
        plt.axis("off")

        # probability map
        plt.subplot(1, 4, 3)
        plt.imshow(prob_map, cmap="jet")
        plt.title("Predicted cavity\nprobability map")
        plt.axis("off")

        # overlay with boxes
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title("Bounding boxes & mask overlay")
        plt.axis("off")

        plt.suptitle(os.path.basename(path))
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Adjust threshold / min_area as needed
    show_example(num_examples=3, threshold=0.5, min_area=80)
