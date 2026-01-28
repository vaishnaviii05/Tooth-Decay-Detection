# visualize_model_xai.py
"""
Visualize how the cavity detection model works, with XAI (Grad-CAM style).

Shows, for random patch images:
  - BEFORE: Original X-ray patch
  - BEFORE: Preprocessed (CLAHE + resize) input to the model
  - MODEL OUTPUT: Predicted cavity probability map
  - XAI: Grad-CAM heatmap overlay (where the model is "looking")
  - AFTER: Final mask + filtered bounding boxes overlay on the X-ray

The post-processing now:
  - Filters out tiny and huge blobs
  - Rejects contours likely outside tooth regions
  - Labels each box as "Cavity"

Run from the dataset folder:
  python visualize_model_xai.py
"""

import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

# ----------------- CONFIG (match your training setup) -----------------
DATA_ROOT = os.getcwd()
IMAGE_SIZE = 512   # must match train_from_archive_updated.py
IMAGES_DIR = os.path.join(DATA_ROOT, "images_cut")  # patch mode folder
MODEL_PATH = os.path.join(DATA_ROOT, "best_model.keras")

THRESHOLD = 0.5
MIN_AREA = 80          # minimum contour area for a bounding box (in pixels)
MAX_AREA_FRAC = 0.20   # maximum area as fraction of total image area (e.g. 0.2 = 20%)
MIN_MEAN_INTENSITY = 35  # ignore very dark regions (likely background)
# ----------------------------------------------------------------------


# ---------- utility: robust imread ----------
def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print("imread error:", e)
        return None


# ---------- preprocessing: same as training ----------
def preprocess_image_rgb(img, target_size=IMAGE_SIZE):
    """
    Convert original X-ray patch to the exact format used during training:
    - grayscale
    - CLAHE enhancement
    - normalized to [0,1]
    - resized to target_size x target_size
    - stacked to 3 channels
    """
    if img is None:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    norm = enhanced.astype(np.float32) / 255.0
    resized = cv2.resize(norm, (target_size, target_size), interpolation=cv2.INTER_AREA)
    stacked = np.stack([resized, resized, resized], axis=-1)
    return stacked


# ---------- custom loss & metric (same as training) ----------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    dice = (2. * inter + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)
    return bce + (1.0 - dice)


# ---------- load trained model ----------
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef}
)
print("Loaded model from:", MODEL_PATH)


# ---------- choose last Conv2D layer for Grad-CAM (fixed) ----------
def get_last_conv_layer(m):
    """
    Find the last Conv2D layer to use for Grad-CAM.
    Prefer a Conv2D with filters > 1 (not the final 1-channel output).
    Uses layer.filters instead of layer.output_shape.
    """
    conv_layers = [layer for layer in m.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No Conv2D layers found in model.")

    chosen = None
    # go backwards and pick last Conv2D with filters>1 if possible
    for layer in reversed(conv_layers):
        filters = getattr(layer, "filters", None)
        if filters is not None and filters > 1:
            chosen = layer
            break

    # if all convs had 1 filter (unlikely), just take the last conv layer
    if chosen is None:
        chosen = conv_layers[-1]

    print("Using Grad-CAM layer:", chosen.name, "filters:", getattr(chosen, "filters", "N/A"))
    return chosen


last_conv_layer = get_last_conv_layer(model)


# ---------- Grad-CAM for segmentation ----------
def compute_gradcam_heatmap(model, last_conv_layer, img_tensor):
    """
    img_tensor: (1, H, W, 3), float32, preprocessed
    We define the target as the mean of the output mask:
       loss = mean(predictions)
    Then Grad-CAM over last_conv_layer activations.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)   # conv_outputs: (1,h,w,c), predictions: (1,H,W,1)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)  # (1,h,w,c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)

    conv_outputs = conv_outputs[0]  # (h,w,c)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (h,w)

    heatmap = tf.maximum(heatmap, 0)  # ReLU
    max_val = tf.reduce_max(heatmap) + 1e-8
    heatmap /= max_val

    return heatmap.numpy()  # (h,w) in [0,1]


# ---------- helper: filter contours based on area + intensity ----------
def filter_contours(contours, orig_gray, mask_clean):
    """
    Filters out:
      - very small contours
      - very large contours (covering almost full patch)
      - dark regions (likely background / non-tooth)
    """
    h0, w0 = orig_gray.shape
    max_area = MAX_AREA_FRAC * h0 * w0

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > max_area:
            continue

        # Create a temporary mask for this contour to compute mean intensity
        temp_mask = np.zeros_like(mask_clean, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=-1)
        mean_intensity = cv2.mean(orig_gray, mask=temp_mask)[0]

        if mean_intensity < MIN_MEAN_INTENSITY:
            # Too dark, likely non-tooth background
            continue

        valid_contours.append(cnt)

    return valid_contours


# ---------- main prediction + XAI pipeline ----------
def predict_single_image_with_xai(img_path, threshold=THRESHOLD):
    """
    Returns:
      orig_gray   - original grayscale patch
      model_input - preprocessed image fed to the model (CLAHE + resize)
      prob_map    - predicted cavity probability map (H_model, W_model)
      mask_clean  - final binary mask (resized to original patch size)
      blended     - original + bounding boxes + filled contours
      xai_overlay - original + Grad-CAM heatmap overlay (masked by cavity)
    """
    # 1) load original
    orig = imread_unicode(img_path)
    if orig is None:
        raise FileNotFoundError(img_path)

    if len(orig.shape) == 2:
        orig_gray = orig.copy()
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = orig.copy()
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    h0, w0 = orig_gray.shape

    # 2) preprocess to model input
    model_input = preprocess_image_rgb(orig)  # (IMAGE_SIZE, IMAGE_SIZE, 3)
    inp = np.expand_dims(model_input, axis=0)  # (1,H,W,3)

    # 3) model prediction
    pred = model.predict(inp)[0, ..., 0]  # (H,W), float [0,1]

    # 4) Grad-CAM heatmap
    heatmap_small = compute_gradcam_heatmap(model, last_conv_layer, inp)
    heatmap_512 = cv2.resize(heatmap_small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    heatmap_orig = cv2.resize(heatmap_512, (w0, h0), interpolation=cv2.INTER_LINEAR)

    # 5) convert prediction to mask at original resolution
    prob_resized = cv2.resize((pred * 255).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_LINEAR)
    _, mask_bin = cv2.threshold(prob_resized, int(threshold * 255), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) contours + filtered bounding boxes on mask_clean
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours(contours, orig_gray, mask_clean)

    boxes_img = orig_bgr.copy()
    overlay_mask = orig_bgr.copy()

    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # red bounding box
        cv2.rectangle(boxes_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Add label "Cavity"
        cv2.putText(
            boxes_img,
            "Cavity",
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        # green filled contour
        cv2.drawContours(overlay_mask, [cnt], -1, (0, 255, 0), -1)

    blended = cv2.addWeighted(overlay_mask, 0.4, boxes_img, 0.6, 0)

    # 7) XAI overlay using heatmap_orig, masked by predicted cavities
    #    (so heatmap focuses on predicted cavity zones only)
    heatmap_norm = heatmap_orig.copy()

    if np.any(mask_clean > 0):
        # Restrict Grad-CAM intensity to predicted cavity areas
        cavity_mask_float = (mask_clean > 0).astype(np.float32)
        heatmap_norm = heatmap_norm * cavity_mask_float

    heatmap_norm = np.uint8(255 * heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)  # BGR
    xai_overlay = cv2.addWeighted(heatmap_color, 0.4, orig_bgr, 0.6, 0)

    return orig_gray, model_input, pred, mask_clean, blended, xai_overlay


# ---------- display several examples ----------
def show_examples(num_examples=3, threshold=THRESHOLD):
    img_paths = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if len(img_paths) == 0:
        raise RuntimeError(f"No PNG images found in {IMAGES_DIR}")
    chosen = random.sample(img_paths, min(num_examples, len(img_paths)))

    for path in chosen:
        print("\n=== Example:", os.path.basename(path), "===")
        orig_gray, model_inp, prob_map, mask_clean, blended, xai_overlay = \
            predict_single_image_with_xai(path, threshold=threshold)

        plt.figure(figsize=(16, 8))

        # BEFORE: Original patch
        plt.subplot(2, 3, 1)
        plt.imshow(orig_gray, cmap="gray")
        plt.title("BEFORE: Original patch")
        plt.axis("off")

        # BEFORE: Preprocessed input
        plt.subplot(2, 3, 2)
        plt.imshow(model_inp[..., 0], cmap="gray")
        plt.title("BEFORE: Preprocessed\n(CLAHE + resize)")
        plt.axis("off")

        # MODEL OUTPUT: probability map
        plt.subplot(2, 3, 3)
        plt.imshow(prob_map, cmap="jet")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("MODEL OUTPUT:\nCavity probability map")
        plt.axis("off")

        # XAI: Grad-CAM heatmap overlay
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(xai_overlay, cv2.COLOR_BGR2RGB))
        plt.title("XAI: Grad-CAM\n(focused on predicted cavities)")
        plt.axis("off")

        # AFTER: final binary mask
        plt.subplot(2, 3, 5)
        plt.imshow(mask_clean, cmap="gray")
        plt.title("AFTER: Final cavity mask")
        plt.axis("off")

        # AFTER: bounding boxes + contour overlay
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title("AFTER: Cavity boxes\n+ contour overlay")
        plt.axis("off")

        plt.suptitle(os.path.basename(path))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Images directory:", IMAGES_DIR)
    print("Model path:", MODEL_PATH)
    show_examples(num_examples=3, threshold=THRESHOLD)
