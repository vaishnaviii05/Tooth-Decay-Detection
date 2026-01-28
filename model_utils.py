# model_utils.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

# --------- CONFIG (must match your training setup) ----------
IMAGE_SIZE = 512

# post-processing thresholds
THRESHOLD = 0.5            # probability threshold for mask
MIN_AREA = 80              # minimum contour area (small blobs ignored)
MAX_AREA_FRAC = 0.20       # max area as fraction of image (huge blobs ignored)
MIN_MEAN_INTENSITY = 35    # ignore very dark regions (non-tooth background)

# TOOTH REGION: we hard-limit all detections & heatmaps to this band
# (fractions of image height/width – tune if needed for your pano)
TOOTH_Y_MIN_FRAC = 0.38    # 38% from top
TOOTH_Y_MAX_FRAC = 0.78    # 78% from top
TOOTH_X_MIN_FRAC = 0.10    # 10% from left
TOOTH_X_MAX_FRAC = 0.90    # 90% from left

# Approximate dental arch teeth count for pano (you can mention 28 in report)
TEETH_TOTAL = 28
TEETH_COLUMNS = 14         # used to approximate "tooth index" along width

# Intensity threshold to classify very bright patch as RCT (root canal)
RCT_INTENSITY_THRESH = 180
# ------------------------------------------------------------


# --------- custom loss + metric (same as training) ----------
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


# --------- utility: imread with Unicode support ----------
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


# --------- preprocessing (same as training) ----------
def preprocess_image_rgb(img, target_size=IMAGE_SIZE):
    """
    Returns:
      stacked  - CLAHE-enhanced, normalized, resized, 3-channel image
      gray     - original grayscale image (for filtering / stats)
    """
    if img is None:
        return None, None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    norm = enhanced.astype(np.float32) / 255.0
    resized = cv2.resize(norm, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    stacked = np.stack([resized, resized, resized], axis=-1)
    return stacked, gray


# --------- Grad-CAM helpers ----------
def get_last_conv_layer(model):
    """
    Find the last Conv2D layer to use for Grad-CAM.
    Prefer Conv2D layers with filters > 1.
    """
    conv_layers = [layer for layer in model.layers
                   if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No Conv2D layers in model.")

    chosen = None
    for layer in reversed(conv_layers):
        filters = getattr(layer, "filters", None)
        if filters is not None and filters > 1:
            chosen = layer
            break

    if chosen is None:
        chosen = conv_layers[-1]

    print("Grad-CAM using layer:", chosen.name,
          "filters:", getattr(chosen, "filters", "N/A"))
    return chosen


def compute_gradcam_heatmap(model, last_conv_layer, img_tensor):
    """
    img_tensor: (1, H, W, 3), float32, preprocessed
    Returns a Grad-CAM heatmap in [0,1].
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (h,w,c)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap) + 1e-8
    heatmap /= max_val

    return heatmap.numpy()


# --------- load model once ----------
def load_model(model_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"bce_dice_loss": bce_dice_loss,
                        "dice_coef": dice_coef}
    )
    last_conv = get_last_conv_layer(model)
    return model, last_conv


# --------- build tooth ROI mask ----------
def build_tooth_roi_mask(shape_hw):
    """
    shape_hw: (h, w)
    Returns a binary mask (uint8) with 255 only in the
    approximate teeth band (central band of pano).
    """
    h0, w0 = shape_hw
    tooth_mask = np.zeros((h0, w0), dtype=np.uint8)

    y_min = int(TOOTH_Y_MIN_FRAC * h0)
    y_max = int(TOOTH_Y_MAX_FRAC * h0)
    x_min = int(TOOTH_X_MIN_FRAC * w0)
    x_max = int(TOOTH_X_MAX_FRAC * w0)

    tooth_mask[y_min:y_max, x_min:x_max] = 255
    return tooth_mask


# --------- filter contours (area + intensity + inside tooth ROI) ----------
def filter_contours(contours, orig_gray, mask_clean, tooth_roi_mask):
    """
    Filters out:
      - very small contours
      - very large contours (covering most of patch)
      - very dark regions (non-tooth background)
      - contours whose centroid lies outside tooth ROI
    """
    h0, w0 = orig_gray.shape
    max_area = MAX_AREA_FRAC * h0 * w0

    ys, xs = np.where(tooth_roi_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return []

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > max_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # centroid must lie inside tooth ROI bounds
        if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
            continue

        # mean intensity inside this contour
        temp_mask = np.zeros_like(mask_clean, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=-1)
        mean_intensity = cv2.mean(orig_gray, mask=temp_mask)[0]
        if mean_intensity < MIN_MEAN_INTENSITY:
            continue

        valid_contours.append(cnt)

    return valid_contours


# --------- main inference ----------
def run_inference_on_image(model, last_conv_layer, image_path,
                           out_dir, threshold=THRESHOLD):
    """
    1. Load image
    2. Preprocess
    3. Predict mask
    4. Hard-gate everything to tooth ROI:
         - mask outside tooth band set to 0
         - contours filtered by ROI + area + intensity
         - Grad-CAM also clipped to tooth ROI (and cavities if present)
    5. Draw bounding boxes labeled "Cavity detected" or "RCT"
    6. Return paths + summary stats for frontend report
    """

    os.makedirs(out_dir, exist_ok=True)

    orig = imread_unicode(image_path)
    if orig is None:
        raise FileNotFoundError(image_path)

    if len(orig.shape) == 2:
        orig_gray = orig.copy()
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = orig.copy()
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    h0, w0 = orig_gray.shape

    # build tooth ROI mask for this image size
    tooth_roi_mask = build_tooth_roi_mask((h0, w0))

    # preprocess
    model_input, _ = preprocess_image_rgb(orig)
    inp = np.expand_dims(model_input, axis=0)

    # prediction
    pred = model.predict(inp)[0, ..., 0]  # (H,W) in [0,1]

    # resize prediction to original size
    prob_resized = cv2.resize((pred * 255).astype(np.uint8), (w0, h0),
                              interpolation=cv2.INTER_LINEAR)
    _, mask_bin = cv2.threshold(prob_resized,
                                int(threshold * 255),
                                255,
                                cv2.THRESH_BINARY)

    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,
                                  kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE,
                                  kernel, iterations=1)

    # >>> HARD LIMIT: apply tooth ROI so no cavities outside teeth <<<
    mask_clean = cv2.bitwise_and(mask_clean, tooth_roi_mask)

    # contours and filtering
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered = filter_contours(contours, orig_gray, mask_clean, tooth_roi_mask)

    boxes_img = orig_bgr.copy()
    overlay_mask = orig_bgr.copy()

    # stats for report
    cavity_teeth = set()
    rct_teeth = set()
    col_width = w0 / TEETH_COLUMNS

    for cnt in filtered:
        x, y, w, h = cv2.boundingRect(cnt)

        # determine approximate tooth position by column
        tooth_id = int((x + w / 2) // col_width) + 1

        # per-contour mask for intensity
        mask_single = np.zeros_like(mask_clean, dtype=np.uint8)
        cv2.drawContours(mask_single, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(orig_gray, mask=mask_single)[0]

        if mean_intensity > RCT_INTENSITY_THRESH:
            rct_teeth.add(tooth_id)
            label_text = f"RCT (Tooth {tooth_id})"
            color_box = (0, 255, 255)  # yellow-ish for RCT
        else:
            cavity_teeth.add(tooth_id)
            label_text = f"Cavity (Tooth {tooth_id})"
            color_box = (0, 0, 255)    # red for cavity

        # rectangle & label on boxes_img
        cv2.rectangle(boxes_img, (x, y), (x + w, y + h), color_box, 2)
        cv2.putText(
            boxes_img,
            label_text,
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_box,
            1,
            cv2.LINE_AA
        )

        # green filled contour on overlay
        cv2.drawContours(overlay_mask, [cnt], -1, (0, 255, 0), -1)

    blended_boxes = cv2.addWeighted(overlay_mask, 0.4, boxes_img, 0.6, 0)

    # Grad-CAM
    heat_small = compute_gradcam_heatmap(model, last_conv_layer, inp)
    heat_resized = cv2.resize(heat_small, (w0, h0), interpolation=cv2.INTER_LINEAR)

    roi_float = (tooth_roi_mask > 0).astype(np.float32)
    cavity_float = (mask_clean > 0).astype(np.float32)

    # >>> HARD LIMIT: Grad-CAM only inside tooth ROI (and cavities if present) <<<
    if np.any(cavity_float > 0):
        heat_masked = heat_resized * roi_float * cavity_float
    else:
        heat_masked = heat_resized * roi_float

    heat_norm = np.uint8(255 * heat_masked)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    gradcam_overlay = cv2.addWeighted(heat_color, 0.4, orig_bgr, 0.6, 0)

    # ---- save images (Unicode-safe) ----
    orig_path = os.path.join(out_dir, "original.png")
    mask_path = os.path.join(out_dir, "mask.png")
    boxes_path = os.path.join(out_dir, "boxes.png")
    gradcam_path = os.path.join(out_dir, "gradcam.png")

    cv2.imencode(".png", orig_bgr)[1].tofile(orig_path)
    cv2.imencode(".png", mask_clean)[1].tofile(mask_path)
    cv2.imencode(".png", blended_boxes)[1].tofile(boxes_path)
    cv2.imencode(".png", gradcam_overlay)[1].tofile(gradcam_path)

    return {
        "original": orig_path,
        "mask": mask_path,
        "boxes": boxes_path,
        "gradcam": gradcam_path,
        "teeth_total": TEETH_TOTAL,
        "cavity_teeth": sorted(list(cavity_teeth)),
        "rct_teeth": sorted(list(rct_teeth)),
    }
