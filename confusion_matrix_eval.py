import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import backend as K

# ---------------- CONFIG ----------------
IMAGE_SIZE = 512
DATA_ROOT = os.getcwd()
IMAGES_DIR = os.path.join(DATA_ROOT, "images_cut")
MASKS_DIR  = os.path.join(DATA_ROOT, "labels_cut")
MODEL_PATH = os.path.join(DATA_ROOT, "best_model.keras")
THRESHOLD = 0.5
# -----------------------------------------

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
    return bce + (1 - dice)

print("\nLoading model:", MODEL_PATH)
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"dice_coef": dice_coef, "bce_dice_loss": bce_dice_loss}
)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    norm = gray.astype(np.float32) / 255.0
    stacked = np.stack([norm, norm, norm], axis=-1)
    return stacked

y_true_pixels = []
y_pred_pixels = []

image_files = sorted(os.listdir(IMAGES_DIR))

print("\nEvaluating pixel-level confusion matrix...\n")

for fname in image_files:
    img_path = os.path.join(IMAGES_DIR, fname)
    mask_path = os.path.join(MASKS_DIR, fname)

    if not os.path.exists(mask_path):
        print("Missing mask for:", fname)
        continue

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print("Error loading:", fname)
        continue

    gt = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    gt_bin = (gt > 128).astype(np.uint8).flatten()

    inp = np.expand_dims(preprocess(img), axis=0)
    pred = model.predict(inp)[0, ..., 0]
    pred_bin = (pred > THRESHOLD).astype(np.uint8).flatten()

    y_true_pixels.extend(gt_bin)
    y_pred_pixels.extend(pred_bin)

cm = confusion_matrix(y_true_pixels, y_pred_pixels)
report = classification_report(y_true_pixels, y_pred_pixels, target_names=["No Cavity", "Cavity"])

print("\n================ CONFUSION MATRIX ================")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\n================ CLASSIFICATION REPORT ================")
print(report)
