# train_from_archive_fixed.py
"""
Robust panoramic dental caries segmentation pipeline.
Auto-detects dataset folders and converts COCO -> masks if needed.
Flexible mask matching (exact basename, mask_/label_ prefixes, substring).
Trains a U-Net with EfficientNetB0 encoder (if available) and provides inference visualization.

Place this script in the dataset folder (the folder that contains images/, labels/, annotations/, images_cut/, labels_cut/).
Run: python train_from_archive_fixed.py
"""

import os
import glob
import json
import random
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ---------------- CONFIG (you probably won't need to edit) ----------------
DATA_ROOT = os.getcwd()   # default: current working directory (the folder you run the script from)
IMAGE_SIZE = 512          # reduce to 256 if OOM
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
SEED = 42
# ------------------------------------------------------------------------

print("Running script from:", os.getcwd())
print("DATA_ROOT =", DATA_ROOT)

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------- utility: robust imread / imwrite for Unicode / Windows paths ----------
def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print("imread_unicode error:", e)
        return None

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    if ext == '':
        ext = '.png'
        path += ext
    _, enc = cv2.imencode(ext, img)
    enc.tofile(path)

# ---------- dataset folder detection ----------
images_cut_dir = os.path.join(DATA_ROOT, 'images_cut')
labels_cut_dir = os.path.join(DATA_ROOT, 'labels_cut')
images_dir = os.path.join(DATA_ROOT, 'images')
labels_dir = os.path.join(DATA_ROOT, 'labels')
annotations_dir = os.path.join(DATA_ROOT, 'annotations')

use_patch_mode = False
if os.path.isdir(images_cut_dir) and os.path.isdir(labels_cut_dir):
    if len(glob.glob(os.path.join(images_cut_dir, '*'))) > 0 and len(glob.glob(os.path.join(labels_cut_dir, '*'))) > 0:
        use_patch_mode = True

if use_patch_mode:
    print(">>> Using PATCH mode (images_cut/ + labels_cut/)")
    IMAGES_DIR = images_cut_dir
    MASKS_DIR = labels_cut_dir
else:
    print(">>> Using FULL-IMAGE mode (images/ + labels/ or annotations/)")
    IMAGES_DIR = images_dir
    MASKS_DIR = labels_dir

print("Images dir:", IMAGES_DIR)
print("Masks  dir:", MASKS_DIR)
print("Annotations dir:", annotations_dir)

# ---------- helper: sample file listing ----------
def list_sample_files(dirpath, n=5, exts=('png','jpg','jpeg','tif','tiff','bmp')):
    if not os.path.isdir(dirpath):
        return [], 0
    files = []
    for e in exts:
        files += glob.glob(os.path.join(dirpath, f'*.{e}'))
    files = sorted(files)
    return files[:n], len(files)

samp_imgs, n_imgs = list_sample_files(IMAGES_DIR)
samp_masks, n_masks = list_sample_files(MASKS_DIR)
print(f"Found {n_imgs} image files (sample): {samp_imgs}")
print(f"Found {n_masks} mask files  (sample): {samp_masks}")

# ---------- COCO -> masks conversion (polygons; RLE if pycocotools installed) ----------
def convert_coco_to_masks(coco_json_path, images_dir, masks_out_dir, target_size=None):
    os.makedirs(masks_out_dir, exist_ok=True)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    images_info = {img['id']: img for img in coco.get('images', [])}
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    # pycocotools optional (for RLE)
    has_pycoco = False
    try:
        from pycocotools import mask as maskUtils
        has_pycoco = True
        print("pycocotools found: RLE masks will be handled.")
    except Exception:
        print("pycocotools not found. RLE will be skipped if present. Install pycocotools to enable RLE support.")

    for img_id, img_info in images_info.items():
        fname = img_info.get('file_name')
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            print("Skipping: image missing ->", img_path)
            continue
        # read image to get shape (if not present in JSON)
        try:
            tmp = imread_unicode(img_path)
            if tmp is None:
                print("Could not read", img_path); continue
            h, w = tmp.shape[:2]
        except Exception as e:
            print("read error", e); continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_by_image.get(img_id, []):
            seg = ann.get('segmentation')
            if not seg:
                continue
            if isinstance(seg, list):
                # polygon(s)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1,2)
                    cv2.fillPoly(mask, [pts], 255)
            else:
                # RLE or compressed RLE
                if has_pycoco:
                    try:
                        m = maskUtils.decode(seg)
                        if m.ndim == 3:
                            m = np.any(m, axis=2).astype(np.uint8)
                        mask = np.maximum(mask, (m*255).astype(np.uint8))
                    except Exception as e:
                        print("pycocotools decode error:", e)
                else:
                    print("RLE segmentation present but pycocotools not installed; skipping RLE for this annotation.")
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(masks_out_dir, out_name)
        if target_size:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        imwrite_unicode(out_path, mask)
    print("COCO -> masks conversion complete. Masks saved to:", masks_out_dir)

# If masks missing and annotations present -> attempt conversion
if not use_patch_mode:
    masks_exist = os.path.isdir(MASKS_DIR) and len(glob.glob(os.path.join(MASKS_DIR, '*'))) > 0
    ann_jsons = glob.glob(os.path.join(annotations_dir, '*.json')) if os.path.isdir(annotations_dir) else []
    if (not masks_exist) and len(ann_jsons) > 0:
        print("No masks found in labels/. Converting COCO JSON -> masks using:", ann_jsons[0])
        convert_coco_to_masks(ann_jsons[0], IMAGES_DIR, MASKS_DIR, target_size=None)
    elif not masks_exist and len(ann_jsons) == 0:
        print("Warning: No masks found and no annotation JSON found. You'll need masks or annotations to train.")
    else:
        print("Masks folder found and non-empty; skipping conversion.")

# ---------- flexible matching: find mask for each image ----------
def build_mask_index(mask_dir, exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
    idx = {}  # maps basename_lower -> fullpath (if multiple, keep first)
    all_masks = []
    for ext in exts:
        all_masks += glob.glob(os.path.join(mask_dir, f'*{ext}'))
    for m in all_masks:
        base = os.path.splitext(os.path.basename(m))[0].lower()
        idx.setdefault(base, []).append(m)
    return idx, all_masks

def find_best_mask_for_image(image_path, mask_index, all_masks):
    """
    Try a list of heuristics:
      1) exact basename match
      2) mask_*, label_* prefixes
      3) basename substring match in any mask filename
    """
    base = os.path.splitext(os.path.basename(image_path))[0].lower()
    # 1) exact
    if base in mask_index:
        return mask_index[base][0]
    # 2) prefixes
    for pref in ('mask_'+base, 'label_'+base, 'lbl_'+base):
        if pref in mask_index:
            return mask_index[pref][0]
    # 3) substring match (mask filename contains base)
    for m in all_masks:
        if base in os.path.basename(m).lower():
            return m
    return None

def get_image_mask_pairs(images_dir, masks_dir):
    exts_img = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
    image_files = []
    for e in exts_img:
        image_files += glob.glob(os.path.join(images_dir, f'*{e}'))
    image_files = sorted(image_files)
    mask_index, all_masks = build_mask_index(masks_dir)
    pairs = []
    missing = []
    for img in image_files:
        best = find_best_mask_for_image(img, mask_index, all_masks)
        if best:
            pairs.append((img, best))
        else:
            missing.append(os.path.splitext(os.path.basename(img))[0])
    print(f"Scanned {len(image_files)} images; found {len(pairs)} matching masks; missing masks for {len(missing)} images.")
    if len(missing) > 0:
        print("Examples of images missing masks (basenames):", missing[:10])
    return pairs

# ---------- pre-processing functions ----------
def preprocess_opencv(img, target_size=IMAGE_SIZE):
    if img is None:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    norm = enhanced.astype(np.float32) / 255.0
    resized = cv2.resize(norm, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=-1)

def preprocess_mask(mask_img, target_size=IMAGE_SIZE):
    if mask_img is None:
        return None
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img.copy()
    _, bin_mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
    resized = cv2.resize(bin_mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(resized.astype(np.float32), axis=-1)

# ---------- build tf.data dataset ----------
def tf_dataset_from_pairs(pairs, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    def gen():
        for img_p, mask_p in pairs:
            img = imread_unicode(img_p)
            mask = imread_unicode(mask_p, flags=cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue
            img_inp = preprocess_opencv(img)
            mask_inp = preprocess_mask(mask)
            if augment:
                # horizontal flip
                if random.random() > 0.5:
                    img_inp = np.flip(img_inp, axis=1).copy()
                    mask_inp = np.flip(mask_inp, axis=1).copy()
            yield img_inp.astype(np.float32), mask_inp.astype(np.float32)
    ds = tf.data.Dataset.from_generator(gen,
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=([IMAGE_SIZE, IMAGE_SIZE,1], [IMAGE_SIZE, IMAGE_SIZE,1]))
    if shuffle:
        ds = ds.shuffle(256, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------- model (EfficientNet encoder if possible, else simple UNet) ----------
def build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inp = layers.Input(shape=input_shape)
    # replicate channel to 3 for pretrained encoder
    x_in = layers.Concatenate()([inp, inp, inp])
    try:
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x_in)
        skip_names = ['block2a_activation','block3a_activation','block4a_activation','block6a_activation']
        skips = [base.get_layer(n).output for n in skip_names]
        x = base.output
        def up_block(x, skip, filters):
            x = layers.UpSampling2D((2,2))(x)
            x = layers.Concatenate()([x, skip])
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            return x
        d1 = up_block(x, skips[-1], 256)
        d2 = up_block(d1, skips[-2], 128)
        d3 = up_block(d2, skips[-3], 64)
        d4 = up_block(d3, skips[-4], 32)
        x = layers.UpSampling2D((2,2))(d4)
        x = layers.Conv2D(32,3,padding='same',activation='relu')(x)
        x = layers.Conv2D(16,3,padding='same',activation='relu')(x)
        out = layers.Conv2D(1,1,activation='sigmoid')(x)
        model = models.Model(inputs=inp, outputs=out)
        print("Built UNet with EfficientNetB0 encoder (ImageNet weights).")
        return model
    except Exception as e:
        print("EfficientNet unavailable or error:", e)
        print("Falling back to simple U-Net.")
        # simple U-Net
        def conv_block(x, f):
            x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
            x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
            return x
        c1 = conv_block(inp, 32); p1 = layers.MaxPooling2D()(c1)
        c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
        c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
        c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
        b = conv_block(p4, 512)
        def dec(x, skip, f):
            x = layers.UpSampling2D()(x)
            x = layers.Concatenate()([x, skip])
            return conv_block(x, f)
        d4 = dec(b, c4, 256)
        d3 = dec(d4, c3, 128)
        d2 = dec(d3, c2, 64)
        d1 = dec(d2, c1, 32)
        out = layers.Conv2D(1,1,activation='sigmoid')(d1)
        model = models.Model(inputs=inp, outputs=out)
        return model

# ---------- loss & metrics ----------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return 1 - (2.*inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2.*inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# ---------- prepare pairs ----------
pairs = get_image_mask_pairs(IMAGES_DIR, MASKS_DIR)
if len(pairs) < 4:
    raise RuntimeError("Not enough image-mask pairs found. See debug prints above. Pairs found: {}".format(len(pairs)))

random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
val_pairs = pairs[split:]
print(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

train_ds = tf_dataset_from_pairs(train_pairs, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_ds = tf_dataset_from_pairs(val_pairs, batch_size=BATCH_SIZE, shuffle=False, augment=False)

# ---------- build & compile model ----------
model = build_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=dice_loss, metrics=[dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)])
print(model.summary())

# ---------- callbacks & training ----------
checkpoint_path = os.path.join(DATA_ROOT, 'best_model.h5')
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', mode='max', patience=6, restore_best_weights=True, verbose=1)
]

print("Starting training for {} epochs...".format(EPOCHS))
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

print("Training finished. Best model saved to:", checkpoint_path)

# ---------- inference helper ----------
def predict_and_visualize(model_path, image_path, out_path, threshold=0.5, min_area=100):
    m = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    img = imread_unicode(image_path)
    if img is None:
        print("Failed to read", image_path); return
    orig = img.copy()
    h0,w0 = orig.shape[:2]
    inp = preprocess_opencv(img)
    pred = m.predict(np.expand_dims(inp, axis=0))[0,...,0]
    pred_resized = cv2.resize((pred*255).astype(np.uint8), (w0,h0), interpolation=cv2.INTER_LINEAR)
    _, pred_bin = cv2.threshold(pred_resized, int(threshold*255), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    pred_clean = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    pred_clean = cv2.morphologyEx(pred_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(pred_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = orig.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.drawContours(overlay, [c], -1, (0,255,0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0, vis)
    imwrite_unicode(out_path, vis)
    print("Saved visualization to", out_path)
    return pred_resized, pred_clean, vis

# Example usage: after training you can run:
# predict_and_visualize(checkpoint_path, os.path.join(IMAGES_DIR, 'some_image.png'), os.path.join(DATA_ROOT, 'vis.png'))
