# train_from_archive_updated.py
"""
Updated panoramic dental caries segmentation pipeline.
- Uses 3-channel inputs so EfficientNetB0 pretrained weights load correctly.
- Combined BCE + Dice loss.
- train dataset .repeat() + explicit steps_per_epoch to avoid 'end of sequence' warnings.
- Auto-uses patch mode (images_cut/labels_cut) if present, else full-image mode.
- Place this script inside the dataset folder and run it.
"""

import os
import glob
import random
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ---------- CONFIG ----------
DATA_ROOT = os.getcwd()  # script run folder
IMAGE_SIZE = 512         # reduce to 256 if OOM
BATCH_SIZE = 4           # reduce if OOM; patches allow larger BATCH
EPOCHS = 30
LR = 1e-4
SEED = 42
# ----------------------------

print("DATA_ROOT =", DATA_ROOT)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------- utilities for Windows unicode paths ----------
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

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    if ext == '':
        ext = '.png'
        path += ext
    _, enc = cv2.imencode(ext, img)
    enc.tofile(path)

# ---------- dataset detection ----------
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
    IMAGES_DIR = images_cut_dir
    MASKS_DIR = labels_cut_dir
    print("Using PATCH mode: images_cut/ + labels_cut/")
else:
    IMAGES_DIR = images_dir
    MASKS_DIR = labels_dir
    print("Using FULL-IMAGE mode: images/ + labels/ (or annotations/)")

print("Images dir:", IMAGES_DIR)
print("Masks dir: ", MASKS_DIR)
print("Annotations dir:", annotations_dir)

# ---------- optional: COCO JSON -> masks conversion (polygon & RLE with pycocotools if installed) ----------
def convert_coco_to_masks(coco_json_path, images_dir, masks_out_dir, target_size=None):
    os.makedirs(masks_out_dir, exist_ok=True)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    images_info = {img['id']: img for img in coco.get('images', [])}
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)
    # try pycocotools
    try:
        from pycocotools import mask as maskUtils
        has_pycoco = True
        print("pycocotools available (RLE supported).")
    except Exception:
        has_pycoco = False
        print("pycocotools not installed (RLE will be skipped).")
    for img_id, img_info in images_info.items():
        fname = img_info['file_name']
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            continue
        tmp = imread_unicode(img_path)
        if tmp is None:
            continue
        h, w = tmp.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_by_image.get(img_id, []):
            seg = ann.get('segmentation')
            if not seg:
                continue
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 255)
            else:
                # RLE
                if has_pycoco:
                    try:
                        m = maskUtils.decode(seg)
                        if m.ndim == 3:
                            m = np.any(m, axis=2).astype(np.uint8)
                        mask = np.maximum(mask, (m*255).astype(np.uint8))
                    except Exception as e:
                        print("RLE decode error:", e)
                else:
                    print("Skipping RLE segmentation; install pycocotools to convert.")
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(masks_out_dir, out_name)
        if target_size:
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        imwrite_unicode(out_path, mask)
    print("Converted COCO JSON -> masks at", masks_out_dir)

# convert if masks missing and JSON present
masks_exist = os.path.isdir(MASKS_DIR) and len(glob.glob(os.path.join(MASKS_DIR, '*'))) > 0
jsons = glob.glob(os.path.join(annotations_dir, '*.json')) if os.path.isdir(annotations_dir) else []
if not masks_exist and len(jsons) > 0 and not use_patch_mode:
    print("No masks found; converting COCO JSON -> masks using:", jsons[0])
    convert_coco_to_masks(jsons[0], IMAGES_DIR, MASKS_DIR, target_size=None)

# ---------- helper to pair images and masks (flexible rules) ----------
def build_mask_index(mask_dir, exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
    idx = {}
    all_masks = []
    for ext in exts:
        all_masks += glob.glob(os.path.join(mask_dir, f'*{ext}'))
    for m in all_masks:
        base = os.path.splitext(os.path.basename(m))[0].lower()
        idx.setdefault(base, []).append(m)
    return idx, all_masks

def find_best_mask_for_image(image_path, mask_index, all_masks):
    base = os.path.splitext(os.path.basename(image_path))[0].lower()
    if base in mask_index:
        return mask_index[base][0]
    for pref in ('mask_'+base, 'label_'+base, 'lbl_'+base):
        if pref in mask_index:
            return mask_index[pref][0]
    # substring match
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
    print(f"Scanned {len(image_files)} images; found {len(pairs)} matching masks; missing for {len(missing)} images.")
    if len(missing)>0:
        print("Examples missing masks:", missing[:10])
    return pairs

# ---------- preprocessing (produce 3-channel images for encoder) ----------
def preprocess_image_rgb(img, target_size=IMAGE_SIZE):
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
    stacked = np.stack([resized, resized, resized], axis=-1)  # (H,W,3)
    return stacked

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

# ---------- tf.data builder ----------
def tf_dataset_from_pairs(pairs, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    def gen():
        for img_p, mask_p in pairs:
            img = imread_unicode(img_p)
            mask = imread_unicode(mask_p, flags=cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue
            img_inp = preprocess_image_rgb(img)
            mask_inp = preprocess_mask(mask)
            if augment:
                if random.random() > 0.5:
                    img_inp = np.flip(img_inp, axis=1).copy()
                    mask_inp = np.flip(mask_inp, axis=1).copy()
            yield img_inp.astype(np.float32), mask_inp.astype(np.float32)
    ds = tf.data.Dataset.from_generator(gen,
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=([IMAGE_SIZE, IMAGE_SIZE,3], [IMAGE_SIZE, IMAGE_SIZE,1]))
    if shuffle:
        ds = ds.shuffle(256, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------- model: EfficientNetB0 encoder UNet (fallback to simple UNet) ----------
def build_unet_efficientnet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    inp = layers.Input(shape=input_shape)
    try:
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inp)
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
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
        out = layers.Conv2D(1, 1, activation='sigmoid')(x)
        model = models.Model(inputs=inp, outputs=out)
        print("Built UNet with EfficientNetB0 encoder.")
        return model
    except Exception as e:
        print("EfficientNetB0 unavailable or failed to build:", e)
        print("Falling back to simple U-Net.")
        # fallback simple UNet with 3-channel input
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

# ---------- combined loss & metric ----------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2.*inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    # dice
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    dice = (2.*inter + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)
    return bce + (1 - dice)

# ---------- prepare pairs and datasets ----------
pairs = get_image_mask_pairs(IMAGES_DIR, MASKS_DIR)
if len(pairs) < 4:
    raise RuntimeError(f"Not enough image-mask pairs found ({len(pairs)}). Check your dataset.")

random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
val_pairs = pairs[split:]
print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

train_ds = tf_dataset_from_pairs(train_pairs, batch_size=BATCH_SIZE, shuffle=True, augment=True)
train_ds = train_ds.repeat()  # infinite for epochs
val_ds = tf_dataset_from_pairs(val_pairs, batch_size=BATCH_SIZE, shuffle=False, augment=False)

steps_per_epoch = max(1, len(train_pairs) // BATCH_SIZE)
validation_steps = max(1, len(val_pairs) // BATCH_SIZE)
print("steps_per_epoch =", steps_per_epoch, "validation_steps =", validation_steps)

# ---------- build, compile, train ----------
model = build_unet_efficientnet()
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=bce_dice_loss, metrics=[dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()

ckpt = os.path.join(DATA_ROOT, 'best_model.keras')  # use native Keras format
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', mode='max', patience=6, restore_best_weights=True, verbose=1)
]

model.fit(train_ds,
          epochs=EPOCHS,
          steps_per_epoch=steps_per_epoch,
          validation_data=val_ds,
          validation_steps=validation_steps,
          callbacks=callbacks)

print("Training finished. Best model at:", ckpt)

# ---------- inference + visualization ----------
def predict_and_visualize(model_path, image_path, out_path, threshold=0.5, min_area=100):
    m = tf.keras.models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h0,w0 = img.shape[:2]
    inp = preprocess_image_rgb(img)
    pred = m.predict(np.expand_dims(inp, axis=0))[0,...,0]
    pred_resized = cv2.resize((pred*255).astype(np.uint8), (w0,h0), interpolation=cv2.INTER_LINEAR)
    _, pred_bin = cv2.threshold(pred_resized, int(threshold*255), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    pred_clean = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    pred_clean = cv2.morphologyEx(pred_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(pred_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = img.copy()
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

# Example usage (uncomment and edit the path to test inference):
# predict_and_visualize(os.path.join(DATA_ROOT, 'best_model.keras'), os.path.join(IMAGES_DIR, '1008.png'), os.path.join(DATA_ROOT, 'vis_1008.png'))
