import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# Folders
IMG_DIR = "dataset/train"
MASK_DIR = "trainData/train_masks"
OUTPUT_JSON = "openeds_coco.json"

CATEGORY_MAP = {
    2: {"id": 1, "name": "iris"},
    3: {"id": 2, "name": "pupil"}
}

coco = {
    "images": [],
    "annotations": [],
    "categories": list(CATEGORY_MAP.values())
}

image_id = 0
annotation_id = 0

img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])

for file_name in tqdm(img_files):
    img_path = os.path.join(IMG_DIR, file_name)
    mask_path = os.path.join(MASK_DIR, file_name.replace(".png", ".npy"))

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # Load .npy mask
    mask = np.load(mask_path)

    # Add image info
    coco["images"].append({
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    })

    for mask_value, cat_info in CATEGORY_MAP.items():
        binary_mask = (mask == mask_value).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue

            segmentation = contour.flatten().tolist()
            x, y, w_box, h_box = cv2.boundingRect(contour)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_info["id"],
                "segmentation": [segmentation],
                "bbox": [x, y, w_box, h_box],
                "area": float(area),
                "iscrowd": 0
            })
            annotation_id += 1

    image_id += 1

# Save COCO JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ Saved COCO-format annotations to: {OUTPUT_JSON}")