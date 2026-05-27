from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Paths
coco_json_path = "dataset/train_data.json"
image_dir = "dataset/train_data"
output_mask_dir = "dataset/train_masks_"

# Create output folder
os.makedirs(output_mask_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(coco_json_path)

# Get all image IDs
img_ids = coco.getImgIds()

# Get category ID for pupil (adjust as needed — usually class ID 2)
pupil_category_ids = [cat['id'] for cat in coco.loadCats(coco.getCatIds()) if cat['name'] == 'pupil']

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    height, width = img_info["height"], img_info["width"]

    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Get all annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=pupil_category_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        mask |= coco.annToMask(ann) * 255  # Convert to 0/255

    # Save mask
    mask_name = os.path.splitext(file_name)[0] + "_mask.png"
    mask_path = os.path.join(output_mask_dir, mask_name)
    Image.fromarray(mask).save(mask_path)