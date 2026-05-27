from detectron2.data.datasets import register_coco_instances
import os

# Register the training dataset
register_coco_instances(
    "openeds_train",  # Dataset name (you can choose this name)
    {},               # Metadata (empty dictionary for now)
    "dataset/train_data.json",  # Path to the COCO annotations (JSON)
    "dataset/train"  # Path to the images folder (train images)
)

# Register the test dataset
register_coco_instances(
    "openeds_test",  # Dataset name (you can choose this name)
    {},               # Metadata (empty dictionary for now)
    "dataset/test_data.json",  # Path to the COCO test annotations
    "dataset/test"  # Path to the test images folder
)