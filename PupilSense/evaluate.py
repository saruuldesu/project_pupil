import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import warnings
warnings.filterwarnings("ignore", message=".*indexing argument.*")
from tqdm import tqdm

import torch
from tqdm import tqdm

def main():
    all_scores = []
    total_preds = 0
    total_images = 0
    # === Register the dataset ===
    register_coco_instances(
        "test_dataset", 
        {}, 
        "./dataset/test_data.json", 
        "./dataset/test"
    )

    # === Load configuration ===
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join("models", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.DATASETS.TEST = ("test_dataset", )
    cfg.MODEL.DEVICE = "cpu"

    # ❗ Set DataLoader workers to 0 to avoid multiprocessing errors on macOS
    cfg.DATALOADER.NUM_WORKERS = 0

    # === Create evaluator and run inference ===

    evaluator = COCOEvaluator(
        "test_dataset", 
        cfg, 
        False, 
        output_dir="./output"
    )
    val_loader = build_detection_test_loader(cfg, "test_dataset")

    predictor = DefaultPredictor(cfg)
    print("Starting evaluation...") 
    val_loader = build_detection_test_loader(cfg, "test_dataset")

    # DEBUG: Try loading just 1 batch
    print("Testing if DataLoader works...")
    for batch in val_loader:
        print("Loaded 1 batch!")
        print("Batch keys:", batch[0].keys())
        break



    print("Running manual inference...")

    for i, inputs in enumerate(tqdm(val_loader)):
        try:
            with torch.no_grad():
                outputs = predictor.model(inputs)

            instances = outputs[0]["instances"]
            scores = instances.scores.cpu().numpy()

            print(f"\n✅ Image: {inputs[0]['file_name']}")
            print(f"🔸 Predictions: {len(scores)}")
            print(f"🔸 Scores: {scores}")

            all_scores.extend(scores)
            total_preds += len(scores)
            total_images += 1

        except Exception as e:
            print(f"❌ Inference failed on image {inputs[0]['file_name']}: {e}")
            break

    # === Summary ===
    if total_images > 0:
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        print("\n📊 === Summary ===")
        print(f"Total images: {total_images}")
        print(f"Total predictions: {total_preds}")
        print(f"Average predictions per image: {total_preds / total_images:.2f}")
        print(f"Average confidence score: {avg_score:.4f}")
    else:
        print("No predictions made.")
    print("Evaluation done.")        # after inferenc


# ✅ Add this block to avoid multiprocessing errors on macOS
if __name__ == "__main__":
    main()
