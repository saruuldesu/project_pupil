
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np

# Load config
cfg = get_cfg()
cfg.merge_from_file("models_pre/Config/config.yaml")  
cfg.MODEL.WEIGHTS = "models_pre/model_final.pth"  # your checkpoint
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"  # or "cuda" if using GPU

img_path1 = "dataset/hh/test1.png"
img_path2 = "dataset/hh/test2.png"
avg_diam = 12
max_diff = 1

predictor = DefaultPredictor(cfg)
def calculate_pupil_to_iris_ratio(image_path):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    instances = outputs["instances"]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()

    iris_area = 0
    pupil_area = 0

    for cls, mask in zip(pred_classes, pred_masks):
        area = np.sum(mask)
        if cls == 0:  # ID 0 means iris if your model class order is [iris, pupil]
            iris_area += area
        elif cls == 1:  # ID 1 means pupil
            pupil_area += area

    if iris_area == 0:
        print("虹彩は検出されませんでした.")
        return None

    ratio = pupil_area / iris_area
    print(f"👁️ 瞳孔面積と虹彩面積の比率： {ratio:.4f}")
    # print("Classes:", predictor.metadata.thing_classes)
    return ratio

def determine_differenceIration(ratio_1,ratio_2):
    diff = abs(ratio_1 - ratio_2)*avg_diam
    print("違い (in mm): {:.2f}".format(diff))
    return diff
def main(imgPath1, imgPath2):
    ratio_1 = calculate_pupil_to_iris_ratio(imgPath1)
    ratio_2 = calculate_pupil_to_iris_ratio(imgPath2)

    if ratio_1 is None or ratio_2 is None:
        return "👁️ 目のセグメンテーションは検出できませんでした。画像を確認してください。"

    diff = determine_differenceIration(ratio_1, ratio_2)

    if diff <= 0.5:
        status = "✅ 正常な瞳孔サイズが検出されました。"
    elif diff <= max_diff:
        status = "⚠️ 軽度の瞳孔不同の可能性が検出されました。"
    else:
        status = "🚨 脳損傷または神経異常の可能性が高い！"

    return f"{status} | 差:: {diff:.2f} мм | 比率 (左: {ratio_1:.4f}, 右: {ratio_2:.4f})"


if __name__ == "__main__":
    main()
