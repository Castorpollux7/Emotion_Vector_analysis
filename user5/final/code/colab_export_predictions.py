"""
[Colab에서 실행] 모델 예측 결과를 CSV로 저장
==============================================
relabel_json.py 의 MODEL_SOURCE = "csv" 방법과 함께 사용합니다.

1. 이 코드를 Colab 셀에 붙여넣기
2. 실행 후 predictions.csv를 로컬로 다운로드
3. relabel_json.py 에서 PRED_CSV_PATH 지정 후 실행
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# ===================== 설정 =====================
BASE_DIR = "/content/drive/MyDrive/Face.Sent"
LABEL_DIR = os.path.join(BASE_DIR, "label")
IMG_DIR   = os.path.join(BASE_DIR, "img")
OUTPUT_CSV = "/content/predictions.csv"

EMOTION_CLASSES = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
IMG_SIZE = (224, 224)
# ================================================


def predict_top2(model, img_path, classes, img_size=(224, 224)):
    img = keras_image.load_img(img_path, target_size=img_size)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    top2_idx = np.argsort(probs)[::-1][:2]
    return classes[top2_idx[0]], classes[top2_idx[1]]


def find_image(filename, img_dir):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(img_dir, split)
        if not os.path.isdir(split_path):
            continue
        for emotion_folder in os.listdir(split_path):
            candidate = os.path.join(split_path, emotion_folder, filename)
            if os.path.exists(candidate):
                return candidate
    return None


# model_ResNet2 는 이미 로드된 상태라고 가정
classes_arr = np.array(EMOTION_CLASSES)
results = []

for split in ["train", "val"]:
    label_split_path = os.path.join(LABEL_DIR, split)
    if not os.path.isdir(label_split_path):
        continue
    for json_file in os.listdir(label_split_path):
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(label_split_path, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for item in items:
            filename = item.get("filename", "")
            img_path = find_image(filename, IMG_DIR)
            if img_path is None:
                print(f"[NOT FOUND] {filename}")
                continue
            top1, top2 = predict_top2(model_ResNet2, img_path, classes_arr, IMG_SIZE)
            results.append({"filename": filename, "top1": top1, "top2": top2})
            print(f"{filename} → top1: {top1}, top2: {top2}")

pred_df = pd.DataFrame(results)
pred_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 예측 결과 저장 완료: {OUTPUT_CSV} ({len(pred_df)}개)")

# Colab에서 다운로드
from google.colab import files
files.download(OUTPUT_CSV)
