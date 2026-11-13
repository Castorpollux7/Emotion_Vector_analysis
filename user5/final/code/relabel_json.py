"""
JSON 라벨 파일 faceExp 재라벨링 스크립트
=========================================
모델 추론 결과(top1, top2)로 JSON 라벨의 faceExp 필드를 업데이트합니다.

[실행 방법]
  방법 1 - 모델 직접 추론:
      MODEL_SOURCE = "model"  로 설정 후 MODEL_PATH 지정

  방법 2 - 사전 예측 CSV 사용 (Colab에서 미리 저장한 경우):
      MODEL_SOURCE = "csv"  로 설정 후 PRED_CSV_PATH 지정
      CSV 형식: filename, top1, top2

설치:
    pip install tensorflow pillow pandas tqdm
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===================== 설정 =====================

# "model" = h5 모델로 직접 추론 | "csv" = 사전 예측 CSV 사용
MODEL_SOURCE = "model"

# [방법 1] 모델 경로
MODEL_PATH = r"D:\project 3\model_ResNet2.h5"

# [방법 2] 사전 예측 CSV (filename, top1, top2 컬럼 필요)
PRED_CSV_PATH = r"D:\project 3\predictions.csv"

# JSON 라벨 파일이 있는 폴더 (train/val 하위 폴더 구조 가능)
JSON_DIR = r"D:\project 3\label"

# 이미지 폴더 (train/<emotion>/, val/<emotion>/ 구조)
IMG_DIR = r"D:\project 3\img"

# 감정 클래스 (모델 훈련 시 사용한 순서와 동일하게)
EMOTION_CLASSES = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]

# 이미지 입력 크기
IMG_SIZE = (224, 224)

# 업데이트할 faceExp 위치 (True = 업데이트)
UPDATE_ANNOT_A = True
UPDATE_ANNOT_B = True
UPDATE_ANNOT_C = True
UPDATE_FACEEXP_UPLOADER = False  # 최상위 faceExp_uploader도 업데이트할지

# ================================================


def find_image(filename, img_dir):
    """train/<emotion>/, val/<emotion>/ 구조에서 파일 찾기"""
    for split in ["train", "val", "test"]:
        split_path = os.path.join(img_dir, split)
        if not os.path.isdir(split_path):
            continue
        for emotion_folder in os.listdir(split_path):
            candidate = os.path.join(split_path, emotion_folder, filename)
            if os.path.exists(candidate):
                return candidate
    # 직접 경로
    direct = os.path.join(img_dir, filename)
    if os.path.exists(direct):
        return direct
    return None


def predict_top2_model(model, img_path, classes, img_size):
    """모델로 top1, top2 감정 예측"""
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    top2_idx = np.argsort(probs)[::-1][:2]
    return classes[top2_idx[0]], classes[top2_idx[1]]


def build_face_exp(top1, top2):
    """faceExp 객체 생성"""
    return {"top1": top1, "top2": top2}


def update_item(item, face_exp_value):
    """JSON 항목의 faceExp 필드 업데이트"""
    changed = False
    if UPDATE_FACEEXP_UPLOADER and "faceExp_uploader" in item:
        item["faceExp_uploader"] = face_exp_value
        changed = True
    for annot_key, should_update in [
        ("annot_A", UPDATE_ANNOT_A),
        ("annot_B", UPDATE_ANNOT_B),
        ("annot_C", UPDATE_ANNOT_C),
    ]:
        if should_update and annot_key in item and "faceExp" in item[annot_key]:
            item[annot_key]["faceExp"] = face_exp_value
            changed = True
    return changed


def relabel_with_model(json_dir, img_dir, model_path, classes, img_size):
    """모델 추론으로 JSON 파일 업데이트"""
    from tensorflow.keras.models import load_model

    print(f"모델 로딩: {model_path}")
    model = load_model(model_path)
    classes = np.array(classes)

    json_files = []
    for root, _, files in os.walk(json_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    print(f"\nJSON 파일 {len(json_files)}개 처리 시작\n")
    not_found = []
    updated_count = 0

    for json_path in tqdm(json_files, desc="JSON 라벨링"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]
        file_changed = False

        for item in items:
            filename = item.get("filename", "")
            img_path = find_image(filename, img_dir)
            if img_path is None:
                not_found.append(filename)
                continue

            top1, top2 = predict_top2_model(model, img_path, classes, img_size)
            face_exp = build_face_exp(top1, top2)
            if update_item(item, face_exp):
                file_changed = True
                updated_count += 1

        if file_changed:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    return updated_count, not_found


def relabel_with_csv(json_dir, csv_path):
    """사전 예측 CSV로 JSON 파일 업데이트"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    pred_map = {
        row["filename"].strip(): (str(row["top1"]).strip(), str(row["top2"]).strip())
        for _, row in df.iterrows()
    }
    print(f"CSV 예측 {len(pred_map)}개 로드 완료")

    json_files = []
    for root, _, files in os.walk(json_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    print(f"JSON 파일 {len(json_files)}개 처리 시작\n")
    not_found = []
    updated_count = 0

    for json_path in tqdm(json_files, desc="JSON 라벨링"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]
        file_changed = False

        for item in items:
            filename = item.get("filename", "")
            if filename not in pred_map:
                not_found.append(filename)
                continue

            top1, top2 = pred_map[filename]
            face_exp = build_face_exp(top1, top2)
            if update_item(item, face_exp):
                file_changed = True
                updated_count += 1

        if file_changed:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    return updated_count, not_found


def main():
    print("=" * 50)
    print("JSON faceExp 재라벨링 (top1 / top2)")
    print("=" * 50)

    if not os.path.isdir(JSON_DIR):
        print(f"❌ JSON 폴더를 찾을 수 없습니다: {JSON_DIR}")
        return

    if MODEL_SOURCE == "model":
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            return
        updated, not_found = relabel_with_model(
            JSON_DIR, IMG_DIR, MODEL_PATH, EMOTION_CLASSES, IMG_SIZE
        )
    elif MODEL_SOURCE == "csv":
        if not os.path.exists(PRED_CSV_PATH):
            print(f"❌ 예측 CSV를 찾을 수 없습니다: {PRED_CSV_PATH}")
            return
        updated, not_found = relabel_with_csv(JSON_DIR, PRED_CSV_PATH)
    else:
        print(f"❌ MODEL_SOURCE는 'model' 또는 'csv'여야 합니다.")
        return

    print(f"\n✅ 업데이트 완료: {updated}개 항목")
    if not_found:
        print(f"⚠️  이미지/예측 없음: {len(not_found)}개")
        for name in not_found[:10]:
            print(f"   - {name}")
        if len(not_found) > 10:
            print(f"   ... 외 {len(not_found) - 10}개")


if __name__ == "__main__":
    main()
