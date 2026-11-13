"""
HSEmotion 기반 재추론 및 5분류(Neutral 포함) 통합 파이프라인
======================================================================
입력: 원본 이미지, 오분류 정정 CSV 파일
출력: 재분류된 5감정 폴더 구조의 심볼릭 링크(img, segmentation) 및 통합 JSON(label_mod)
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
import urllib.request
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
# [사용자 설정] 입출력 경로 및 환경 변수
# ==============================================================================
# 1. 입력 경로 (건드리지 않는 원본 데이터)
RAW_IMG_DIR = "/workspace/데이터셋/img"
RAW_SEG_DIR = "/workspace/데이터셋/segmentation"
CSV_PATH    = "/workspace/user1/data_classification/3차 프로젝트(4조)-오분류 감정 라벨링_수정판.csv"

# 2. 출력 경로 (재분류된 최종 데이터셋이 모일 새로운 폴더)
OUTPUT_BASE = "/workspace/user1/dataset_v2"
LBL_OUT_DIR = os.path.join(OUTPUT_BASE, "label_mod")
IMG_OUT_DIR = os.path.join(OUTPUT_BASE, "img")         # 용량 낭비를 막기 위해 심볼릭 링크로 복사됨
SEG_OUT_DIR = os.path.join(OUTPUT_BASE, "segmentation") # NPZ 원본 심볼릭 링크

# 3. 모델 설정
MODEL_DIR  = "/workspace/user1/data_classification/relabeled_user5"
MODEL_NAME = "enet_b2_8"
# ==============================================================================

EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

# 출력 폴더들을 미리 생성
for d in [LBL_OUT_DIR, IMG_OUT_DIR, SEG_OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 1. 오분류 CSV 매핑 로드 ──────────────────────────────────────────
def load_correction_map(csv_path):
    print(f"CSV 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    df['정정 감정'] = df['정정 감정'].fillna('')
    df['BOX 이상'] = df['BOX 이상'].fillna('No')
    
    correction_map = {}
    box_issue_map = {}
    
    for _, row in df.iterrows():
        fname = str(row['파일명']).strip()
        new_emo = str(row['정정 감정']).strip().lower()
        
        if new_emo:
            # neutral 통합 (blank -> neutral 포함)
            if new_emo == 'blank': new_emo = 'neutral'
            correction_map[fname] = new_emo
            box_issue_map[fname] = str(row['BOX 이상']).strip()
            
    print(f"✅ 수동 정정 데이터 {len(correction_map)}건 로드 완료.\n")
    return correction_map, box_issue_map

# ── 얼굴 감지 모델 로드 (원본 동일) ──────────────────────────────────
def load_face_detector():
    model_path  = os.path.join(MODEL_DIR, "face_model.caffemodel")
    config_path = os.path.join(MODEL_DIR, "deploy.prototxt")

    if not os.path.exists(model_path):
        urllib.request.urlretrieve("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", model_path)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", config_path)
    return cv2.dnn.readNetFromCaffe(config_path, model_path)

def detect_face(net, img_bgr):
    h, w = img_bgr.shape[:2]
    for threshold in [0.4, 0.25, 0.15]:
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        if faces:
            return max(faces, key=lambda f: f[2] * f[3])

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    if len(faces) > 0:
        return tuple(max(faces, key=lambda f: f[2] * f[3]))
    return None

# ── 이미지 1장 분석 ─────────────────────────────────────────────────
def analyze_image(img_path, fer, net):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None

    h, w = img_bgr.shape[:2]
    face = detect_face(net, img_bgr)

    if face is None:
        face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bbox = {"minX": 0, "minY": 0, "maxX": w, "maxY": h}
        face_detected = False
    else:
        x, y, fw, fh = face
        face_rgb = cv2.cvtColor(img_bgr[y:y+fh, x:x+fw], cv2.COLOR_BGR2RGB)
        bbox = {"minX": int(x), "minY": int(y), "maxX": int(x+fw), "maxY": int(y+fh)}
        face_detected = True

    try:
        emotion_label, scores = fer.predict_emotions(face_rgb, logits=False)
    except Exception as e:
        return {"error": str(e)}

    # Top 1~3까지 감정 정보 명확히 기록
    score_dict = {e: round(float(s), 4) for e, s in zip(EMOTIONS, scores)}
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_scores[:3]

    return {
        "bbox": bbox,
        "face_detected": face_detected,
        "top1": {"emotion": top3[0][0], "score": top3[0][1]},
        "top2": {"emotion": top3[1][0], "score": top3[1][1]},
        "top3": {"emotion": top3[2][0], "score": top3[2][1]},
        "all_scores": score_dict
    }

# ── 메인 실행부 ───────────────────────────────────────────────────
def run_rebuild_pipeline():
    print("=" * 50)
    print("🚀 5분류 기반 재추론 & 데이터셋 통합 파이프라인 시작")
    print("=" * 50)

    # 모델 및 CSV 로드
    correction_map, box_issue_map = load_correction_map(CSV_PATH)
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    fer = HSEmotionRecognizer(model_name=MODEL_NAME)
    net = load_face_detector()

    # 결과를 모아둘 딕셔너리 구조: results[split][final_emotion] = [...]
    final_results = { 'train': {e: [] for e in ['anger', 'happy', 'panic', 'sadness', 'neutral']},
                      'val':   {e: [] for e in ['anger', 'happy', 'panic', 'sadness', 'neutral']} }

    for split in ["train", "val"]:
        split_dir = os.path.join(RAW_IMG_DIR, split)
        if not os.path.isdir(split_dir): continue

        # NPZ 원본 심볼릭 링크 생성 (segmentation/원본 보존)
        seg_split_dir = os.path.join(RAW_SEG_DIR, split)
        if os.path.exists(seg_split_dir):
            os.makedirs(os.path.join(SEG_OUT_DIR, split), exist_ok=True)
            for npz_file in os.listdir(seg_split_dir):
                src_npz = os.path.join(seg_split_dir, npz_file)
                dst_npz = os.path.join(SEG_OUT_DIR, split, npz_file)
                if not os.path.exists(dst_npz):
                    os.symlink(src_npz, dst_npz)

        # 이미지 처리 시작
        for folder_name in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path): continue
            
            # 원본 감정 파악 (blank -> neutral 로 치환)
            original_emotion = 'neutral' if folder_name == 'blank' else folder_name

            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
            
            for filename in tqdm(image_files, desc=f"[{split}/{original_emotion}]", unit="장"):
                # 1. 재분류 감정 결정 (CSV 우선, 없으면 폴더명)
                final_emotion = correction_map.get(filename, original_emotion)
                # 혹시라도 등록되지 않은 감정이 나온 경우 예외처리
                if final_emotion not in final_results[split]:
                    continue

                # 2. 이미지 심볼릭 링크 생성 (용량 절약하며 5분류 구조 생성)
                src_img = os.path.join(folder_path, filename)
                dst_img_dir = os.path.join(IMG_OUT_DIR, split, final_emotion)
                os.makedirs(dst_img_dir, exist_ok=True)
                dst_img = os.path.join(dst_img_dir, filename)
                if not os.path.exists(dst_img):
                    os.symlink(src_img, dst_img)

                # 3. 모델 추론
                res = analyze_image(src_img, fer, net)
                if res is None or "error" in res:
                    continue

                # 4. JSON 데이터 조립
                final_results[split][final_emotion].append({
                    "filename": filename,
                    "split": split,
                    "original_emotion": original_emotion, # ★ 중요: NPZ 마스크 조회를 위한 키값
                    "final_emotion": final_emotion,       # 정정된 최종 감정
                    "is_box_issue": box_issue_map.get(filename, "No"), # 박스 이상 여부
                    "bbox": res["bbox"],
                    "annot_HSEmotion": {
                        "top1": res["top1"],
                        "top2": res["top2"],
                        "top3": res["top3"],
                        "all_scores": res["all_scores"],
                        "face_detected": res["face_detected"]
                    }
                })

    # ── 2. JSON 결과 저장 ─────────────────────────────────────────────
    print("\n💾 5분류 통합 JSON 저장 중...")
    for split in final_results:
        os.makedirs(os.path.join(LBL_OUT_DIR, split), exist_ok=True)
        for emotion, records in final_results[split].items():
            if not records: continue
            
            json_name = f"{split}_{emotion}.json"
            json_path = os.path.join(LBL_OUT_DIR, split, json_name)
            
            output_data = {
                "info": {
                    "split": split,
                    "emotion_class": emotion,
                    "total_images": len(records),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "images": records
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            print(f"  ✅ {json_name} ({len(records)}장)")

    print(f"\n🎉 작업 완료! 최종 데이터셋 위치: {OUTPUT_BASE}")

if __name__ == "__main__":
    run_rebuild_pipeline()
