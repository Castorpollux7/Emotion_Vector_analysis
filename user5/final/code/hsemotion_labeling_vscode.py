"""
HSEmotion 기반 감정 라벨링 + 폴더별 JSON 생성 (VSCode 리눅스 서버용)
======================================================================
설치:
    pip install hsemotion-onnx opencv-python tqdm

실행:
    python hsemotion_labeling.py

결과 구조:
    /workspace/user5/relabeled/
    ├── train_sadness.json
    ├── train_happy.json
    ├── train_anger.json
    ├── train_panic.json
    ├── train_blank.json
    ├── val_sadness.json
    └── ...
"""

import os
import cv2
import json
import numpy as np
import urllib.request
from tqdm import tqdm
from datetime import datetime

# ================= 경로 설정 =================
INPUT_DIR  = "/workspace/user5/img/img(classify)"   # ← 이미지 폴더
OUTPUT_DIR = "/workspace/user5/relabeled"            # ← JSON 저장 폴더
MODEL_NAME = "enet_b2_8"

# 얼굴 감지 모델 저장 위치
MODEL_DIR  = "/workspace/user5"
# =============================================

EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear",
            "Happiness", "Neutral", "Sadness", "Surprise"]

FOLDER_LABEL_MAP = {
    "sadness": "sadness",
    "happy":   "happy",
    "anger":   "anger",
    "panic":   "panic",
    "blank":   "blank",
}

CHECKPOINT_EVERY = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 얼굴 감지 모델 로드 ───────────────────────────────────────
def load_face_detector():
    model_path  = os.path.join(MODEL_DIR, "face_model.caffemodel")
    config_path = os.path.join(MODEL_DIR, "deploy.prototxt")

    if not os.path.exists(model_path):
        print("얼굴 감지 모델 다운로드 중...")
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_path
        )
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            config_path
        )
        print("✅ 모델 다운로드 완료")

    return cv2.dnn.readNetFromCaffe(config_path, model_path)


# ── 얼굴 감지 ─────────────────────────────────────────────────
def detect_face(net, img_bgr):
    """신뢰도 임계값을 낮춰가며 얼굴 탐색 → 가장 큰 얼굴 반환"""
    h, w = img_bgr.shape[:2]

    for threshold in [0.4, 0.25, 0.15]:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
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

    # Haar Cascade fallback
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    if len(faces) > 0:
        return tuple(max(faces, key=lambda f: f[2] * f[3]))

    return None


# ── 이미지 1장 분석 ───────────────────────────────────────────
def analyze_image(img_path, fer, net):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]

    face = detect_face(net, img_bgr)

    if face is None:
        # 얼굴 못 찾으면 전체 이미지로 시도
        face_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bbox          = {"minX": 0, "minY": 0, "maxX": w, "maxY": h}
        face_detected = False
    else:
        x, y, fw, fh  = face
        face_rgb      = cv2.cvtColor(img_bgr[y:y+fh, x:x+fw], cv2.COLOR_BGR2RGB)
        bbox          = {"minX": int(x), "minY": int(y),
                         "maxX": int(x+fw), "maxY": int(y+fh)}
        face_detected = True

    try:
        emotion_label, scores = fer.predict_emotions(face_rgb, logits=False)
    except Exception as e:
        return {"error": str(e)}

    score_dict    = {e: round(float(s), 4) for e, s in zip(EMOTIONS, scores)}
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top3          = sorted_scores[:3]

    return {
        "bbox":          bbox,
        "face_detected": face_detected,
        "top1":          {"emotion": top3[0][0], "score": top3[0][1]},
        "top2":          {"emotion": top3[1][0], "score": top3[1][1]},
        "top3":          {"emotion": top3[2][0], "score": top3[2][1]},
        "all_scores":    score_dict
    }


# ── 폴더 단위 처리 ────────────────────────────────────────────
def process_folder(folder_path, folder_label, split, fer, net):
    valid_ext   = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted([f for f in os.listdir(folder_path)
                          if f.lower().endswith(valid_ext)])

    results         = []
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{split}_{folder_label}_checkpoint.json")

    # 체크포인트 복구
    processed_set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        results       = ckpt["results"]
        processed_set = set(ckpt["processed"])
        print(f"  [체크포인트] {len(processed_set)}장 복구")

    remaining = [f for f in image_files if f not in processed_set]

    if not remaining:
        print(f"  이미 완료됨 ({len(image_files)}장)")
        return results

    for filename in tqdm(remaining, desc=f"  {split}/{folder_label}", unit="장", leave=False):
        img_path = os.path.join(folder_path, filename)

        try:
            result = analyze_image(img_path, fer, net)

            if result is None:
                entry = {"filename": filename, "folder_label": folder_label,
                         "split": split, "error": "이미지 읽기 실패"}
            elif "error" in result:
                entry = {"filename": filename, "folder_label": folder_label,
                         "split": split, "error": result["error"]}
            else:
                entry = {
                    "filename":     filename,
                    "folder_label": folder_label,
                    "split":        split,
                    "bbox":         result["bbox"],
                    "annot_HSEmotion": {
                        "top1":          result["top1"],
                        "top2":          result["top2"],
                        "top3":          result["top3"],
                        "all_scores":    result["all_scores"],
                        "face_detected": result["face_detected"]
                    }
                }

            results.append(entry)
            processed_set.add(filename)

            # 체크포인트 저장
            if len(processed_set) % CHECKPOINT_EVERY == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "results":   results,
                        "processed": list(processed_set),
                        "saved_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f, ensure_ascii=False)

        except Exception as e:
            tqdm.write(f"    [에러] {filename}: {e}")
            results.append({"filename": filename, "folder_label": folder_label,
                            "split": split, "error": str(e)})
            processed_set.add(filename)

    # 최종 체크포인트
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "processed": list(processed_set),
                   "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False)

    return results


# ── 메인 ──────────────────────────────────────────────────────
def run_labeling():
    print("=" * 50)
    print(f"HSEmotion 라벨링 시작 (모델: {MODEL_NAME})")
    print(f"입력 경로: {INPUT_DIR}")
    print(f"출력 경로: {OUTPUT_DIR}")
    print("=" * 50)

    # 경로 확인
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 이미지 폴더 없음: {INPUT_DIR}")
        print("   INPUT_DIR 경로를 확인해주세요.")
        return

    # 모델 로드
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    except ImportError:
        print("❌ hsemotion-onnx 설치 필요:")
        print("   pip install hsemotion-onnx")
        return

    print(f"\n모델 로드 중: {MODEL_NAME}")
    fer = HSEmotionRecognizer(model_name=MODEL_NAME)
    net = load_face_detector()
    print("✅ 모델 로드 완료\n")

    total_stats = {}

    for split in ["train", "val"]:
        split_dir = os.path.join(INPUT_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"⚠️  {split} 폴더 없음: {split_dir}")
            continue

        print(f"\n━━━ {split.upper()} 처리 중 ━━━")

        for folder_name in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            folder_label = FOLDER_LABEL_MAP.get(folder_name.lower(), folder_name.lower())
            print(f"\n[{split}/{folder_name}] 처리 중...")

            results = process_folder(folder_path, folder_label, split, fer, net)

            # JSON 저장
            json_path   = os.path.join(OUTPUT_DIR, f"{split}_{folder_label}.json")
            output_data = {
                "info": {
                    "folder":       folder_name,
                    "folder_label": folder_label,
                    "split":        split,
                    "model":        f"HSEmotion_{MODEL_NAME}",
                    "total_images": len(results),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "images": results
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            success = sum(1 for r in results if "annot_HSEmotion" in r)
            face_ok = sum(1 for r in results
                          if "annot_HSEmotion" in r
                          and r["annot_HSEmotion"]["face_detected"])
            error   = sum(1 for r in results if "error" in r)

            total_stats[f"{split}/{folder_label}"] = {
                "total": len(results), "success": success,
                "face_ok": face_ok, "error": error
            }

            print(f"  ✅ JSON 저장: {json_path}")
            print(f"     전체 {len(results)}장 / 성공 {success}장 / 얼굴감지 {face_ok}장 / 에러 {error}장")

    # 최종 리포트
    print(f"""
╔══════════════════════════════════════════════════════╗
║              전체 라벨링 완료 리포트                  ║
╠══════════════════════════════════════════════════════╣""")
    for folder, stats in total_stats.items():
        print(f"║  {folder:<20}: 전체 {stats['total']:>4} / 얼굴감지 {stats['face_ok']:>4} ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"\n📁 JSON 저장 위치: {OUTPUT_DIR}")

    # shared 폴더 복사 안내
    print(f"""
팀원 공유하려면 터미널에서:
    cp -r {OUTPUT_DIR} /workspace/shared/
""")


# ── 결과 확인 ─────────────────────────────────────────────────
def verify_json(split="train", folder_label="sadness", n=3):
    json_path = os.path.join(OUTPUT_DIR, f"{split}_{folder_label}.json")
    if not os.path.exists(json_path):
        print(f"JSON 없음: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n📋 {json_path}")
    print(f"전체 이미지: {len(data['images'])}장\n")

    for i, img in enumerate(data["images"][:n]):
        print(f"[샘플 {i+1}]")
        print(json.dumps(img, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    run_labeling()
    verify_json(split="train", folder_label="sadness", n=2)
