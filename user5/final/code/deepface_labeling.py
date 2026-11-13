"""
DeepFace 기반 감정 분류 + TOP3 라벨링 (Google Colab용)
==================================================
- 얼굴 감지: RetinaFace (가장 정확)
- 감정 분류: DeepFace (Xception 백엔드)
- TOP3 감정 + confidence 추출
- 체크포인트 지원 (중단 후 이어서 가능)
- 결과: CSV + JSON 저장

[Colab 실행 순서]
  셀 1 - 패키지 설치:
    !pip install deepface tqdm pandas opencv-python -q

  셀 2 - Google Drive 연결:
    from google.colab import drive
    drive.mount('/content/drive')

  셀 3 - 이 파일 전체 실행 (아래 경로 먼저 수정)
"""

# ── Google Colab: Drive 연결 ──────────────────────────────────
import os, sys

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

os.environ["DEEPFACE_HOME"] = "/content/drive/MyDrive/.deepface" if IN_COLAB else os.path.expanduser("~/.deepface")

import cv2
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ================= 설정 =================
# ↓↓↓ 본인 Google Drive 경로로 수정하세요 ↓↓↓
if IN_COLAB:
    INPUT_DIR  = "/content/drive/MyDrive/project3/img"      # ← Drive 이미지 폴더
    OUTPUT_DIR = "/content/drive/MyDrive/project3/labeled"  # ← Drive 저장 폴더
else:
    INPUT_DIR  = r"D:\project 3\img"
    OUTPUT_DIR = r"D:\project 3\labeled"
CHECKPOINT_EVERY = 50                         # 몇 장마다 저장할지
CONFIDENCE_THRESHOLD = 0.55                   # 이 이상만 confirmed로

# 분류할 감정 목록 (DeepFace 기준)
# DeepFace 반환값: angry, disgust, fear, happy, sad, surprise, neutral
EMOTION_MAP = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "panic",
    "happy":    "happy",
    "sad":      "sadness",
    "surprise": "panic",
    "neutral":  "blank",
}
TARGET_EMOTIONS = ["sadness", "anger", "panic", "happy", "blank"]
# ========================================

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "labeling_checkpoint.json")


# ── 폴더 생성 ─────────────────────────────────────────────────
def setup_folders():
    for split in ["train", "val"]:
        for emotion in TARGET_EMOTIONS:
            os.makedirs(os.path.join(OUTPUT_DIR, "confirmed", split, emotion), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_DIR, "review_needed", split, emotion), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "no_face"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "error"), exist_ok=True)


# ── 체크포인트 ─────────────────────────────────────────────────
def save_checkpoint(processed_files, log, stats):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "processed_files": processed_files,
            "log":             log,
            "stats":           stats,
            "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False)
    os.replace(tmp, CHECKPOINT_PATH)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    print(f"[체크포인트 발견] {ckpt['saved_at']} — {len(ckpt['processed_files'])}장 완료")
    print(f"이어서 진행합니다.\n처음부터 하려면 삭제: {CHECKPOINT_PATH}\n")
    return ckpt


# ── 파일 탐색 ─────────────────────────────────────────────────
def collect_images(input_dir):
    """하위 폴더 포함 모든 이미지 수집 → [(파일경로, split), ...]"""
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    results = []

    for root, dirs, files in os.walk(input_dir):
        # 경로에서 split 추출 (train/val 폴더 구조 대응)
        rel = os.path.relpath(root, input_dir).lower()
        if "val" in rel:
            split = "val"
        else:
            split = "train"

        for fname in files:
            if fname.lower().endswith(valid_ext):
                results.append((os.path.join(root, fname), split))

    return sorted(results)


# ── DeepFace 감정 분석 (백엔드 순차 재시도) ──────────────────
# 얼굴 미감지 시 다음 백엔드로 자동 전환
DETECTOR_BACKENDS = [
    "retinaface",   # 1순위: 가장 정확, 다양한 각도에 강함
    "mtcnn",        # 2순위: 근거리 셀카에 강함
    "opencv",       # 3순위: 정면 얼굴에 빠름
    "ssd",          # 4순위: 속도 빠름
    "mediapipe",    # 5순위: 마지막 수단
]

def parse_emotion_result(result):
    """DeepFace 결과 파싱 → 공통 형식으로 변환"""
    res = result[0] if isinstance(result, list) else result

    face_confidence = res.get("face_confidence", 1.0)
    if face_confidence < 0.3:
        return None

    raw_emotions = res.get("emotion", {})
    if not raw_emotions:
        return None

    # 점수 정규화 (0~1)
    total = sum(raw_emotions.values())
    if total == 0:
        return None
    normalized = {k: v / total for k, v in raw_emotions.items()}

    # TOP3 추출 + 감정명 매핑
    top3_raw = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
    top3     = [(EMOTION_MAP.get(e, e), round(s, 4)) for e, s in top3_raw]
    dominant   = top3[0][0]
    confidence = top3[0][1]

    return {
        "top3":       top3,
        "dominant":   dominant,
        "confidence": confidence,
        "all_scores": {EMOTION_MAP.get(k, k): round(v, 4)
                       for k, v in normalized.items()}
    }

def analyze_emotion(img_path):
    """
    DeepFace 감정 분석 — 얼굴 미감지 시 백엔드 순차 재시도
    retinaface → mtcnn → opencv → ssd → mediapipe
    모든 백엔드 실패 시 None 반환
    """
  # ── 기존 analyze_emotion 함수를 아래로 교체 ───────────────────

# 시도할 백엔드 순서 (정확도 높은 순)
DETECTOR_BACKENDS = [
    "retinaface",   # 1순위: 가장 정확
    "mtcnn",        # 2순위: 다양한 각도에 강함
    "opencv",       # 3순위: 빠르고 안정적
    "ssd",          # 4순위: 저조도에 상대적으로 강함
]

def analyze_emotion(img_path):
    """
    여러 백엔드를 순서대로 시도
    모두 실패하면 None 반환
    """
    from deepface import DeepFace

    last_error = None

    for backend in DETECTOR_BACKENDS:
        try:
            result = DeepFace.analyze(
                img_path=img_path,
                actions=["emotion"],
                detector_backend=backend,
                enforce_detection=True,   # ← True로 설정해서 실패 시 다음 백엔드로 넘어감
                silent=True
            )

            res = result[0] if isinstance(result, list) else result

            face_confidence = res.get("face_confidence", 1.0)
            if face_confidence < 0.3:
                continue  # 신뢰도 낮으면 다음 백엔드 시도

            raw_emotions = res.get("emotion", {})
            if not raw_emotions:
                continue

            # 점수 정규화
            total = sum(raw_emotions.values())
            if total == 0:
                continue
            normalized = {k: v / total for k, v in raw_emotions.items()}

            # TOP3 추출
            top3_raw = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
            top3     = [(EMOTION_MAP.get(e, e), round(s, 4)) for e, s in top3_raw]
            dominant   = top3[0][0]
            confidence = top3[0][1]

            if dominant not in TARGET_EMOTIONS:
                dominant = "blank"

            # 성공 시 백엔드 정보도 같이 반환
            return {
                "top3":       top3,
                "dominant":   dominant,
                "confidence": confidence,
                "all_scores": {EMOTION_MAP.get(k, k): round(v, 4)
                               for k, v in normalized.items()},
                "backend":    backend   # ← 어떤 백엔드로 성공했는지 기록
            }

        except Exception as e:
            error_msg = str(e).lower()
            if "face could not be detected" in error_msg or "no face" in error_msg:
                # 얼굴 못 찾음 → 다음 백엔드로
                last_error = f"{backend}: no face"
                continue
            else:
                # 다른 에러 → 다음 백엔드로
                last_error = f"{backend}: {str(e)}"
                continue

    # 모든 백엔드 실패
    return None


# ── 메인 ──────────────────────────────────────────────────────
def run_labeling():
    print("DeepFace 라벨링 시작")
    print("=" * 50)

    # DeepFace 임포트 확인
    try:
        from deepface import DeepFace
        print("✅ DeepFace 로드 완료")
    except ImportError:
        print("❌ DeepFace가 없어요. 설치: pip install deepface")
        return

    setup_folders()

    # 이미지 수집
    all_images = collect_images(INPUT_DIR)
    total = len(all_images)
    print(f"✅ 이미지 {total}장 발견\n")

    # 체크포인트 복구
    ckpt = load_checkpoint()
    if ckpt:
        processed_set = set(ckpt["processed_files"])
        log           = ckpt["log"]
        stats         = ckpt["stats"]
        all_images    = [(p, s) for p, s in all_images if p not in processed_set]
        print(f"남은 작업: {len(all_images)}장 / 전체 {total}장\n")
    else:
        processed_set = set()
        log           = []
        stats         = {
            "confirmed":    0,
            "review_needed": 0,
            "no_face":      0,
            "error":        0
        }

    batch_counter = 0

    for img_path, split in tqdm(all_images, desc="라벨링 진행", unit="장"):
        filename = os.path.basename(img_path)

        try:
            emotion_result = analyze_emotion(img_path)

            # 얼굴 미감지
            if emotion_result is None:
                dest = os.path.join(OUTPUT_DIR, "no_face", filename)
                shutil.copy2(img_path, dest)
                stats["no_face"] += 1
                log.append({
                    "file":       img_path,
                    "filename":   filename,
                    "split":      split,
                    "result":     "no_face",
                    "dominant":   None,
                    "confidence": None,
                    "top3":       None
                })
                processed_set.add(img_path)
                batch_counter += 1
                continue

            dominant   = emotion_result["dominant"]
            confidence = emotion_result["confidence"]
            top3       = emotion_result["top3"]

            # 감정이 TARGET_EMOTIONS에 없으면 가장 가까운 것으로
            if dominant not in TARGET_EMOTIONS:
                dominant = "blank"

            # confidence 기준으로 confirmed / review_needed 분류
            if confidence >= CONFIDENCE_THRESHOLD:
                dest_folder = os.path.join(OUTPUT_DIR, "confirmed", split, dominant)
                result_key  = "confirmed"
                stats["confirmed"] += 1
            else:
                dest_folder = os.path.join(OUTPUT_DIR, "review_needed", split, dominant)
                result_key  = "review_needed"
                stats["review_needed"] += 1

            shutil.copy2(img_path, os.path.join(dest_folder, filename))

            log.append({
                "file":         img_path,
                "filename":     filename,
                "split":        split,
                "result":       result_key,
                "dominant":     dominant,
                "confidence":   confidence,
                "top1":         f"{top3[0][0]}:{top3[0][1]:.3f}",
                "top2":         f"{top3[1][0]}:{top3[1][1]:.3f}" if len(top3) > 1 else "",
                "top3":         f"{top3[2][0]}:{top3[2][1]:.3f}" if len(top3) > 2 else "",
                "used_backend": emotion_result.get("backend", ""),
                "all_scores":   emotion_result["all_scores"]
            })
            processed_set.add(img_path)

        except Exception as e:
            tqdm.write(f"  [에러] {filename}: {e}")
            shutil.copy2(img_path, os.path.join(OUTPUT_DIR, "error", filename))
            stats["error"] += 1
            log.append({
                "file":     img_path,
                "filename": filename,
                "split":    split,
                "result":   "error",
                "error":    str(e)
            })
            processed_set.add(img_path)

        # 체크포인트 저장
        batch_counter += 1
        if batch_counter % CHECKPOINT_EVERY == 0:
            save_checkpoint(list(processed_set), log, stats)
            tqdm.write(f"  💾 체크포인트 저장 ({len(processed_set)}/{total}장)")

    # 최종 저장
    save_checkpoint(list(processed_set), log, stats)

    # CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, "labeling_result.csv")
    df = pd.DataFrame(log)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 리포트
    print(f"""
╔══════════════════════════════════════════╗
║           라벨링 완료 리포트              ║
╠══════════════════════════════════════════╣
║  전체:          {total:>5}장                ║
║  자동 채택:     {stats['confirmed']:>5}장  (confidence ≥ {CONFIDENCE_THRESHOLD}) ║
║  검수 필요:     {stats['review_needed']:>5}장                ║
║  얼굴 미감지:   {stats['no_face']:>5}장                ║
║  에러:          {stats['error']:>5}장                ║
╚══════════════════════════════════════════╝
CSV 저장: {csv_path}
""")


# ── 결과 미리보기 ─────────────────────────────────────────────
def preview_results(n=5):
    """라벨링 결과 CSV 샘플 출력"""
    csv_path = os.path.join(OUTPUT_DIR, "labeling_result.csv")
    if not os.path.exists(csv_path):
        print("labeling_result.csv 없음. run_labeling() 먼저 실행하세요.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n전체 {len(df)}장 결과 샘플 ({n}장):\n")
    print(df[["filename", "split", "dominant", "confidence", "top1", "top2", "top3"]].head(n).to_string(index=False))

    print(f"\n감정별 분포:")
    print(df["dominant"].value_counts().to_string())

    print(f"\nsplit별 분포:")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    # 경로 확인
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 이미지 폴더 없음: {INPUT_DIR}")
    else:
        print(f"✅ 이미지 폴더: {INPUT_DIR}")
        print(f"✅ 저장 위치:   {OUTPUT_DIR}\n")
        run_labeling()
        preview_results(n=10)
