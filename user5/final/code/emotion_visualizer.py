import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ================= 설정 =================
INPUT_DIR        = r"C:\Users\사용자이름\Pictures\sadness"   # ← 본인 경로로 변경
OUTPUT_DIR       = r"C:\Users\사용자이름\Pictures\visualized" # ← 본인 경로로 변경
CHECKPOINT_EVERY = 100
MODEL_NAME       = 'enet_b0_8_best_afew'

EMOTION_COLORS = {
    'Anger':     (0,   0,   255),
    'Contempt':  (0,   165, 255),
    'Disgust':   (0,   128, 0  ),
    'Fear':      (128, 0,   128),
    'Happiness': (0,   255, 255),
    'Neutral':   (255, 200, 0  ),
    'Sadness':   (255, 0,   0  ),
    'Surprise':  (0,   255, 0  ),
}
EMOTIONS        = list(EMOTION_COLORS.keys())
DEFAULT_COLOR   = (200, 200, 200)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'viz_checkpoint.json')
# ========================================


# ── 체크포인트 ───────────────────────────────────────────────────
def save_checkpoint(processed_files, stats):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = CHECKPOINT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({
            'processed_files': processed_files,
            'stats':           stats,
            'saved_at':        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, ensure_ascii=False)
    os.replace(tmp, CHECKPOINT_PATH)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)
    print(f"[체크포인트 발견] {ckpt['saved_at']} — {len(ckpt['processed_files'])}장 완료")
    print(f"이어서 진행합니다. 처음부터 하려면 아래 파일을 삭제하세요:")
    print(f"  {CHECKPOINT_PATH}\n")
    return ckpt


# ── 얼굴 감지 (RetinaFace) ───────────────────────────────────────
def detect_faces(img_bgr):
    try:
        from retinaface import RetinaFace
        result = RetinaFace.detect_faces(img_bgr)

        if not isinstance(result, dict) or len(result) == 0:
            return []

        faces = []
        for face_data in result.values():
            x1, y1, x2, y2 = face_data['facial_area']
            x1, y1 = max(0, x1), max(0, y1)
            faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    except Exception:
        return []


# ── 오버레이 그리기 ───────────────────────────────────────────────
def draw_overlay(img, x, y, w, h, emotion, scores):
    color        = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)
    confidence   = float(np.max(scores))
    img_h, img_w = img.shape[:2]
    font         = cv2.FONT_HERSHEY_SIMPLEX

    # 1. 얼굴 박스
    thickness = max(2, img_w // 300)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # 2. 메인 레이블
    label_text = f"{emotion} {int(confidence * 100)}%"
    font_scale = max(0.4, img_w / 1200)
    text_thick = max(1, img_w // 500)

    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, text_thick)
    label_x = x
    label_y = y - th - 8 if y - th - 8 > 0 else y + h + th + 8

    cv2.rectangle(img,
                  (label_x, label_y - th - 4),
                  (label_x + tw + 8, label_y + baseline),
                  color, -1)
    cv2.putText(img, label_text,
                (label_x + 4, label_y - 2),
                font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

    # 3. TOP 3 감정 박스
    score_dict = {e: float(s) for e, s in zip(EMOTIONS, scores)}
    top3 = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:3]

    top3_font_scale = max(0.35, img_w / 1600)
    top3_thickness  = 2
    line_height     = max(18, img_h // 35)
    padding         = 6

    max_tw = max(
        cv2.getTextSize(f"{e}: {int(s * 100)}%", font, top3_font_scale, top3_thickness)[0][0]
        for e, s in top3
    )
    box_x1 = x
    box_y1 = y + h + 4
    box_x2 = x + max_tw + padding * 2 + 60
    box_y2 = box_y1 + line_height * 3 + padding * 2

    if box_y2 > img_h:
        box_y1 = y - h - line_height * 3 - padding * 2 - 4
        box_y2 = box_y1 + line_height * 3 + padding * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

    for i, (emo, score) in enumerate(top3):
        emo_color = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
        line_y    = box_y1 + padding + line_height * i + line_height - 4
        text_str  = f"{emo}: {int(score * 100)}%"

        cv2.putText(img, text_str,
                    (box_x1 + padding, line_y),
                    font, top3_font_scale, (0, 255, 255), top3_thickness, cv2.LINE_AA)

        (etw, _), _ = cv2.getTextSize(text_str, font, top3_font_scale, top3_thickness)
        mini_bar_x  = box_x1 + padding + etw + 6
        mini_bar_y  = line_y - 8
        mini_bar_w  = box_x2 - mini_bar_x - padding
        mini_filled = int(mini_bar_w * score)

        cv2.rectangle(img,
                      (mini_bar_x, mini_bar_y),
                      (mini_bar_x + mini_bar_w, mini_bar_y + 8),
                      (80, 80, 80), -1)
        cv2.rectangle(img,
                      (mini_bar_x, mini_bar_y),
                      (mini_bar_x + max(1, mini_filled), mini_bar_y + 8),
                      emo_color, -1)

    # 4. 상단 감정 텍스트
    top_text  = f"▲ {emotion}"
    top_scale = max(0.5, img_w / 900)
    (ttw, tth), _ = cv2.getTextSize(top_text, font, top_scale, 2)
    cv2.putText(img, top_text,
                ((img_w - ttw) // 2, tth + 10),
                font, top_scale, color, 2, cv2.LINE_AA)

    # 5. Confidence 바 (우측 상단)
    bar_x, bar_y_pos, bar_w, bar_h = img_w - 160, 15, 140, 14
    bar_filled = int(bar_w * confidence)
    cv2.rectangle(img, (bar_x, bar_y_pos), (bar_x + bar_w, bar_y_pos + bar_h), (60, 60, 60), -1)
    cv2.rectangle(img, (bar_x, bar_y_pos), (bar_x + bar_filled, bar_y_pos + bar_h), color, -1)
    cv2.rectangle(img, (bar_x, bar_y_pos), (bar_x + bar_w, bar_y_pos + bar_h), (200, 200, 200), 1)
    cv2.putText(img, f"{int(confidence * 100)}%",
                (bar_x + bar_w + 5, bar_y_pos + bar_h - 2),
                font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ── 메인 ─────────────────────────────────────────────────────────
def visualize_all():
    # 라이브러리 임포트 확인
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    except ImportError:
        print("[ERROR] hsemotion-onnx 가 없습니다.")
        print("  → pip install hsemotion-onnx")
        return

    try:
        from retinaface import RetinaFace
        print("얼굴 감지: RetinaFace ✅")
    except ImportError:
        print("[ERROR] retina-face 가 없습니다.")
        print("  → pip install retina-face")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"모델 로드 중: {MODEL_NAME} ...")
    fer = HSEmotionRecognizer(model_name=MODEL_NAME)
    print("모델 로드 완료!\n")

    valid_ext   = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_ext)
    ])
    total = len(image_files)

    # 체크포인트 복구
    ckpt = load_checkpoint()
    if ckpt:
        processed_set = set(ckpt['processed_files'])
        stats         = ckpt['stats']
        image_files   = [f for f in image_files if f not in processed_set]
        print(f"남은 작업: {len(image_files)}장 / 전체 {total}장\n")
    else:
        processed_set = set()
        stats         = {'success': 0, 'no_face': 0, 'error': 0}
        print(f"총 {total}장 시각화 시작 ({datetime.now().strftime('%H:%M:%S')})\n")

    batch_counter = 0

    for filename in tqdm(image_files, desc="시각화 진행", unit="장"):
        img_path = os.path.join(INPUT_DIR, filename)
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError("이미지 읽기 실패 (손상 또는 미지원 형식)")

            faces   = detect_faces(img_bgr)
            no_face = not faces

            # 얼굴 미감지 → 원본 그대로 저장
            if no_face:
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img_bgr)
                stats['no_face'] += 1
                processed_set.add(filename)
                batch_counter += 1
                continue

            # 가장 큰 얼굴 선택
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_rgb   = cv2.cvtColor(img_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)

            emotion_label, scores = fer.predict_emotions(face_rgb, logits=False)
            result_img = draw_overlay(img_bgr.copy(), x, y, w, h, emotion_label, scores)

            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), result_img)
            stats['success'] += 1
            processed_set.add(filename)

        except Exception as e:
            stats['error'] += 1
            processed_set.add(filename)
            tqdm.write(f"  [에러] {filename}: {e}")

        # 체크포인트 저장
        batch_counter += 1
        if batch_counter % CHECKPOINT_EVERY == 0:
            save_checkpoint(list(processed_set), stats)
            tqdm.write(f"  💾 체크포인트 저장 ({len(processed_set)}/{total}장)")

    # 최종 체크포인트
    save_checkpoint(list(processed_set), stats)

    print(f"""
╔══════════════════════════════════════╗
║         시각화 완료 리포트            ║
╠══════════════════════════════════════╣
║  전체:       {total:>5}장               ║
║  성공:       {stats['success']:>5}장               ║
║  얼굴미감지: {stats['no_face']:>5}장               ║
║  에러:       {stats['error']:>5}장               ║
╚══════════════════════════════════════╝
저장 위치: {OUTPUT_DIR}""")


if __name__ == "__main__":
    visualize_all()
