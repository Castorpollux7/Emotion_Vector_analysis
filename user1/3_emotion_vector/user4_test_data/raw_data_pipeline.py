import os
import glob
import cv2
import torch
import pandas as pd
from tqdm import tqdm

# 기존 모듈 임포트
from emotion_model_02 import EmotionPredictor
from complex_emotion import EmotionAnchorSystem

def is_valid_face(face_image, min_size=48, blur_threshold=80.0):
    """
    크롭된 얼굴 이미지의 유효성을 검사하는 필터링 함수
    """
    h, w = face_image.shape[:2]
    if h < min_size or w < min_size:
        return False, "Too_Small"
    
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < blur_threshold:
        return False, f"Blurry_{fm:.1f}"
        
    return True, "Valid"

def run_raw_data_pipeline():
    # 1. 경로 설정
    SOURCE_BASE_DIR = "/workspace/user4/Sample/01.원천데이터"
    CROP_SAVE_DIR = "/workspace/user4/Sample/02.Cropped_Faces"
    EXCEL_PATH = "감정조합임시_v3.xlsx"
    MODEL_PATH = "final_ensemble_v1.pth"
    OUTPUT_CSV = "raw_data_complex_emotion_results.csv"

    os.makedirs(CROP_SAVE_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 모듈 초기화
    predictor = EmotionPredictor(MODEL_PATH)
    emotion_system = EmotionAnchorSystem(EXCEL_PATH, expansion_k=0.3, device=device)
    
    # OpenCV 기본 얼굴 탐지기 로드 (설치 필요 없음)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    target_emotions = ['Anger', 'Happy', 'Panic', 'Sadness']
    results = []

    # 3. 원천 이미지 목록 로드
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        image_paths.extend(glob.glob(os.path.join(SOURCE_BASE_DIR, "**", ext), recursive=True))
    
    print(f"총 {len(image_paths)}장의 이미지를 발견했습니다. OpenCV 엔진으로 탐지를 시작합니다.")

    # 4. 처리 루프
    for img_path in tqdm(image_paths, desc="이미지 처리 중"):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
            
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 탐지 (scaleFactor와 minNeighbors로 정밀도 조절)
        # 관객석처럼 얼굴이 작을 수 있으므로 minSize를 30x30으로 하향 조정
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for face_idx, (x, y, w, h) in enumerate(faces):
            # 좌표 추출 및 여유분(Padding) 추가 (얼굴 주변을 조금 더 넓게 잡음)
            padding = int(w * 0.1)
            y1, y2 = max(0, y - padding), min(image_bgr.shape[0], y + h + padding)
            x1, x2 = max(0, x - padding), min(image_bgr.shape[1], x + w + padding)
            
            face_img_bgr = image_bgr[y1:y2, x1:x2]
            
            # 품질 필터링
            is_valid, reason = is_valid_face(face_img_bgr)
            if not is_valid: continue
                
            # 유효 얼굴 저장
            folder_name = os.path.basename(os.path.dirname(img_path))
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            face_save_name = f"{folder_name}_{file_name}_face{face_idx}.jpg"
            cv2.imwrite(os.path.join(CROP_SAVE_DIR, face_save_name), face_img_bgr)

            # 감정 추론
            face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
            preds = predictor.predict(face_img_rgb)
            
            prob_dict = {item['emotion']: item['probability'] / 100.0 for item in preds}
            probs_array = [[prob_dict.get(emo, 0.0) for emo in target_emotions + ['Neutral']]]
            probs_tensor = torch.tensor(probs_array, dtype=torch.float32, device=device)

            final_labels, distances = emotion_system.classify(probs_tensor)

            results.append({
                'source_folder': folder_name,
                'original_image': os.path.basename(img_path),
                'complex_label': final_labels[0],
                'distance': round(distances[0].item(), 4),
                'Anger_prob': prob_dict.get('Anger', 0.0),
                'Happy_prob': prob_dict.get('Happy', 0.0),
                'Panic_prob': prob_dict.get('Panic', 0.0),
                'Sadness_prob': prob_dict.get('Sadness', 0.0)
            })

    # 5. 결과 저장
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n[성공] {len(results)}개의 얼굴 분석 완료. 결과: {OUTPUT_CSV}")
    else:
        print("\n[알림] 분석된 얼굴이 없습니다. 탐지 임계값이나 데이터 경로를 확인하세요.")

if __name__ == "__main__":
    run_raw_data_pipeline()