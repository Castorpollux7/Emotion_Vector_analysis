import os
import cv2
import pandas as pd
import torch
from tqdm import tqdm
from emotion_model_02 import EmotionPredictor
from complex_emotion import EmotionAnchorSystem

def run_batch_inference():
    # 1. 경로 설정 (새로 추출한 순정 얼굴 데이터 폴더)
    SOURCE_DIR = "/workspace/user1/dataset/open_images_faces/" # crop된 안면데이터 경로
    EXCEL_PATH = "감정조합임시_v3.xlsx"  # 실제 사용 중인 엑셀 파일명으로 확인
    MODEL_PATH = "final_ensemble_v1.pth"
    # OUTPUT_CSV = "complex_emotion_statistics.csv" # temperature 20.0
    # OUTPUT_CSV = "complex_emotion_statistics_02.csv" # temperature 2.0
    # OUTPUT_CSV = "complex_emotion_statistics_03.csv" # 경외 비율 변경(happy-panic) 7:3 --> 6:4
    OUTPUT_CSV = "complex_emotion_statistics_04.csv" # 편안 비율 변경(happy-neutral) 2:8 --> 3.5:6.5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 시스템 초기화
    predictor = EmotionPredictor(MODEL_PATH)
    emotion_system = EmotionAnchorSystem(EXCEL_PATH, expansion_k=0.3, device=device)
    
    # 4가지 기본 감정 타겟
    target_emotions = ['Anger', 'Happy', 'Panic', 'Sadness']
    results = []

    # 3. 이미지 파일 목록 로드
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.jpg')]
    print(f"총 {len(image_files)}장의 얼굴 이미지에 대해 추론을 시작합니다...")

    # 4. 일괄 추론 루프
    for img_name in tqdm(image_files, desc="복합 감정 분류 진행률"):
        img_path = os.path.join(SOURCE_DIR, img_name)
        image_rgb = cv2.imread(img_path)
        if image_rgb is None: continue
        
        # BGR -> RGB 변환 (모델 입력용)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # 기존 모델 추론 (Top-N 리스트 반환)
        preds = predictor.predict(image_rgb)
        
        # 확률 벡터 재조립 (5차원으로 맞춤)
        prob_dict = {item['emotion']: item['probability'] / 100.0 for item in preds}
        probs_array = [[prob_dict.get(emo, 0.0) for emo in target_emotions + ['Neutral']]]
        probs_tensor = torch.tensor(probs_array, dtype=torch.float32, device=device)

        # 복합 감정 엔진 통과
        final_labels, distances = emotion_system.classify(probs_tensor)

        # 데이터 저장 (사후 분석을 위해 원시 확률도 함께 저장)
        results.append({
            'filename': img_name,
            'complex_label': final_labels[0],
            'distance': round(distances[0].item(), 4),
            'Anger_prob': prob_dict.get('Anger', 0.0),
            'Happy_prob': prob_dict.get('Happy', 0.0),
            'Panic_prob': prob_dict.get('Panic', 0.0),
            'Sadness_prob': prob_dict.get('Sadness', 0.0)
        })

    # 5. CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='cp949')
    print("-" * 30)
    print(f"★ 추론 완료. 전체 통계 데이터가 '{OUTPUT_CSV}'에 저장되었습니다.")

if __name__ == "__main__":
    run_batch_inference()