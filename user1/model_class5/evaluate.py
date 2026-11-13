import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from user1.model_class5.emotion_model import EmotionPredictor

def evaluate_model(data_dir, weight_path, threshold=55.0): # 0이면 4감정 분류
    # 1. 모델 초기화
    predictor = EmotionPredictor(ensemble_weight_path=weight_path)
    
    # 2. 클래스 정의 (모두 소문자로 통일)
    target_names = ['anger', 'happy', 'panic', 'sadness', 'neutral']
    
    y_true = []
    y_pred = []

    print(f"🚀 소문자 기반 검증 시작 (Path: {data_dir})")
    
    for class_name in target_names:
        # 소문자 폴더 경로 설정
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️ 폴더를 찾을 수 없습니다: {class_path}")
            continue
            
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            continue
            
        print(f"📦 '{class_name}' 클래스 처리 중... ({len(image_files)}개)")
        
        for img_name in tqdm(image_files, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 3. 예측 수행
            result = predictor.predict(img_rgb, threshold=threshold)            
            # 반환 형식 처리 (문자열만 추출)
            if isinstance(result, tuple):
                pred_name = result[0]
            elif isinstance(result, list):
                pred_name = result[0]['emotion']
            else:
                pred_name = str(result)
            
            # --- 핵심: 모델 출력값을 소문자로 강제 변환하여 폴더명과 맞춤 ---
            y_true.append(class_name)        # 이미 소문자 ('anger')
            y_pred.append(pred_name.lower()) # 대문자 'Anger' 등을 'anger'로 변환
            
    if not y_true:
        print("❌ 분석할 데이터를 찾지 못했습니다. 경로와 폴더명을 확인하세요.")
        return

    # 4. 결과 리포트 출력
    print("\n" + "="*60)
    print("📋 5-Class Classification Report (Lowercase Match)")
    print("="*60)
    # labels를 지정하여 target_names 순서(anger~neutral)대로 출력
    print(classification_report(y_true, y_pred, labels=target_names, zero_division=0))

    # 5. 혼동 행렬 시각화
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Neutral Threshold: {threshold}%)')
    
    # 서버 환경에서 창이 안 뜰 수 있으므로 파일로 저장
    plt.savefig('evaluation_result.png')
    print("\n📊 혼동 행렬이 'evaluation_result.png'로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    # 실제 경로로 수정하세요
    VAL_DATA_DIR = "/workspace/user1/dataset_v2/img/val" 
    WEIGHT_PATH = "final_ensemble_v1.pth"
    
    evaluate_model(VAL_DATA_DIR, WEIGHT_PATH, threshold=55.0)