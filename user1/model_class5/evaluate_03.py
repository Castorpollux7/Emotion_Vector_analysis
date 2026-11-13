import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from emotion_model_03 import EmotionPredictor

def evaluate_model(data_dir, weight_path, threshold=90.16):
    # 1. 모델 초기화
    predictor = EmotionPredictor(ensemble_weight_path=weight_path)
    
    # 2. 클래스 정의 (모두 소문자로 통일)
    target_names = ['anger', 'happy', 'panic', 'sadness', 'neutral']
    
    y_true = []
    y_pred = []

    print(f"🚀 소문자 기반 검증 시작 (Path: {data_dir} / Threshold: {threshold}%)")
    
    for class_name in target_names:
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
            
            # 3. 예측 수행 (여기서 threshold와 temperature 제어)
            # temperature=1.0으로 설정하여 확률 뭉개짐 방지
            result = predictor.predict(img_rgb, threshold=threshold, temperature=1.0)            
            
            # 반환 형식 처리
            if isinstance(result, tuple):
                pred_name = result[0]
            elif isinstance(result, list):
                pred_name = result[0]['emotion']
            else:
                pred_name = str(result)
            
            y_true.append(class_name)
            y_pred.append(pred_name.lower())
            
    if not y_true:
        print("❌ 분석할 데이터를 찾지 못했습니다. 경로와 폴더명을 확인하세요.")
        return

    # 4. 결과 리포트 출력
    print("\n" + "="*60)
    print(f"📋 5-Class Classification Report (Threshold: {threshold}%)")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=target_names, zero_division=0))

    # 5. 혼동 행렬 시각화
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Neutral Threshold: {threshold}%)')
    
    plt.savefig('evaluation_result.png')
    print("\n📊 혼동 행렬이 'evaluation_result.png'로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    # 설정값
    VAL_DATA_DIR = "/workspace/user1/dataset_v2/img/val" 
    WEIGHT_PATH = "final_ensemble_v1.pth"
    
    # ★ 여기서 값을 조절하여 실험하세요 ★
    MY_THRESHOLD = 55.0 
    
    evaluate_model(VAL_DATA_DIR, WEIGHT_PATH, threshold=MY_THRESHOLD)