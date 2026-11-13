import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import optuna
from tqdm import tqdm
from sklearn.metrics import f1_score
from emotion_model import EmotionPredictor

# 1. GPU 메모리 제어 (A40 리소스 50% 제한 설정 가능)
# torch.cuda.set_per_process_memory_fraction(0.5) 

def get_gpu_cached_logits(data_dir, weight_path, device='cuda'):
    """모든 이미지의 로짓을 한 번에 추출하여 GPU 메모리에 적재"""
    predictor = EmotionPredictor(ensemble_weight_path=weight_path, device=device)
    target_names = ['anger', 'happy', 'panic', 'sadness', 'neutral']
    
    all_logits = []
    all_labels = []
    label_map = {name: i for i, name in enumerate(target_names)}
    
    print(f"📦 [Step 1] 로짓 추출 시작 (대상: {data_dir})")
    
    for class_name in target_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ 폴더 없음: {class_path}")
            continue
            
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not img_files: continue
        
        for img_name in tqdm(img_files, desc=f"Caching {class_name}"):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = predictor.transform(image=img_rgb)['image'].unsqueeze(0).to(device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # 앙상블 로직 시뮬레이션 (보정치 전 단계의 Raw Logits 추출)
                eff_out = predictor.eff_model(transformed)
                vit_out = predictor.vit_model(F.interpolate(transformed, size=(224, 224), mode='bilinear', align_corners=False))
                
                # 가중치 0.5:0.5 앙상블 로짓
                ensemble_logits = (eff_out * 0.5) + (vit_out * 0.5)
            
            all_logits.append(ensemble_logits) # (1, 4)
            all_labels.append(label_map[class_name])

    # (N, 4) 형태의 텐서로 병합하여 GPU에 고정
    cached_logits = torch.cat(all_logits, dim=0).to(device)
    cached_labels = torch.tensor(all_labels).to(device)
    
    return cached_logits, cached_labels

def objective(trial, logits, labels, device):
    """Optuna 목적 함수: GPU 벡터 연산으로 F1-score 계산"""
    # 탐색 범위 설정
    b_anger = trial.suggest_float('b_anger', -4.0, 4.0)
    b_happy = trial.suggest_float('b_happy', -4.0, 4.0)
    b_panic = trial.suggest_float('b_panic', -4.0, 4.0)
    b_sadness = trial.suggest_float('b_sadness', -4.0, 4.0)
    threshold = trial.suggest_float('threshold', 40.0, 95.0)
    
    biases = torch.tensor([b_anger, b_happy, b_panic, b_sadness]).to(device)
    
    # 1. 모든 데이터에 보정치 한 번에 적용 (Vectorized Add)
    adjusted_logits = logits + biases  # (N, 4)
    
    # 2. Softmax 및 확률 계산 (Vectorized Softmax)
    probs = F.softmax(adjusted_logits, dim=1) * 100
    
    # 3. 예측값 결정
    max_probs, pred_indices = torch.max(probs, dim=1)
    
    # 임계값 미만은 4(neutral)로 할당
    # labels 리스트: ['anger', 'happy', 'panic', 'sadness', 'neutral'] 이므로 neutral은 인덱스 4
    final_preds = torch.where(max_probs < threshold, 
                              torch.tensor(4).to(device), 
                              pred_indices)
    
    # 4. 성능 평가 (F1-score 계산을 위해 CPU로 이동)
    y_true = labels.cpu().numpy()
    y_pred = final_preds.cpu().numpy()
    
    # Macro F1-score (5개 클래스 균형 성능)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

if __name__ == "__main__":
    # 환경 설정
    VAL_DATA_DIR = "/workspace/user1/dataset_v2/img/val" 
    WEIGHT_PATH = "final_ensemble_v1.pth"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 데이터 로드 및 캐싱
    logits_cache, labels_cache = get_gpu_cached_logits(VAL_DATA_DIR, WEIGHT_PATH, DEVICE)
    
    # 2. Optuna 최적화 실행
    study = optuna.create_study(direction='maximize')
    
    print(f"\n🚀 [Step 2] GPU 병렬 최적화 시작 (1,000회 시도)...")
    # A40에서는 1,000회도 매우 빠르게 끝납니다.
    study.optimize(lambda trial: objective(trial, logits_cache, labels_cache, DEVICE), n_trials=1000)
    
    # 3. 결과 출력
    print("\n" + "="*50)
    print("🏆 최적화 결과")
    print(f"최고 Macro F1: {study.best_value:.4f}")
    print("\n최적 보정치(Bias) & 임계값:")
    best_params = study.best_params
    print(f"  - Anger 보정: {best_params['b_anger']:.3f}")
    print(f"  - Happy 보정: {best_params['b_happy']:.3f}")
    print(f"  - Panic 보정: {best_params['b_panic']:.3f}")
    print(f"  - Sadness 보정: {best_params['b_sadness']:.3f}")
    print(f"  - Threshold: {best_params['threshold']:.2f}%")
    print("="*50)
    
    print("\n💡 위 수치들을 'emotion_model.py'의 보정 로직에 업데이트하세요.")