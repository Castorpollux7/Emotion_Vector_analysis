import torch
import torch.nn.functional as F
import optuna
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from emotion_model import EmotionPredictor

# 1. GPU로 모든 로짓 캐싱
def get_gpu_cached_logits(data_dir, weight_path, device='cuda'):
    predictor = EmotionPredictor(ensemble_weight_path=weight_path, device=device)
    target_names = ['anger', 'happy', 'panic', 'sadness', 'neutral']
    
    all_logits = []
    all_labels = []
    
    label_map = {name: i for i, name in enumerate(target_names)}
    
    print("📦 [Step 1] 로짓 추출 및 GPU 메모리 적재 중...")
    # A40의 성능을 활용해 배치 처리를 하면 좋지만, 구조상 이미지별 추출 후 스택
    for class_name in target_names:
        # ... (기존 폴더 순회 및 이미지 로드 로직 동일) ...
        # (생략: 이전 답변의 이미지 로드 부분 참고)
        
        # 추출된 로짓을 리스트에 담기
        # logits_tensor.shape = (1, 4)
        all_logits.append(logits_tensor) 
        all_labels.append(label_map[class_name])

    # 모든 로짓을 하나의 큰 텐서로 결합 (N, 4)
    cached_logits = torch.cat(all_logits, dim=0).to(device)
    cached_labels = torch.tensor(all_labels).to(device)
    
    return cached_logits, cached_labels

# 2. 벡터화된 목적 함수
def objective(trial, logits, labels, device):
    # 탐색 범위 (-4.0 ~ 4.0)
    biases = torch.tensor([
        trial.suggest_float('b_anger', -4.0, 4.0),
        trial.suggest_float('b_happy', -4.0, 4.0),
        trial.suggest_float('b_panic', -4.0, 4.0),
        trial.suggest_float('b_sadness', -4.0, 4.0)
    ]).to(device)
    
    threshold = trial.suggest_float('threshold', 40.0, 95.0)

    # [핵심] GPU에서 모든 이미지에 대해 한 번에 보정치 더하기
    adjusted_logits = logits + biases  # (N, 4)
    
    # Softmax 계산 (병렬)
    probs = F.softmax(adjusted_logits, dim=1) * 100
    
    # 최대 확률과 해당 인덱스 추출
    max_probs, pred_indices = torch.max(probs, dim=1)
    
    # 임계값 적용: threshold 미만은 4(neutral)로 할당
    final_preds = torch.where(max_probs < threshold, 
                              torch.tensor(4).to(device), 
                              pred_indices)

    # F1-score 계산을 위해 CPU로 이동
    y_true = labels.cpu().numpy()
    y_pred = final_preds.cpu().numpy()

    return f1_score(y_true, y_pred, average='macro')

# 메인 실행부에서 500~1000회 trial 수행