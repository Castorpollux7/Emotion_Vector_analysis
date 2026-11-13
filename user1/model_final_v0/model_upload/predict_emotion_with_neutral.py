import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

# ==========================================
# 1. 모델 클래스 정의 (기존과 동일)
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine * self.s

class EffNetArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0)
        self.arcface = ArcMarginProduct(in_features=1408, out_features=num_classes)
    def forward(self, x): return self.arcface(self.backbone(x))

class ViTArcFace(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.arcface = ArcMarginProduct(in_features=self.backbone.num_features, out_features=num_classes)
    def forward(self, x): return self.arcface(self.backbone(x))

# ==========================================
# 2. 통합 예측 클래스 (Neutral 임계값 로직 추가됨)
# ==========================================
class EmotionPredictor:
    def __init__(self, ensemble_weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.idx_to_emotion = {0: 'Anger', 1: 'Happy', 2: 'Panic', 3: 'Sadness'}
        
        # [설정값] Neutral 판별을 위한 임계값 (예: 55%)
        self.neutral_threshold = 55.0 
        
        self.transform = A.Compose([
            A.Resize(260, 260),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # 모델 생성 및 가중치 로드
        print("🧠 앙상블 모델 로딩 중...")
        self.eff_model = EffNetArcFace().to(self.device)
        self.vit_model = ViTArcFace().to(self.device)
        
        checkpoint = torch.load(ensemble_weight_path, map_location=self.device, weights_only=False)
        self.eff_model.load_state_dict(checkpoint['eff_state_dict'])
        self.vit_model.load_state_dict(checkpoint['vit_state_dict'])
        
        self.eff_model.eval()
        self.vit_model.eval()
        print("✅ 모델 로딩 완료!")
        
    def predict(self, image_rgb, temperature=20.0):
        transformed = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)
        
        # 4-Crop TTA
        ttas = [
            transformed,
            torch.flip(transformed, dims=[3]),
            F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)[..., 13:273, 13:273],
            torch.flip(F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)[..., 13:273, 13:273], dims=[3])
        ]
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            eff_out, vit_out = 0, 0
            for img in ttas:
                eff_out += self.eff_model(img)
                vit_out += self.vit_model(F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False))
            
            eff_logits = (eff_out / 4).cpu().numpy()
            vit_logits = (vit_out / 4).cpu().numpy()
            
        # 앙상블 및 사후 교정
        ensemble_logits = (eff_logits * 0.5) + (vit_logits * 0.5)
        ensemble_logits[:, 0] -= 2.1
        ensemble_logits[:, 2] += 1.05
        ensemble_logits[:, 3] += 0.42
        
        # 확률 변환
        logits_tensor = torch.tensor(ensemble_logits[0]) / temperature
        probs = F.softmax(logits_tensor, dim=0).numpy() * 100
        
        # --- [추가된 로직] Neutral 임계값 판별 ---
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        
        if max_prob < self.neutral_threshold:
            final_emotion = "Neutral"
        else:
            final_emotion = self.idx_to_emotion[max_idx]
            
        # 상세 결과 리스트 (상위 3개)
        detail_results = [{'emotion': self.idx_to_emotion[idx], 'probability': round(probs[idx], 2)} 
                          for idx in np.argsort(probs)[::-1][:3]]
                          
        return final_emotion, max_prob, detail_results

# ==========================================
# 3. 메인 실행부
# ==========================================
if __name__ == "__main__":
    WEIGHT_FILE_PATH = "final_ensemble_v1.pth" 
    TEST_IMAGE_PATH = "/workspace/user1/model_final_v0/model_upload/test_face_02.jpg" 

    try:
        # 1. 예측기 인스턴스 생성 (모델 로드 포함)
        predictor = EmotionPredictor(WEIGHT_FILE_PATH)
        
        # 2. 이미지 로드 (RGB 변환까지만 수행, 나머지는 Predictor 내부에서 처리)
        img = cv2.imread(TEST_IMAGE_PATH)
        if img is None:
            raise ValueError("이미지 경로를 확인하세요.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. 예측 실행
        emotion, score, details = predictor.predict(img_rgb)
        
        # 4. 결과 출력
        print("\n" + "="*40)
        print(f"📸 대상 이미지: {os.path.basename(TEST_IMAGE_PATH)}")
        print(f"🎯 최종 예측 감정: {emotion} (확신도: {score:.1f}%)")
        print("-" * 40)
        print("📊 상세 확률 분포 (Top 3):")
        for d in details:
            print(f"  - {d['emotion']}: {d['probability']}%")
        print("="*40)
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")