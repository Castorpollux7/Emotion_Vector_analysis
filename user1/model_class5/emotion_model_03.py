import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

# --- 1. 모델 아키텍처 정의 ---
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

# --- 2. 모델 래퍼 클래스 ---
class EmotionPredictor:
    def __init__(self, ensemble_weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.idx_to_emotion = {0: 'Anger', 1: 'Happy', 2: 'Panic', 3: 'Sadness'}
        self.transform = A.Compose([
            A.Resize(260, 260),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.eff_model = EffNetArcFace().to(self.device)
        self.vit_model = ViTArcFace().to(self.device)
        
        checkpoint = torch.load(ensemble_weight_path, map_location=self.device)
        self.eff_model.load_state_dict(checkpoint['eff_state_dict'])
        self.vit_model.load_state_dict(checkpoint['vit_state_dict'])
        
        self.eff_model.eval()
        self.vit_model.eval()
        
    # emotion_model.py 파일 내부

    def predict(self, image_rgb, temperature=1.0, threshold=55.0): # <- 인자(parameter) 추가
        # 1. 4-Crop TTA 전처리 (기존 로직)
        transformed = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)
        
        ttas = [
            transformed,
            torch.flip(transformed, dims=[3]),
            F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)[..., 13:273, 13:273],
            torch.flip(F.interpolate(transformed, scale_factor=1.1, mode='bilinear', align_corners=False)[..., 13:273, 13:273], dims=[3])
        ]
        
        # 2. 모델 추론
        with torch.no_grad(), torch.amp.autocast('cuda'):
            eff_out, vit_out = 0, 0
            for img in ttas:
                eff_out += self.eff_model(img)
                vit_out += self.vit_model(F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False))
            
            eff_logits = (eff_out / 4).cpu().numpy()
            vit_logits = (vit_out / 4).cpu().numpy()
            
        # 3. 앙상블 및 편향 보정
        ensemble_logits = (eff_logits * 0.5) + (vit_logits * 0.5)
        ensemble_logits[:, 0] -= 2.136  # Anger
        ensemble_logits[:, 1] -= 0.893   # Happy
        ensemble_logits[:, 2] += 2.826 # Panic
        ensemble_logits[:, 3] -= 1.013 # Sadness
        
        # 4. 확률 계산 (Temperature 적용)
        logits_tensor = torch.tensor(ensemble_logits[0]) / temperature
        probs = F.softmax(logits_tensor, dim=0).numpy() * 100
        
        # 5. 최종 결과 결정 (Threshold 로직 적용)
        top_idx = np.argmax(probs)
        max_prob = probs[top_idx]
        
        # 임계값(threshold)보다 낮으면 'neutral'로 판정
        if max_prob < threshold:
            final_emotion = 'neutral'
        else:
            # evaluate.py의 소문자 target_names와 맞추기 위해 .lower() 적용
            final_emotion = self.idx_to_emotion[top_idx].lower()
            
        # 상세 결과 리스트 생성
        detail_results = [{'emotion': self.idx_to_emotion[idx].lower(), 'probability': round(probs[idx], 2)} 
                        for idx in np.argsort(probs)[::-1]]
        
        # (최종 감정, 최대 확률, 상세 리스트) 형태로 반환
        return final_emotion, max_prob, detail_results