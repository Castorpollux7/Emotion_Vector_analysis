# predict.py
import cv2
import warnings
from emotion_model import EmotionPredictor

# 불필요한 경고문구 숨김
warnings.filterwarnings("ignore")

# 1. 모델 로드 (가중치 파일 지정)
model = EmotionPredictor("/workspace/user1/model_final_v0/model_upload/final_ensemble_v1.pth")

# 2. 원본 이미지 로드 및 전처리
original_image = cv2.imread("test_face.jpg") 
if original_image is None:
    raise FileNotFoundError("test_face.jpg 파일을 찾을 수 없습니다.")

image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 3. 모델 예측 수행 (리스트 1개 반환)
preds = model.predict(image)

# 4. 전체 Top-3 결과 확인
print("--- 추론 결과 (Top 3) ---")
for rank, data in enumerate(preds, 1):
    print(f"{rank}위: {data['emotion']:7s} ({data['probability']}%)")

# 5. 최종 예측 감정 결정 (Neutral 임계값 55% 적용)
NEUTRAL_THRESHOLD = 55.0
top_emotion = preds[0]['emotion']
top_prob = preds[0]['probability']

if top_prob < NEUTRAL_THRESHOLD:
    final_class = "Neutral"
else:
    final_class = top_emotion

print("-" * 25)
print(f"★ 최종 예측 감정: {final_class} ({top_prob:.1f}%)")
