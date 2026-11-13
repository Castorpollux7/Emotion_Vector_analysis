# predict.py
import cv2
from user1.model_final_v0.model_upload.emotion_model import EmotionPredictor

# 1. 모델 로드 (Keras의 load_model 느낌)
model = EmotionPredictor("/workspace/user1/model_final_v0/movie_rating_v1.pth")

# 2. 이미지 준비 및 예측 (Keras의 model.predict 느낌)
image = cv2.cvtColor(cv2.imread("/workspace/데이터셋/img/train/sadness/zsybc56c3416b8f95c895f45517e816651235cf4b3635482e8e6e8647f578gr7z.jpg"), cv2.COLOR_BGR2RGB)
preds = model.predict(image)

print(preds)
# 출력: [{'emotion': 'Panic', 'probability': 85.2}, {'emotion': 'Anger', 'probability': 12.1}, ...]