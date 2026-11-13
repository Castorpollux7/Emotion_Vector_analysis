# predict.py
import cv2
import warnings

from predict_emotion_with_neutral import EmotionPredictor

# 불필요한 경고문구 숨김
warnings.filterwarnings("ignore")

# 1. 모델 로드 (가중치 파일 지정)
model = EmotionPredictor("final_ensemble_v1.pth")

# 2. 원본 이미지 로드 및 전처리
original_image = cv2.imread("test_face.jpg") 
if original_image is None:
    raise FileNotFoundError("test_face.jpg 파일을 찾을 수 없습니다.")

image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 3. 모델 예측 수행 (반환되는 3개 값을 각각 변수에 담습니다)
# 수정된 Predictor.predict()는 [최종감정, 최고확률, 상세리스트]를 반환합니다.
final_class, final_prob, preds = model.predict(image)

# 4. 전체 결과 확인 (상세 리스트 출력)
print("--- 추론 결과 (전체 분포) ---")
for rank, data in enumerate(preds, 1):
    # 이제 preds는 리스트이므로 정상적으로 작동합니다.
    print(f"{rank}위: {data['emotion']:7s} ({data['probability']}%)")

# 5. 최종 결과 출력
# 이미 모델 내부 로직(55% 임계값)이 적용된 final_class를 그대로 씁니다.
print("-" * 25)
print(f"★ 최종 예측 감정: {final_class} ({final_prob:.1f}%)")
