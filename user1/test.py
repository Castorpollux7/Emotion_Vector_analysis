import fiftyone as fo
import pandas as pd
import os
from tqdm import tqdm

# 1. 경로 설정
CSV_PATH = "complex_emotion_statistics_02.csv"
IMAGE_DIR = "/workspace/user1/dataset/open_images_faces/"
DATASET_NAME = "emotion_complex_v2_check"

# 2. 데이터 로드 (UTF-8-SIG는 한글 깨짐 방지용)
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
except:
    df = pd.read_csv(CSV_PATH, encoding='cp949')

# 3. FiftyOne 데이터셋 생성
if DATASET_NAME in fo.list_datasets():
    fo.delete_dataset(DATASET_NAME)
dataset = fo.Dataset(DATASET_NAME)

# 4. 샘플 생성 및 필드 추가
samples = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="FiftyOne 매칭 중"):
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    if not os.path.exists(img_path): continue
    
    sample = fo.Sample(filepath=img_path)
    
    # 복합 감정 라벨 추가
    sample["complex_emotion"] = fo.Classification(
        label=row['complex_label'],
        confidence=float(1 - row['distance']) # 거리가 가까울수록 높은 확신도
    )
    
    # 메타데이터: 거리 및 기본 감정 확률
    sample["distance"] = float(row['distance'])
    sample["anger_p"] = float(row['Anger_prob'])
    sample["happy_p"] = float(row['Happy_prob'])
    sample["panic_p"] = float(row['Panic_prob'])
    sample["sadness_p"] = float(row['Sadness_prob'])
    
    samples.append(sample)

dataset.add_samples(samples)
dataset.persistent = True

# 5. 대시보드 실행 (서버 환경일 경우 포트 포워딩 필요)
session = fo.launch_app(dataset, remote=True, port=5151)
print(f"FiftyOne 세션이 포트 5151에서 실행 중입니다.")