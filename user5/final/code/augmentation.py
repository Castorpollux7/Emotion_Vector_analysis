"""
Step 2: 극장 환경 시뮬레이션 Augmentation
==========================================
목적:
    - 저조도/저화질 극장 환경을 학습 데이터에 반영
    - 모델이 악조건에서도 감정 특징을 추출하도록 강건성 확보
    - 원본 대비 N배 데이터 증강

설치:
    pip install albumentations opencv-python tqdm numpy Pillow

실행:
    python augmentation.py

결과 구조:
    /workspace/user5/img_augmented/
    ├── train/
    │   ├── anger/      ← 원본 + 증강 이미지
    │   ├── blank/
    │   ├── happy/
    │   ├── panic/
    │   └── sadness/
    └── val/
        └── (동일 구조, val은 증강 없이 원본만)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ================= 설정 =================
INPUT_DIR   = "/workspace/user5/img/img(classify)"    # 원본 이미지
OUTPUT_DIR  = "/workspace/user5/img_augmented"         # 증강 결과 저장
AUG_COUNT   = 3      # 원본 1장당 증강 이미지 몇 장 생성 (3 = 4배 증가)
SEED        = 42     # 재현성을 위한 랜덤 시드

# 클래스 목록
EMOTIONS    = ["anger", "blank", "happy", "panic", "sadness"]
# ========================================

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════
# 극장 환경 Augmentation 함수들
# ══════════════════════════════════════════════════════════════

def apply_low_light(img, gamma_range=(0.3, 0.6)):
    """
    저조도 시뮬레이션 - 감마 보정으로 어둡게
    gamma < 1.0 : 어두워짐 (극장 환경)
    """
    gamma = np.random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_gaussian_noise(img, var_range=(10, 40)):
    """가우시안 노이즈 추가 - 카메라 센서 노이즈 시뮬레이션"""
    var   = np.random.uniform(*var_range)
    noise = np.random.normal(0, var ** 0.5, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_blur(img):
    """
    블러 적용 - 초점 흐림 또는 움직임 블러
    극장에서 멀리 있는 관객 얼굴의 뭉개짐 시뮬레이션
    """
    blur_type = np.random.choice(["gaussian", "motion"])

    if blur_type == "gaussian":
        # 가우시안 블러 (초점 흐림)
        ksize = np.random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        # 모션 블러 (움직임 흔들림)
        ksize  = np.random.choice([5, 7, 9])
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = kernel / ksize
        return cv2.filter2D(img, -1, kernel)


def apply_compression(img, quality_range=(20, 50)):
    """
    JPEG 압축 노이즈 - 저화질 카메라 시뮬레이션
    quality 낮을수록 화질 저하
    """
    quality = int(np.random.uniform(*quality_range))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded   = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_horizontal_flip(img):
    """좌우 반전 - 관객 위치 다양성"""
    return cv2.flip(img, 1)


def apply_brightness_contrast(img, brightness_range=(-40, 40), contrast_range=(0.7, 1.3)):
    """밝기/대비 조정 - 스크린 반사광 시뮬레이션"""
    brightness = np.random.uniform(*brightness_range)
    contrast   = np.random.uniform(*contrast_range)
    result     = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255)
    return result.astype(np.uint8)


def apply_color_shift(img, shift_range=(-20, 20)):
    """
    색조 변화 - 스크린 빛에 의한 피부색 변화 시뮬레이션
    (파란빛/붉은빛 등)
    """
    shift = np.random.uniform(*shift_range)
    channel = np.random.randint(0, 3)  # B, G, R 중 하나
    result  = img.copy().astype(np.float32)
    result[:, :, channel] = np.clip(result[:, :, channel] + shift, 0, 255)
    return result.astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# 극장 환경 Augmentation 파이프라인
# ══════════════════════════════════════════════════════════════

# 3가지 시나리오 정의
SCENARIOS = {
    "dark":       # 어두운 극장 - 가장 흔한 케이스
        [apply_low_light, apply_gaussian_noise, apply_blur],

    "glare":      # 스크린 반사광
        [apply_brightness_contrast, apply_color_shift, apply_gaussian_noise],

    "lowquality": # 저화질 카메라
        [apply_compression, apply_blur, apply_gaussian_noise],
}


def augment_image(img):
    """
    랜덤하게 시나리오 선택 후 Augmentation 적용
    + 50% 확률로 좌우 반전 추가
    """
    # 시나리오 선택
    scenario_name = np.random.choice(list(SCENARIOS.keys()))
    transforms    = SCENARIOS[scenario_name]

    result = img.copy()

    # 각 변환 적용 (70% 확률로 각 변환 적용)
    for transform in transforms:
        if np.random.random() < 0.7:
            try:
                result = transform(result)
            except Exception:
                pass

    # 50% 확률로 좌우 반전
    if np.random.random() < 0.5:
        result = apply_horizontal_flip(result)

    return result, scenario_name


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

def run_augmentation():
    print("=" * 55)
    print("Step 2: 극장 환경 시뮬레이션 Augmentation")
    print("=" * 55)
    print(f"입력 경로: {INPUT_DIR}")
    print(f"출력 경로: {OUTPUT_DIR}")
    print(f"증강 배수: 원본 1장 + 증강 {AUG_COUNT}장 = {AUG_COUNT + 1}배\n")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ 입력 폴더 없음: {INPUT_DIR}")
        return

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    stats     = {}

    for split in ["train", "val"]:
        split_dir = os.path.join(INPUT_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"⚠️  {split} 폴더 없음: {split_dir}")
            continue

        print(f"\n━━━ {split.upper()} 처리 중 ━━━")

        for emotion in EMOTIONS:
            src_folder  = os.path.join(split_dir, emotion)
            dest_folder = os.path.join(OUTPUT_DIR, split, emotion)

            if not os.path.isdir(src_folder):
                print(f"  ⚠️  {emotion} 폴더 없음")
                continue

            os.makedirs(dest_folder, exist_ok=True)

            image_files = [f for f in os.listdir(src_folder)
                           if f.lower().endswith(valid_ext)]

            orig_count = 0
            aug_count  = 0

            for filename in tqdm(image_files,
                                 desc=f"  {split}/{emotion}", unit="장", leave=False):
                src_path = os.path.join(src_folder, filename)
                img      = cv2.imread(src_path)

                if img is None:
                    continue

                # 원본 복사
                dest_orig = os.path.join(dest_folder, filename)
                cv2.imwrite(dest_orig, img)
                orig_count += 1

                # val은 증강 없이 원본만 저장
                if split == "val":
                    continue

                # train만 증강
                name, ext = os.path.splitext(filename)
                for i in range(AUG_COUNT):
                    aug_img, scenario = augment_image(img)
                    aug_filename      = f"{name}_aug{i+1}_{scenario}{ext}"
                    dest_aug          = os.path.join(dest_folder, aug_filename)
                    cv2.imwrite(dest_aug, aug_img)
                    aug_count += 1

            total = orig_count + aug_count
            stats[f"{split}/{emotion}"] = {
                "original":  orig_count,
                "augmented": aug_count,
                "total":     total
            }
            print(f"  {split}/{emotion:<10}: 원본 {orig_count:>4}장 + 증강 {aug_count:>5}장 = {total:>5}장")

    # 최종 리포트
    total_orig = sum(s["original"]  for s in stats.values())
    total_aug  = sum(s["augmented"] for s in stats.values())
    total_all  = sum(s["total"]     for s in stats.values())

    print(f"""
╔══════════════════════════════════════════════════╗
║         Augmentation 완료 리포트                  ║
╠══════════════════════════════════════════════════╣
║  원본 이미지:    {total_orig:>6}장                   ║
║  증강 이미지:    {total_aug:>6}장                   ║
║  전체 이미지:    {total_all:>6}장                   ║
║  증강 배수:      {total_all/max(total_orig,1):>5.1f}배                    ║
╠══════════════════════════════════════════════════╣
║  적용된 시나리오                                   ║
║  - dark:       저조도 + 노이즈 + 블러              ║
║  - glare:      스크린 반사광 + 색조 변화           ║
║  - lowquality: JPEG 압축 + 블러 + 노이즈          ║
╚══════════════════════════════════════════════════╝
저장 위치: {OUTPUT_DIR}
""")

    # shared 폴더로 복사 안내
    print("팀원 공유하려면:")
    print(f"  cp -r {OUTPUT_DIR} /workspace/shared/")


# ══════════════════════════════════════════════════════════════
# 증강 결과 샘플 시각화 (선택)
# ══════════════════════════════════════════════════════════════

def preview_augmentation(sample_img_path, save_path=None):
    """
    증강 결과를 그리드 이미지로 시각화
    sample_img_path: 테스트할 이미지 경로
    save_path: 저장 경로 (None이면 현재 폴더)
    """
    img = cv2.imread(sample_img_path)
    if img is None:
        print(f"❌ 이미지 없음: {sample_img_path}")
        return

    results    = [("원본", img.copy())]
    scenarios  = list(SCENARIOS.keys()) * 2  # 각 시나리오 2번씩

    for scenario_name in scenarios:
        transforms = SCENARIOS[scenario_name]
        aug = img.copy()
        for transform in transforms:
            aug = transform(aug)
        results.append((scenario_name, aug))

    # 그리드 생성 (2행 × 4열)
    h, w = img.shape[:2]
    cols = 4
    rows = (len(results) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, (label, result_img) in enumerate(results[:rows * cols]):
        row = idx // cols
        col = idx % cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = cv2.resize(result_img, (w, h))
        # 라벨 텍스트
        cv2.putText(grid, label,
                    (col*w + 10, row*h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    save = save_path or "/workspace/user5/augmentation_preview.jpg"
    cv2.imwrite(save, grid)
    print(f"✅ 미리보기 저장: {save}")
    return grid


if __name__ == "__main__":
    run_augmentation()

    # 증강 샘플 확인 (첫 번째 이미지로 테스트)
    sample_dirs = [
        os.path.join(INPUT_DIR, "train", emotion)
        for emotion in EMOTIONS
    ]
    for d in sample_dirs:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                sample = os.path.join(d, files[0])
                print(f"\n샘플 미리보기 생성 중: {sample}")
                preview_augmentation(sample)
                break
