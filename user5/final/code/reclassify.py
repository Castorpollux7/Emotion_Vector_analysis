"""
오분류 감정 재분류기 (VSCode용)
=================================
CSV 파일의 'fix emotion' 컬럼 기준으로
이미지를 해당 감정 폴더로 재분류합니다.

설치:
    pip install pandas tqdm

실행:
    python reclassify.py
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

# ================= 경로 설정 =================
# CSV 파일 경로
CSV_PATH   = r"D:\project 3\오분류_감정_라벨링.csv"

# 원본 이미지가 있는 폴더 (여기 안에 sadness, anger 등 폴더 존재)
INPUT_DIR  = r"D:\project 3\img"

# 재분류 결과가 저장될 폴더
OUTPUT_DIR = r"D:\project 3\reclassified"

# 분류할 감정 목록
EMOTIONS   = ["sadness", "anger", "panic", "happy", "blank"]
# =============================================


def setup_folders():
    """결과 폴더 생성"""
    for emotion in EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "error"), exist_ok=True)
    print(f"✅ 결과 폴더 생성 완료: {OUTPUT_DIR}")


def find_image(filename):
    """
    INPUT_DIR 안의 모든 하위 폴더에서 이미지 찾기
    예: D:/project 3/img/sadness/파일명.jpg
    """
    # 1. 직접 경로 시도
    direct = os.path.join(INPUT_DIR, filename)
    if os.path.exists(direct):
        return direct

    # 2. 하위 폴더에서 탐색
    for emotion_folder in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, emotion_folder)
        if not os.path.isdir(folder_path):
            continue
        candidate = os.path.join(folder_path, filename)
        if os.path.exists(candidate):
            return candidate

    return None  # 찾지 못함


def reclassify():
    setup_folders()

    # CSV 읽기
    print(f"\nCSV 불러오는 중: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✅ 총 {len(df)}개 항목 로드 완료")
    print(f"컬럼 확인: {df.columns.tolist()}\n")

    # 컬럼명 정리 (공백 제거)
    df.columns = df.columns.str.strip()

    # 필수 컬럼 확인
    if "name" not in df.columns or "fix emotion" not in df.columns:
        print("❌ CSV에 'name' 또는 'fix emotion' 컬럼이 없어요!")
        print(f"   현재 컬럼: {df.columns.tolist()}")
        return

    # 통계
    stats = {e: 0 for e in EMOTIONS}
    stats["error"]    = 0
    stats["not_found"] = 0

    print("분류 시작...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="재분류 진행", unit="장"):
        filename    = str(row["name"]).strip()
        fix_emotion = str(row["fix emotion"]).strip().lower()

        # 감정 정규화 (표기 차이 대응)
        emotion_map = {
            "sadness": "sadness", "sad": "sadness",
            "anger":   "anger",   "angry": "anger",
            "panic":   "panic",   "fear": "panic", "surprise": "panic",
            "happy":   "happy",   "happiness": "happy",
            "blank":   "blank",   "neutral": "blank",
        }
        target_emotion = emotion_map.get(fix_emotion)

        if target_emotion is None:
            tqdm.write(f"  [알 수 없는 감정] {filename} → '{fix_emotion}'")
            stats["error"] += 1
            continue

        # 이미지 파일 찾기
        src_path = find_image(filename)

        if src_path is None:
            tqdm.write(f"  [파일 없음] {filename}")
            stats["not_found"] += 1
            continue

        # 목적지 경로
        dest_path = os.path.join(OUTPUT_DIR, target_emotion, filename)

        # 이미 존재하면 스킵
        if os.path.exists(dest_path):
            continue

        # 복사
        try:
            shutil.copy2(src_path, dest_path)
            stats[target_emotion] += 1
        except Exception as e:
            tqdm.write(f"  [복사 실패] {filename}: {e}")
            stats["error"] += 1

    # 결과 출력
    total_success = sum(stats[e] for e in EMOTIONS)
    print(f"""
╔══════════════════════════════════════╗
║         재분류 완료 리포트            ║
╠══════════════════════════════════════╣
║  전체 항목:    {len(df):>5}개              ║
║  성공:         {total_success:>5}개              ║
║  파일 없음:    {stats['not_found']:>5}개              ║
║  오류:         {stats['error']:>5}개              ║
╠══════════════════════════════════════╣
║  감정별 분류 결과                     ║""")
    for emotion in EMOTIONS:
        print(f"║  {emotion:<10}: {stats[emotion]:>5}개              ║")
    print(f"""╚══════════════════════════════════════╝
저장 위치: {OUTPUT_DIR}""")


if __name__ == "__main__":
    # ── 경로 확인 ──────────────────────────────────────────────
    print("=" * 50)
    print("경로 확인")
    print("=" * 50)

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV 파일을 찾을 수 없어요: {CSV_PATH}")
        print("   CSV_PATH를 확인해주세요.")
    elif not os.path.exists(INPUT_DIR):
        print(f"❌ 이미지 폴더를 찾을 수 없어요: {INPUT_DIR}")
        print("   INPUT_DIR을 확인해주세요.")
    else:
        print(f"✅ CSV:       {CSV_PATH}")
        print(f"✅ 이미지:    {INPUT_DIR}")
        print(f"✅ 저장 위치: {OUTPUT_DIR}")
        print()
        reclassify()
