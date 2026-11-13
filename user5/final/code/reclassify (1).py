"""
오분류 감정 재분류기 + 원본 삭제 (VSCode용)
=============================================
1. CSV의 'fix emotion' 기준으로 이미지를 재분류
2. 복사 성공 확인 후 원본 이미지 삭제
3. 실수 방지를 위해 DRY_RUN 모드 지원

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
CSV_PATH   = r"D:\project 3\오분류_감정_라벨링.csv"
INPUT_DIR  = r"D:\project 3\img"
OUTPUT_DIR = r"D:\project 3\img"   # 같은 img 폴더 안에 재분류

EMOTIONS   = ["sadness", "anger", "panic", "happy", "blank"]

# ⚠️ 안전 장치
# True  → 실제 변경 없이 결과만 미리보기 (먼저 이걸로 테스트!)
# False → 실제 복사 + 원본 삭제 실행
DRY_RUN = True
# =============================================

EMOTION_MAP = {
    "sadness":   "sadness", "sad":       "sadness",
    "anger":     "anger",   "angry":     "anger",
    "panic":     "panic",   "fear":      "panic",   "surprise": "panic",
    "happy":     "happy",   "happiness": "happy",
    "blank":     "blank",   "neutral":   "blank",
}


def setup_folders():
    for emotion in EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)


def find_image(filename):
    """INPUT_DIR 하위 모든 폴더에서 파일 탐색"""
    direct = os.path.join(INPUT_DIR, filename)
    if os.path.exists(direct):
        return direct
    for folder in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        candidate = os.path.join(folder_path, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def verify_copy(src, dest):
    """파일 크기로 복사 성공 여부 확인"""
    return (
        os.path.exists(dest) and
        os.path.getsize(dest) == os.path.getsize(src)
    )


def reclassify():
    mode_str = "시뮬레이션 모드 (실제 변경 없음)" if DRY_RUN else "실제 실행 모드 (복사 + 원본 삭제)"
    print(f"\n{'='*50}")
    print(f"  {mode_str}")
    print(f"{'='*50}\n")

    setup_folders()

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    print(f"CSV 로드 완료: {len(df)}개 항목\n")

    stats = {e: 0 for e in EMOTIONS}
    stats.update({"not_found": 0, "copy_fail": 0,
                  "unknown_emotion": 0, "deleted": 0, "skipped": 0})
    log = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="재분류 진행", unit="장"):
        filename    = str(row["name"]).strip()
        fix_emotion = str(row["fix emotion"]).strip().lower()
        target      = EMOTION_MAP.get(fix_emotion)

        # 알 수 없는 감정
        if target is None:
            tqdm.write(f"  [알 수 없는 감정] {filename} → '{fix_emotion}'")
            stats["unknown_emotion"] += 1
            log.append({"file": filename, "result": "unknown_emotion",
                        "fix_emotion": fix_emotion})
            continue

        # 이미지 찾기
        src_path = find_image(filename)
        if src_path is None:
            tqdm.write(f"  [파일 없음] {filename}")
            stats["not_found"] += 1
            log.append({"file": filename, "result": "not_found"})
            continue

        dest_path = os.path.join(OUTPUT_DIR, target, filename)

        # 이미 목적지에 있는 경우
        if os.path.exists(dest_path):
            # 원본이 다른 위치에 있으면 삭제
            if not DRY_RUN and os.path.normpath(src_path) != os.path.normpath(dest_path):
                os.remove(src_path)
                stats["deleted"] += 1
            else:
                stats["skipped"] += 1
            continue

        # 시뮬레이션 모드
        if DRY_RUN:
            stats[target] += 1
            log.append({"file": filename, "result": "will_move",
                        "from": src_path, "to": dest_path})
            continue

        # 실제 복사
        try:
            shutil.copy2(src_path, dest_path)
        except Exception as e:
            tqdm.write(f"  [복사 실패] {filename}: {e}")
            stats["copy_fail"] += 1
            log.append({"file": filename, "result": "copy_fail", "error": str(e)})
            continue

        # 복사 성공 확인 후 원본 삭제
        if verify_copy(src_path, dest_path):
            if os.path.normpath(src_path) != os.path.normpath(dest_path):
                os.remove(src_path)
                stats["deleted"] += 1
            stats[target] += 1
            log.append({"file": filename, "result": "success", "emotion": target})
        else:
            # 검증 실패 시 목적지 삭제하고 원본 보존
            if os.path.exists(dest_path):
                os.remove(dest_path)
            tqdm.write(f"  [검증 실패 - 원본 보존] {filename}")
            stats["copy_fail"] += 1
            log.append({"file": filename, "result": "verify_fail"})

    # 로그 저장 (실제 실행 시에만)
    if not DRY_RUN:
        log_path = os.path.join(OUTPUT_DIR, "reclassify_log.csv")
        pd.DataFrame(log).to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"\n로그 저장: {log_path}")

    # 결과 리포트
    total_success = sum(stats[e] for e in EMOTIONS)
    label = "시뮬레이션" if DRY_RUN else "실제 실행"
    print(f"""
╔══════════════════════════════════════════╗
║  {label} 완료 리포트{'(변경 없음)' if DRY_RUN else '          '}         ║
╠══════════════════════════════════════════╣
║  전체 항목:       {len(df):>5}개               ║
║  재분류 성공:     {total_success:>5}개               ║
║  원본 삭제:       {stats['deleted']:>5}개               ║
║  파일 없음:       {stats['not_found']:>5}개               ║
║  복사 실패:       {stats['copy_fail']:>5}개               ║
║  알 수 없는 감정: {stats['unknown_emotion']:>5}개               ║
╠══════════════════════════════════════════╣
║  감정별 재분류 결과                       ║""")
    for emotion in EMOTIONS:
        print(f"║  {emotion:<10}: {stats[emotion]:>5}개               ║")
    print("╚══════════════════════════════════════════╝")

    if DRY_RUN:
        print("""
실제로 실행하려면:
  코드 상단의 DRY_RUN = True  →  DRY_RUN = False 로 바꾸고 다시 실행하세요.
""")


if __name__ == "__main__":
    print("경로 확인 중...")

    errors = []
    if not os.path.exists(CSV_PATH):
        errors.append(f"  CSV 파일 없음: {CSV_PATH}")
    if not os.path.exists(INPUT_DIR):
        errors.append(f"  이미지 폴더 없음: {INPUT_DIR}")

    if errors:
        print("\n아래 경로를 확인해주세요:")
        for e in errors:
            print(e)
    else:
        print(f"  CSV:    {CSV_PATH}")
        print(f"  이미지: {INPUT_DIR}")
        print(f"  출력:   {OUTPUT_DIR}")
        reclassify()
