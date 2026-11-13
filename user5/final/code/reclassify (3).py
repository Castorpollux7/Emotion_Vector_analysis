"""
오분류 감정 재분류기 + 원본 삭제 (VSCode용)
=============================================
결과 폴더 구조:
    img/
    ├── train/
    │   ├── sadness/
    │   ├── anger/
    │   ├── panic/
    │   ├── happy/
    │   └── blank/
    └── val/
        ├── sadness/
        └── ...

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

EMOTIONS = ["sadness", "anger", "panic", "happy", "blank"]
SPLITS   = ["train", "val"]

# ⚠️ 안전 장치
# True  → 실제 변경 없이 결과만 미리보기 (먼저 이걸로 테스트!)
# False → 실제 복사 + 원본 삭제 실행
DRY_RUN = True
# =============================================

# CSV의 trn/val 값 → 폴더명 매핑
SPLIT_MAP = {
    "trn": "train",
    "train": "train",
    "val": "val",
    "validation": "val",
}

EMOTION_MAP = {
    "sadness":   "sadness", "sad":       "sadness",
    "anger":     "anger",   "angry":     "anger",
    "panic":     "panic",   "fear":      "panic",   "surprise": "panic",
    "happy":     "happy",   "happiness": "happy",
    "blank":     "blank",   "neutral":   "blank",
}


def setup_folders():
    """train/val × 감정 폴더 생성"""
    for split in SPLITS:
        for emotion in EMOTIONS:
            os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "error"), exist_ok=True)
    print("폴더 구조:")
    for split in SPLITS:
        for emotion in EMOTIONS:
            print(f"  {OUTPUT_DIR}\\{split}\\{emotion}\\")


def find_image(filename):
    """INPUT_DIR 하위 모든 폴더에서 파일 탐색 (2단계 깊이까지)"""
    # 직접 경로
    direct = os.path.join(INPUT_DIR, filename)
    if os.path.exists(direct):
        return direct

    # 1단계 하위 폴더 (예: img/sadness/파일명)
    for folder in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        candidate = os.path.join(folder_path, filename)
        if os.path.exists(candidate):
            return candidate

        # 2단계 하위 폴더 (예: img/train/sadness/파일명)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            candidate2 = os.path.join(subfolder_path, filename)
            if os.path.exists(candidate2):
                return candidate2

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
    print(f"\nCSV 로드 완료: {len(df)}개 항목")

    # trn/val 분포 확인
    split_col = "trn/val"
    if split_col in df.columns:
        print(f"split 분포: {df[split_col].value_counts().to_dict()}\n")

    # 통계 초기화 (split × emotion)
    stats = {}
    for split in SPLITS:
        stats[split] = {e: 0 for e in EMOTIONS}
    stats["not_found"]       = 0
    stats["copy_fail"]       = 0
    stats["unknown_emotion"] = 0
    stats["unknown_split"]   = 0
    stats["deleted"]         = 0
    stats["skipped"]         = 0
    log = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="재분류 진행", unit="장"):
        filename    = str(row["name"]).strip()
        fix_emotion = str(row["fix emotion"]).strip().lower()
        split_raw   = str(row.get("trn/val", "train")).strip().lower()

        # 감정 정규화
        target = EMOTION_MAP.get(fix_emotion)
        if target is None:
            tqdm.write(f"  [알 수 없는 감정] {filename} → '{fix_emotion}'")
            stats["unknown_emotion"] += 1
            log.append({"file": filename, "result": "unknown_emotion",
                        "fix_emotion": fix_emotion})
            continue

        # split 정규화
        split = SPLIT_MAP.get(split_raw)
        if split is None:
            tqdm.write(f"  [알 수 없는 split] {filename} → '{split_raw}' (train으로 처리)")
            split = "train"
            stats["unknown_split"] += 1

        # 이미지 찾기
        src_path = find_image(filename)
        if src_path is None:
            tqdm.write(f"  [파일 없음] {filename}")
            stats["not_found"] += 1
            log.append({"file": filename, "result": "not_found"})
            continue

        # 목적지: img/train/sadness/파일명
        dest_path = os.path.join(OUTPUT_DIR, split, target, filename)

        # 이미 목적지에 있는 경우
        if os.path.exists(dest_path):
            if not DRY_RUN and os.path.normpath(src_path) != os.path.normpath(dest_path):
                os.remove(src_path)
                stats["deleted"] += 1
            else:
                stats["skipped"] += 1
            continue

        # 시뮬레이션 모드
        if DRY_RUN:
            stats[split][target] += 1
            log.append({"file": filename, "result": "will_move",
                        "split": split, "emotion": target,
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
            stats[split][target] += 1
            log.append({"file": filename, "result": "success",
                        "split": split, "emotion": target})
        else:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            tqdm.write(f"  [검증 실패 - 원본 보존] {filename}")
            stats["copy_fail"] += 1
            log.append({"file": filename, "result": "verify_fail"})

    # 로그 저장
    if not DRY_RUN:
        log_path = os.path.join(OUTPUT_DIR, "reclassify_log.csv")
        pd.DataFrame(log).to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"\n로그 저장: {log_path}")

    # 결과 리포트
    label = "시뮬레이션" if DRY_RUN else "실제 실행"
    total_train = sum(stats["train"].values())
    total_val   = sum(stats["val"].values())

    print(f"""
╔══════════════════════════════════════════════╗
║  {label} 완료 리포트                          ║
╠══════════════════════════════════════════════╣
║  전체 항목:       {len(df):>5}개                 ║
║  train 재분류:    {total_train:>5}개                 ║
║  val   재분류:    {total_val:>5}개                 ║
║  원본 삭제:       {stats['deleted']:>5}개                 ║
║  파일 없음:       {stats['not_found']:>5}개                 ║
║  복사 실패:       {stats['copy_fail']:>5}개                 ║
║  알 수 없는 감정: {stats['unknown_emotion']:>5}개                 ║
╠══════════════════════════════════════════════╣
║  감정별 결과 (train / val)                    ║""")
    for emotion in EMOTIONS:
        tr = stats["train"][emotion]
        vl = stats["val"][emotion]
        print(f"║  {emotion:<10}: train {tr:>4}개 / val {vl:>4}개      ║")
    print("╚══════════════════════════════════════════════╝")

    if DRY_RUN:
        print("""
실제로 실행하려면:
  DRY_RUN = True  →  DRY_RUN = False 로 바꾸고 다시 실행하세요.
""")


if __name__ == "__main__":
    print("경로 확인 중...")

    errors = []
    if not os.path.exists(CSV_PATH):
        errors.append(f"  CSV 파일 없음: {CSV_PATH}")
    if not os.path.exists(INPUT_DIR):
        errors.append(f"  이미지 폴더 없음: {INPUT_DIR}")

    if errors:
        print("\n아래 경로를 확인하고 다시 실행해주세요:")
        for e in errors:
            print(e)
    else:
        print(f"  CSV:    {CSV_PATH}")
        print(f"  이미지: {INPUT_DIR}")
        print(f"  출력:   {OUTPUT_DIR}")
        reclassify()
