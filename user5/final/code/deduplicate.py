"""
중복 이미지 제거 스크립트
=========================
reclassify.py 실행 후 같은 파일이 여러 감정 폴더에 복사된 경우,
CSV의 fix emotion + trn/val 기준으로 올바른 경로만 남기고 나머지를 삭제합니다.

실행:
    python deduplicate.py            # 미리보기 (실제 삭제 없음)
    python deduplicate.py --delete   # 실제 삭제
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# ================= 경로 설정 =================
CSV_PATH = r"D:\project 3\[노션] 3차 프로젝트(4조)-오분류 감정 라벨링.csv"
IMG_DIR  = r"D:\project 3\img"
EMOTIONS = ["sadness", "anger", "panic", "happy", "blank"]
# =============================================

EMOTION_MAP = {
    "sadness": "sadness", "sad": "sadness",
    "anger":   "anger",   "angry": "anger",
    "panic":   "panic",   "fear": "panic", "surprise": "panic",
    "happy":   "happy",   "happiness": "happy",
    "blank":   "blank",   "neutral": "blank",
}


def scan_duplicates():
    """IMG_DIR 전체에서 동일 파일명이 여러 경로에 있는 항목 반환"""
    name_to_paths: dict[str, list[str]] = defaultdict(list)
    for root, _, files in os.walk(IMG_DIR):
        for f in files:
            name_to_paths[f].append(os.path.join(root, f))
    return {k: v for k, v in name_to_paths.items() if len(v) > 1}


def load_correct_paths():
    """CSV에서 파일명 → 올바른 경로 매핑 생성"""
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    correct: dict[str, str] = {}
    for _, row in df.iterrows():
        filename    = str(row["name"]).strip()
        fix_emotion = str(row["fix emotion"]).strip().lower()
        trn_val     = str(row["trn/val"]).strip().lower()

        target_emotion = EMOTION_MAP.get(fix_emotion)
        if target_emotion is None:
            continue

        split = "train" if trn_val in ("trn", "train") else "val"
        correct[filename] = os.path.normpath(
            os.path.join(IMG_DIR, split, target_emotion, filename)
        )
    return correct


def run(dry_run: bool):
    print("=" * 55)
    print(f"중복 이미지 제거 ({'미리보기' if dry_run else '실제 삭제'})")
    print("=" * 55)

    print("\n1) 중복 파일 스캔 중...")
    dupes = scan_duplicates()
    print(f"   중복 파일명: {len(dupes)}개")

    print("\n2) CSV에서 올바른 경로 로드 중...")
    correct_paths = load_correct_paths()
    print(f"   CSV 항목: {len(correct_paths)}개")

    to_delete: list[str] = []
    no_correct_path: list[str] = []

    for filename, paths in dupes.items():
        correct = correct_paths.get(filename)
        if correct is None:
            # CSV에 없는 파일 → 가장 최근 수정된 것만 남김
            paths_sorted = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)
            to_delete.extend(paths_sorted[1:])
            no_correct_path.append(filename)
            continue

        # 올바른 경로 제외하고 나머지 삭제
        for p in paths:
            if os.path.normpath(p) != correct:
                to_delete.append(p)

    print(f"\n3) 삭제 대상: {len(to_delete)}개")
    if no_correct_path:
        print(f"   ※ CSV 미등록(최신 파일 유지): {len(no_correct_path)}개")

    if not to_delete:
        print("\n✅ 삭제할 파일이 없습니다.")
        return

    if dry_run:
        print("\n--- 미리보기 (상위 10개) ---")
        for p in to_delete[:10]:
            print(f"  [삭제 예정] {p}")
        if len(to_delete) > 10:
            print(f"  ... 외 {len(to_delete) - 10}개")
        print(f"\n실제로 삭제하려면: python deduplicate.py --delete")
        return

    # 실제 삭제
    deleted   = 0
    failed    = 0
    freed_bytes = 0
    for path in tqdm(to_delete, desc="삭제 진행", unit="개"):
        try:
            freed_bytes += os.path.getsize(path)
            os.remove(path)
            deleted += 1
        except Exception as e:
            tqdm.write(f"  [실패] {path}: {e}")
            failed += 1

    print(f"""
╔══════════════════════════════════════════╗
║          삭제 완료 리포트                 ║
╠══════════════════════════════════════════╣
║  삭제 성공:  {deleted:>5}개                    ║
║  삭제 실패:  {failed:>5}개                    ║
║  확보 용량:  {freed_bytes / 1024 / 1024:>7.1f} MB                ║
╚══════════════════════════════════════════╝""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="실제 삭제 실행 (기본값: 미리보기)")
    args = parser.parse_args()
    run(dry_run=not args.delete)
