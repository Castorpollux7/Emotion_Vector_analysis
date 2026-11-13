"""
Unsplash 영화관 관객 사진 자동 수집기
=========================================
1. Unsplash API로 키워드 검색 → 사진 다운로드
2. YOLO-Face로 사람 수 자동 검증 (4명 이상만 통과)
3. 조건 만족 사진만 저장

설치:
    pip install requests ultralytics opencv-python tqdm

사용:
    python collect_audience.py
"""

import os
import requests
import cv2
import time
from tqdm import tqdm
from pathlib import Path

# ================= 설정 =================
ACCESS_KEY = "F9ao4cQbYvRNYNam0SL5vfA7gl01DuQYFNkqr-e8OB0"   # ← Unsplash에서 발급받은 키

# 검색 키워드 (다양하게 시도)
SEARCH_QUERIES = [
    "movie theater audience",
    "cinema audience watching",
    "theater crowd watching",
    "audience watching screen",
    "people watching movie",
    "cinema viewers",
]

# 저장 경로
OUTPUT_DIR  = "/workspace/user5/unsplash_audience"
RAW_DIR     = os.path.join(OUTPUT_DIR, "raw")        # 다운로드 원본
FILTERED_DIR= os.path.join(OUTPUT_DIR, "filtered")   # 4명 이상 통과한 사진
REJECTED_DIR= os.path.join(OUTPUT_DIR, "rejected")   # 조건 미달 사진

# 필터링 조건
MIN_FACES   = 4         # 최소 얼굴 수
PER_QUERY   = 30        # 검색어당 다운로드할 사진 수 (Unsplash 최대 30)
PAGES       = 3         # 검색어당 페이지 수 (30 × 3 = 90장씩)
# =============================================

for d in [RAW_DIR, FILTERED_DIR, REJECTED_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Step 1: Unsplash API로 사진 검색 + 다운로드
# ══════════════════════════════════════════════════════════════

def search_unsplash(query, page=1, per_page=30):
    """Unsplash 검색 API 호출"""
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}
    params = {
        "query":    query,
        "page":     page,
        "per_page": per_page,
        "orientation": "landscape",   # 가로 사진 우선 (영화관 관객 적합)
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code != 200:
        print(f"  ❌ API 오류: {response.status_code} - {response.text[:100]}")
        return []

    data = response.json()
    return data.get("results", [])


def download_image(url, save_path):
    """이미지 URL → 파일 저장"""
    try:
        response = requests.get(url, timeout=15, stream=True)
        if response.status_code != 200:
            return False
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        return False


def collect_from_unsplash():
    """모든 검색어로 사진 수집"""
    print("=" * 55)
    print("  Step 1: Unsplash 사진 다운로드")
    print("=" * 55)

    all_downloaded = []

    for query in SEARCH_QUERIES:
        print(f"\n🔍 검색: '{query}'")

        for page in range(1, PAGES + 1):
            results = search_unsplash(query, page=page, per_page=PER_QUERY)
            if not results:
                break

            for photo in tqdm(results, desc=f"  페이지 {page}", leave=False):
                photo_id = photo["id"]
                # regular 화질 (1080px) - 학습용 충분
                img_url  = photo["urls"]["regular"]

                # 파일명: query_photoid.jpg
                safe_query = query.replace(" ", "_")[:20]
                filename   = f"{safe_query}_{photo_id}.jpg"
                save_path  = os.path.join(RAW_DIR, filename)

                # 이미 있으면 스킵
                if os.path.exists(save_path):
                    all_downloaded.append(save_path)
                    continue

                if download_image(img_url, save_path):
                    all_downloaded.append(save_path)

            # API rate limit 보호 (시간당 50회)
            time.sleep(1)

    # 중복 제거 (photo_id 기준)
    unique_files = list(set(all_downloaded))
    print(f"\n✅ 총 {len(unique_files)}장 다운로드 완료")
    return unique_files


# ══════════════════════════════════════════════════════════════
# Step 2: YOLO-Face로 사람 수 검증
# ══════════════════════════════════════════════════════════════

def filter_by_face_count(image_paths, min_faces=4):
    """
    YOLO-Face로 얼굴 수 카운트 → 조건 만족만 통과
    """
    print(f"\n{'=' * 55}")
    print(f"  Step 2: 얼굴 수 검증 (최소 {min_faces}명)")
    print(f"{'=' * 55}\n")

    from ultralytics import YOLO
    print("YOLO-Face 모델 로드 중...")
    model = YOLO("yolov8n-face.pt")
    print("✅ 모델 로드 완료\n")

    stats = {"passed": 0, "rejected": 0, "error": 0}

    for img_path in tqdm(image_paths, desc="얼굴 검증", unit="장"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                stats["error"] += 1
                continue

            # YOLO 추론
            results = model(img, verbose=False, conf=0.3)
            face_count = 0

            for r in results:
                if r.boxes is not None:
                    face_count = len(r.boxes)

            filename = os.path.basename(img_path)

            # 조건 만족 여부에 따라 분기
            if face_count >= min_faces:
                # 통과 → filtered 폴더
                new_name = f"faces{face_count}_{filename}"
                dest = os.path.join(FILTERED_DIR, new_name)
                cv2.imwrite(dest, img)

                # bbox 시각화 버전도 저장 (확인용)
                vis_img = img.copy()
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                vis_dest = os.path.join(FILTERED_DIR, f"vis_{new_name}")
                cv2.imwrite(vis_dest, vis_img)

                stats["passed"] += 1
            else:
                # 미달 → rejected 폴더
                new_name = f"faces{face_count}_{filename}"
                dest = os.path.join(REJECTED_DIR, new_name)
                cv2.imwrite(dest, img)
                stats["rejected"] += 1

        except Exception as e:
            stats["error"] += 1
            tqdm.write(f"  [에러] {os.path.basename(img_path)}: {e}")

    return stats


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    if ACCESS_KEY == "여기에_본인_API_키_입력":
        print("❌ Unsplash API 키를 먼저 설정하세요!")
        print("   1. https://unsplash.com/developers 접속")
        print("   2. New Application 생성")
        print("   3. Access Key 복사 → 코드 상단 ACCESS_KEY 변수에 붙여넣기")
        return

    # Step 1: 다운로드
    downloaded = collect_from_unsplash()

    if not downloaded:
        print("❌ 다운로드된 사진이 없어요. API 키와 네트워크 확인하세요.")
        return

    # Step 2: 얼굴 수 검증
    stats = filter_by_face_count(downloaded, min_faces=MIN_FACES)

    # 최종 리포트
    total = stats["passed"] + stats["rejected"] + stats["error"]
    print(f"""
╔══════════════════════════════════════════════╗
║       Unsplash 사진 수집 완료 리포트          ║
╠══════════════════════════════════════════════╣
║  전체 다운로드:    {total:>5}장                 ║
║  ✅ 조건 통과:     {stats['passed']:>5}장 ({MIN_FACES}명 이상)        ║
║  ❌ 조건 미달:     {stats['rejected']:>5}장                 ║
║  ⚠️  에러:         {stats['error']:>5}장                 ║
╠══════════════════════════════════════════════╣
║  통과 사진: {FILTERED_DIR[-30:]:>30}  ║
║  탈락 사진: {REJECTED_DIR[-30:]:>30}  ║
╚══════════════════════════════════════════════╝""")

    if stats["passed"] == 0:
        print("\n💡 통과 사진이 0장이면:")
        print(f"   - MIN_FACES를 {MIN_FACES} → 3으로 낮춰보세요")
        print(f"   - SEARCH_QUERIES에 검색어 추가")


if __name__ == "__main__":
    main()
