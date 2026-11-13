import os
import gc
import torch
from ultralytics import YOLO

def flush_memory():
    """학습 시작 전 시스템 RAM 및 GPU VRAM의 찌꺼기를 강제로 청소하여 OOM 및 스레드 충돌 방지"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # A40(Ampere 아키텍처)의 연산 속도를 극대화하는 CuDNN 벤치마크 활성화
        torch.backends.cudnn.benchmark = True 

def train_production_level():
    flush_memory()

    # 1. 파일 존재 여부 하드 체크 (경로 오류로 인한 실행 중단 사전 차단)
    weights_path = '/workspace/user1/4_YOLO_Detecting/yolo26n.pt'
    yaml_path = '/workspace/user1/4_YOLO_Detecting/face_detecting/theater_face.yaml'
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ 오류: 베이스 가중치 파일을 찾을 수 없습니다 -> {weights_path}")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ 오류: YAML 데이터셋 파일을 찾을 수 없습니다 -> {yaml_path}")

    # 2. 모델 인스턴스화 및 예외 처리 블록(Try-Except) 적용
    try:
        model = YOLO(weights_path)
        
        # 3. A40 최적화 및 안정성 보장 파라미터
        results = model.train(
            data=yaml_path,
            epochs=100,
            batch=32,               # VRAM 안정성 확보
            workers=0,              # Pin memory 스레드 터짐 원천 차단
            imgsz=640,
            device=0,
            
            # --- 실무 레벨 추가 파라미터 ---
            amp=True,               # Automatic Mixed Precision (FP16 연산). A40의 텐서 코어를 활용해 학습 속도를 비약적으로 향상
            seed=42,                # 실험 재현성(Reproducibility) 확보를 위한 시드 고정
            deterministic=True,     # cuDNN 연산 결과를 항상 동일하게 유지 (가중치 디버깅 용이)
            project='runs/detect',
            name='theater_face_production',
            exist_ok=True,          # 동일 이름 폴더 존재 시 덮어쓰기 허용 (잔여 폴더 찌꺼기 방지)
            
            # --- 도메인 물리법칙 제어 증강 ---
            fliplr=0.5,
            flipud=0.0,
            degrees=15.0,
            hsv_s=0.2,
            hsv_v=0.2,
            mosaic=1.0,
            
            patience=20
        )
        print("✅ Production 레벨 파인튜닝이 성공적으로 완료되었습니다.")

    except Exception as e:
        print(f"🚨 학습 중 치명적인 에러 발생: {e}")
    finally:
        # 에러가 나든 정상 종료되든 마지막에 무조건 VRAM을 반환하여 서버 자원 잠김(Lock) 현상 방지
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train_production_level()