import cv2
import numpy as np
from ultralytics import YOLO

def apply_clahe_color(image):
    """
    컬러 이미지의 색상을 유지하면서 조도(명암)만 평탄화하는 CLAHE 전처리.
    BGR을 LAB 색상 공간으로 변환하여 L(Lightness) 채널에만 CLAHE를 적용.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def detect_and_crop_faces(image_path, model_path, conf_thresh=0.45, min_size=36):
    """
    YOLO를 이용해 다수의 얼굴을 탐지하고 크롭하여 반환하는 함수.
    
    Returns:
        valid_faces (list): [{'crop': ndarray, 'box': (x1, y1, x2, y2), 'conf': float}, ...]
    """
    # 1. 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: 이미지를 읽을 수 없습니다. 경로를 확인하세요: {image_path}")
        return []
        
    img_clahe = apply_clahe_color(img)
    
    # 2. YOLO 인퍼런스
    model = YOLO(model_path)
    # save=False: 불필요한 디스크 I/O를 막아 속도 향상
    results = model.predict(source=img_clahe, conf=conf_thresh, save=False, verbose=False)
    
    valid_faces = []
    
    # 3. 바운딩 박스 필터링 및 크롭
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 좌표 및 정보 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            w = x2 - x1
            h = y2 - y1
            
            # [필터링 1] 너무 작은 객체(36px 미만)는 노이즈이거나 감정 판독이 불가능하므로 드랍
            if w < min_size or h < min_size:
                continue
                
            # [필터링 2] 선글라스/마스크 등 가려짐(Occlusion) 클래스인 경우 감정 판독에서 배제
            # (라벨링 시 클래스 1을 Face_Occluded로 지정했다고 가정)
            if cls == 1:
                continue
                
            # 크롭 실행
            face_crop = img_clahe[y1:y2, x1:x2]
            
            valid_faces.append({
                'crop': face_crop,
                'box': (x1, y1, x2, y2),
                'conf': conf
            })
            
    print(f"✅ 탐지 완료: 총 {len(boxes)}개 객체 중 유효 안면 {len(valid_faces)}개 크롭 추출.")
    return valid_faces

# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 어제 논의한 파인튜닝 모델 또는 yolo26n.pt 지정
    # 주의: yolo26n.pt 기본 모델 사용 시 cls 필터링 로직이 COCO 데이터셋 기준으로 작동하므로 수정 필요
    TEST_IMAGE = '/workspace/user5/project 3 face detect v2.v1i.yolov8 (1)/train/images/2l5okfhfIU0_jpg.rf.9d3aaba09c32c0c86cc768b284820176.jpg' 
    MODEL_WEIGHT = '/workspace/user1/4_YOLO_Detecting/yolo26n.pt' 
    
    extracted_faces = detect_and_crop_faces(TEST_IMAGE, MODEL_WEIGHT)
    
    # 크롭된 이미지 개별 저장 테스트
    for idx, face in enumerate(extracted_faces):
        save_path = f"cropped_face_{idx}.jpg"
        cv2.imwrite(save_path, face['crop'])
        print(f"저장 완료: {save_path} (신뢰도: {face['conf']:.2f})")