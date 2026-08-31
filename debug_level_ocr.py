"""레벨 OCR 디버그 스크립트 — 실제 화면에서 OCR 결과 출력."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import cv2
import numpy as np

def main():
    # config 로드
    with open("config/config.json") as f:
        cfg = json.load(f)
    with open("config/config_automation.json") as f:
        auto = json.load(f)

    region = auto["level_ocr"]["region"]
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    print(f"레벨 OCR 영역: x={x}, y={y}, w={w}, h={h}")

    # 화면 캡처
    import mss
    monitor_index = cfg.get("monitor_index", 2)
    print(f"모니터 인덱스: {monitor_index}")

    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        print(f"모니터 정보: {mon}")

        # 전체 모니터 캡처
        shot = sct.grab(mon)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    print(f"캡처 프레임 크기: {frame.shape}")

    # 레벨 영역 크롭
    crop = frame[y:y+h, x:x+w]
    print(f"크롭 크기: {crop.shape}")

    # 크롭 이미지 저장 (눈으로 확인용)
    cv2.imwrite("debug_level_crop.png", crop)
    print("→ debug_level_crop.png 저장됨 (실제로 뭘 보고 있는지 확인)")

    # 전처리 이미지도 저장
    h2, w2 = crop.shape[:2]
    enlarged = cv2.resize(crop, (w2*3, h2*3), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("debug_level_processed.png", binary)
    print("→ debug_level_processed.png 저장됨 (전처리 결과)")

    # OCR 실행
    print("\neasyocr 초기화 중...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    print("OCR 실행 중...")
    results = reader.readtext(binary, detail=1, paragraph=False)
    print(f"\nOCR 결과 ({len(results)}개):")
    for bbox, text, conf in results:
        print(f"  텍스트='{text}'  신뢰도={conf:.2f}")

    # 원본 crop에도 시도
    print("\n원본 crop OCR:")
    results2 = reader.readtext(crop, detail=1, paragraph=False)
    for bbox, text, conf in results2:
        print(f"  텍스트='{text}'  신뢰도={conf:.2f}")

if __name__ == "__main__":
    main()
