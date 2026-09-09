"""
아데나 OCR 단독 테스트 스크립트
사용법:
    python test_adena_live.py              # 실시간 화면 캡처로 테스트
    python test_adena_live.py image.png    # 이미지 파일로 테스트
"""

import sys
import cv2
import numpy as np
import time

# ── 설정 ──────────────────────────────────────────────────────────
SCAN_REGION = {"x": 0, "y": 0, "width": 1920, "height": 850}  # 스캔 영역
KEYWORDS    = ["아데나", "데나", "Adena", "adena", "ADENA"]
# ─────────────────────────────────────────────────────────────────


def get_ocr():
    try:
        import easyocr
        print("[OCR] easyocr 초기화 중... (첫 실행 시 시간 걸림)")
        reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        print("[OCR] easyocr 준비 완료")
        return "easyocr", reader
    except ImportError:
        pass

    try:
        import pytesseract
        print("[OCR] tesseract 사용")
        return "tesseract", pytesseract
    except ImportError:
        pass

    print("[ERROR] easyocr 또는 pytesseract 중 하나를 설치하세요")
    print("        pip install easyocr")
    sys.exit(1)


def capture_screen(region):
    """화면 캡처 (mss 사용)"""
    try:
        import mss
        with mss.mss() as sct:
            mon = {
                "left":   region["x"],
                "top":    region["y"],
                "width":  region["width"],
                "height": region["height"],
            }
            shot = sct.grab(mon)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    except ImportError:
        print("[ERROR] mss 미설치: pip install mss")
        sys.exit(1)


def extract_candidate_boxes(crop_bgr):
    """HSV 색상 필터로 아데나 이름표 후보 박스 추출"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    ranges = [
        # 흰색 테두리 (S<40, V>200)
        {"lower": (0,  0,  200), "upper": (180, 40, 255)},
        # 회색/은색 텍스트 (S<50, V=140~210)
        {"lower": (0,  0,  140), "upper": (180, 50, 255)},
    ]
    for r in ranges:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, r["lower"], r["upper"]))

    k = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = crop_bgr.shape[:2]
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 8000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x-4); y1 = max(0, y-4)
        x2 = min(w_img, x+w+4); y2 = min(h_img, y+h+4)
        boxes.append((x1, y1, x2-x1, y2-y1))

    # 인접 박스 병합
    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(boxes)
        for i, (x1,y1,w1,h1) in enumerate(boxes):
            if used[i]: continue
            bx1,by1,bx2,by2 = x1,y1,x1+w1,y1+h1
            for j, (x2,y2,w2,h2) in enumerate(boxes):
                if i==j or used[j]: continue
                cx1,cy1,cx2,cy2 = x2,y2,x2+w2,y2+h2
                if bx1-12<=cx2 and bx2+12>=cx1 and by1-12<=cy2 and by2+12>=cy1:
                    bx1=min(bx1,cx1); by1=min(by1,cy1)
                    bx2=max(bx2,cx2); by2=max(by2,cy2)
                    used[j]=True; merged=True
            result.append((bx1,by1,bx2-bx1,by2-by1))
            used[i]=True
        boxes = result
    return boxes


def preprocess(patch):
    """3배 업스케일 + thresh150 역방향"""
    h, w = patch.shape[:2]
    big = cv2.resize(patch, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


def do_ocr(ocr_type, ocr, img_bgr):
    """OCR 실행 → [(text, conf), ...]"""
    results = []
    if ocr_type == "easyocr":
        raw = ocr.readtext(img_bgr, detail=1, paragraph=False)
        for (_, text, conf) in raw:
            results.append((text, conf))
    else:  # tesseract
        from PIL import Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        for psm in [6, 11]:
            text = ocr.image_to_string(pil, lang="kor+eng", config=f"--psm {psm} --oem 1").strip()
            if text:
                results.append((text, 0.8))  # tesseract conf 없으므로 고정
    return results


def scan_frame(frame, ocr_type, ocr, show_debug=True):
    """한 프레임에서 아데나 탐지"""
    rx = SCAN_REGION["x"]; ry = SCAN_REGION["y"]
    rw = min(SCAN_REGION["width"],  frame.shape[1]-rx)
    rh = min(SCAN_REGION["height"], frame.shape[0]-ry)
    crop = frame[ry:ry+rh, rx:rx+rw]

    boxes = extract_candidate_boxes(crop)
    found = []

    for (bx, by, bw, bh) in boxes:
        patch = crop[by:by+bh, bx:bx+bw]
        if patch.size == 0:
            continue
        processed = preprocess(patch)
        results = do_ocr(ocr_type, ocr, processed)

        for (text, conf) in results:
            for kw in KEYWORDS:
                if kw.lower() in text.lower():
                    sx = bx + bw//2 + rx
                    sy = by + bh//2 + ry
                    found.append((sx, sy, text.strip(), conf))
                    print(f"  ✅ '{text}'  conf={conf:.2f}  위치=({sx},{sy})")
                    break

        if show_debug:
            color = (0,255,0) if any(f[2] for f in found if True) else (0,120,255)
            cv2.rectangle(crop, (bx,by), (bx+bw,by+bh), (0,200,255), 1)

    if show_debug:
        # 디버그 창 표시
        dbg = cv2.resize(crop, (crop.shape[1]//2, crop.shape[0]//2))
        cv2.imshow("AdenaDetector [ESC=종료 / S=저장]", dbg)

    return found


# ── 메인 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ocr_type, ocr = get_ocr()

    # 이미지 파일 지정 시 단발 테스트
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] 이미지 로드 실패: {img_path}")
            sys.exit(1)
        print(f"[테스트] 이미지: {img_path}  크기: {frame.shape}")
        found = scan_frame(frame, ocr_type, ocr, show_debug=True)
        if not found:
            print("  ❌ 아데나 미탐지")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)

    # 실시간 캡처 루프
    print(f"[실시간] 화면 캡처 시작  스캔영역={SCAN_REGION}")
    print("  ESC: 종료 / S: 현재 프레임 저장")
    frame_count = 0
    while True:
        t0 = time.time()
        frame = capture_screen(SCAN_REGION)
        found = scan_frame(frame, ocr_type, ocr, show_debug=True)
        elapsed = time.time() - t0

        if found:
            print(f"[프레임 {frame_count}]  탐지: {len(found)}개  ({elapsed*1000:.0f}ms)")
        else:
            sys.stdout.write(f"\r[프레임 {frame_count}]  탐지 없음  ({elapsed*1000:.0f}ms)   ")
            sys.stdout.flush()

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == ord('s'):
            fname = f"capture_{frame_count}.png"
            cv2.imwrite(fname, frame)
            print(f"\n[저장] {fname}")

    cv2.destroyAllWindows()
    print("\n종료")
