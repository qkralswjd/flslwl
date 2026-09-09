"""
아데나 OCR 단독 테스트 스크립트
사용법:
    python test_adena_live.py              # 실시간 화면 캡처로 테스트
    python test_adena_live.py image.png    # 이미지 파일로 테스트

파이프라인 (adena_crop.png 실측 최적화):
    1. 흰 테두리 HSV 마스크 (S<50, V>200)
    2. 7×7 dilation × 3 → 이름표 전체 박스 연결
    3. w>30 & h>10 필터 (노이즈 제거)
    4. 4배 업스케일 + thresh=135 역방향 이진화
    5. easyocr (ko+en) 또는 tesseract psm=6 kor+eng
"""

import sys
import cv2
import numpy as np
import time

# ── 설정 ──────────────────────────────────────────────────────────
SCAN_REGION = {"x": 0, "y": 0, "width": 1920, "height": 850}  # 스캔 영역
KEYWORDS    = ["아데나", "데나", "Adena", "adena", "ADENA"]

# 박스 추출 파라미터 (adena_crop.png 실측)
WHITE_LOWER = (0,   0,   200)   # 흰 테두리 HSV 하한 (S<50, V>200)
WHITE_UPPER = (180, 50,  255)   # 흰 테두리 HSV 상한
DILATION_K  = 7                 # dilation 커널 크기 (7×7)
DILATION_IT = 3                 # dilation 반복 횟수
BOX_MIN_W   = 30                # 최소 박스 너비 (px)
BOX_MIN_H   = 10                # 최소 박스 높이 (px)
BOX_PAD     = 4                 # 박스 여백 (px)

# 전처리 파라미터 (thresh=130~145 최적 - 실측)
SCALE       = 4                 # 업스케일 배율
THRESH_VAL  = 135               # 이진화 임계값 (adena_crop.png 기준 최적)
# ─────────────────────────────────────────────────────────────────


def get_ocr():
    """easyocr 또는 tesseract 초기화"""
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
        print("[OCR] tesseract 사용 (psm=6, kor+eng)")
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
    """
    아데나 이름표 후보 박스 추출 (dilation 기반)

    전략:
    - 흰 테두리(HSV S<50, V>200)를 마스크로 추출
    - 7×7 kernel dilation × 3 → 얇은 테두리 선들을 하나의 덩어리로 연결
    - bounding rect → w>30 & h>10 조건으로 노이즈 제거

    Returns:
        [(x, y, w, h), ...] 스캔 영역 내 상대 좌표
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # 흰 테두리 마스크 (adena_crop.png 실측: S<50, V>200)
    mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    # dilation으로 얇은 테두리 선들을 연결 → 이름표 전체 영역 확장
    kernel = np.ones((DILATION_K, DILATION_K), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=DILATION_IT)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = crop_bgr.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 작은 노이즈 제거
        if w < BOX_MIN_W or h < BOX_MIN_H:
            continue
        # 패딩 추가 (글씨가 박스 경계에 걸리지 않도록)
        x1 = max(0, x - BOX_PAD)
        y1 = max(0, y - BOX_PAD)
        x2 = min(w_img, x + w + BOX_PAD)
        y2 = min(h_img, y + h + BOX_PAD)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def preprocess(patch):
    """
    4배 업스케일 + 고정 임계값 역방향 이진화

    실측 최적 파라미터 (adena_crop.png 기준):
    - 배경: V≈50~100 (어두운 반투명)
    - 글씨: V≈130~165 (회색/은색)
    - 흰 테두리: V≥200
    - thresh=135: 배경(V<135)→흰색, 글씨(V≥135)→검은색으로 반전
      → 테서랙트/easyocr에 적합한 '흰 배경에 검은 글씨' 형태
    """
    h, w = patch.shape[:2]
    big = cv2.resize(patch, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV: 픽셀 > thresh → 0(검), ≤ thresh → 255(흰)
    _, t = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
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
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        pil  = PILImage.fromarray(gray)
        # psm=11(sparse text) 우선 — adena_crop.png 기준 최적
        # psm=6(block text) 병행 — 전체 이름표 박스일 때 보조
        for psm in [11, 6]:
            text = ocr.image_to_string(
                pil, lang="kor+eng",
                config=f"--psm {psm} --oem 1"
            ).strip()
            if text:
                results.append((text, 0.8))
    return results


def scan_frame(frame, ocr_type, ocr, show_debug=True):
    """한 프레임에서 아데나 탐지"""
    rx = SCAN_REGION["x"]; ry = SCAN_REGION["y"]
    rw = min(SCAN_REGION["width"],  frame.shape[1] - rx)
    rh = min(SCAN_REGION["height"], frame.shape[0] - ry)
    crop = frame[ry:ry + rh, rx:rx + rw]

    boxes = extract_candidate_boxes(crop)
    found = []

    debug_crop = crop.copy() if show_debug else None

    for (bx, by, bw, bh) in boxes:
        patch = crop[by:by + bh, bx:bx + bw]
        if patch.size == 0:
            continue

        processed = preprocess(patch)
        results   = do_ocr(ocr_type, ocr, processed)

        matched = False
        for (text, conf) in results:
            for kw in KEYWORDS:
                if kw.lower() in text.lower():
                    sx = bx + bw // 2 + rx
                    sy = by + bh // 2 + ry
                    found.append((sx, sy, text.strip(), conf))
                    print(f"  ✅ '{text}'  conf={conf:.2f}  위치=({sx},{sy})  박스=({bx},{by},{bw},{bh})")
                    matched = True
                    break

        if show_debug and debug_crop is not None:
            color = (0, 255, 0) if matched else (0, 180, 255)
            cv2.rectangle(debug_crop, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(debug_crop, f"{bw}x{bh}", (bx, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if show_debug and debug_crop is not None:
        scale = min(1.0, 1280 / max(debug_crop.shape[1], 1))
        dw = int(debug_crop.shape[1] * scale)
        dh = int(debug_crop.shape[0] * scale)
        dbg = cv2.resize(debug_crop, (dw, dh))
        cv2.imshow("AdenaDetector  [ESC=종료 / S=저장]", dbg)

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
        else:
            print(f"\n  [결과] {len(found)}개 탐지 완료")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)

    # 실시간 캡처 루프
    print(f"[실시간] 화면 캡처 시작  스캔영역={SCAN_REGION}")
    print(f"  파라미터: dilation={DILATION_K}x{DILATION_K}×{DILATION_IT}  scale={SCALE}  thresh={THRESH_VAL}")
    print("  ESC: 종료 / S: 현재 프레임 저장")
    frame_count = 0
    while True:
        t0 = time.time()
        frame  = capture_screen(SCAN_REGION)
        found  = scan_frame(frame, ocr_type, ocr, show_debug=True)
        elapsed = time.time() - t0

        if found:
            print(f"[프레임 {frame_count}]  탐지: {len(found)}개  ({elapsed*1000:.0f}ms)")
        else:
            sys.stdout.write(f"\r[프레임 {frame_count}]  탐지 없음  ({elapsed*1000:.0f}ms)   ")
            sys.stdout.flush()

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            break
        elif key == ord('s'):
            fname = f"capture_{frame_count}.png"
            cv2.imwrite(fname, frame)
            print(f"\n[저장] {fname}")

    cv2.destroyAllWindows()
    print("\n종료")
