"""
아데나 OCR 단독 테스트 스크립트
사용법:
    python test_adena_live.py              # 실시간 화면 캡처로 테스트
    python test_adena_live.py image.png    # 이미지 파일로 테스트

속도 전략:
    - 메인 루프: HSV 박스 탐지만 (빠름, ~10ms)
    - OCR: 별도 스레드에서 비동기 처리
    - 박스 위치가 이전과 같으면 OCR 재사용 (캐시)
    - 탐지 확정 시 2단계 클릭 (1차 이동+클릭 → 대기 → 2차 클릭)
"""

import sys
import cv2
import numpy as np
import time
import threading
import queue

# ── 설정 ──────────────────────────────────────────────────────────
SCAN_REGION    = {"x": 0, "y": 0, "width": 1920, "height": 850}
KEYWORDS       = ["아데나", "데나", "Adena", "adena", "ADENA"]
MIN_CONFIDENCE = 0.03    # easyocr 신뢰도 하한

# 클릭 설정
CLICK_ENABLED  = True    # C키로 토글 가능
HOVER_DELAY    = 0.25    # 1차 클릭 후 노란색 변환 대기 (초)
CLICK2_DELAY   = 0.1     # 2차 클릭 후 대기 (초)

# 박스 추출 파라미터
WHITE_LOWER = (0,   0,   200)
WHITE_UPPER = (180, 50,  255)
DILATION_K  = 7
DILATION_IT = 3
BOX_MIN_W   = 30
BOX_MIN_H   = 10
BOX_PAD     = 4

# 전처리 파라미터
SCALE       = 4
THRESH_VAL  = 135
# ─────────────────────────────────────────────────────────────────


# ── 클릭 ─────────────────────────────────────────────────────────
def get_clicker():
    try:
        import pydirectinput
        pydirectinput.FAILSAFE = False
        print("[클릭] pydirectinput 사용")
        return "pydirectinput", pydirectinput
    except ImportError:
        pass
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        print("[클릭] pyautogui 사용")
        return "pyautogui", pyautogui
    except ImportError:
        pass
    print("[경고] 클릭 라이브러리 없음")
    return None, None


def do_click(clicker_type, clicker, x: int, y: int):
    """2단계 클릭: 이동 → 1차 클릭 → 대기 → 2차 클릭"""
    if clicker is None or not CLICK_ENABLED:
        return

    if clicker_type == "pydirectinput":
        # pydirectinput: moveTo 먼저, 그 다음 click
        clicker.moveTo(x, y)
        time.sleep(0.05)
        clicker.click()          # 1차
        time.sleep(HOVER_DELAY)
        clicker.click()          # 2차
    else:
        # pyautogui: click(x, y)로 이동+클릭 동시
        clicker.click(x, y)
        time.sleep(HOVER_DELAY)
        clicker.click(x, y)

    time.sleep(CLICK2_DELAY)
    print(f"  🖱️  클릭 완료: ({x}, {y})")


# ── OCR ──────────────────────────────────────────────────────────
def get_ocr():
    try:
        import easyocr
        print("[OCR] easyocr 초기화 중...")
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
    print("[ERROR] easyocr 또는 pytesseract를 설치하세요")
    sys.exit(1)


def preprocess(patch):
    h, w = patch.shape[:2]
    big  = cv2.resize(patch, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


def do_ocr(ocr_type, ocr, img_bgr):
    """OCR → [(text, conf), ...]"""
    if ocr_type == "easyocr":
        raw = ocr.readtext(img_bgr, detail=1, paragraph=False)
        return [(t, c) for (_, t, c) in raw]
    else:
        from PIL import Image as PILImage
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        pil  = PILImage.fromarray(gray)
        results = []
        for psm in [11, 6]:
            text = ocr.image_to_string(pil, lang="kor+eng",
                                       config=f"--psm {psm} --oem 1").strip()
            if text:
                results.append((text, 0.8))
        return results


def is_adena(text, conf):
    if conf < MIN_CONFIDENCE:
        return False
    return any(kw.lower() in text.lower() for kw in KEYWORDS)


# ── 박스 추출 ─────────────────────────────────────────────────────
def extract_candidate_boxes(crop_bgr):
    hsv     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    kernel  = np.ones((DILATION_K, DILATION_K), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=DILATION_IT)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = crop_bgr.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < BOX_MIN_W or h < BOX_MIN_H:
            continue
        x1 = max(0, x - BOX_PAD);  y1 = max(0, y - BOX_PAD)
        x2 = min(w_img, x+w+BOX_PAD); y2 = min(h_img, y+h+BOX_PAD)
        boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes


# ── 캡처 ─────────────────────────────────────────────────────────
def capture_screen(region):
    try:
        import mss
        with mss.mss() as sct:
            mon = {"left": region["x"], "top": region["y"],
                   "width": region["width"], "height": region["height"]}
            shot = sct.grab(mon)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    except ImportError:
        print("[ERROR] pip install mss")
        sys.exit(1)


# ── OCR 워커 스레드 ───────────────────────────────────────────────
class OcrWorker(threading.Thread):
    """메인 루프와 별도로 OCR을 처리하는 스레드."""

    def __init__(self, ocr_type, ocr):
        super().__init__(daemon=True)
        self.ocr_type   = ocr_type
        self.ocr        = ocr
        self.in_q       = queue.Queue(maxsize=2)   # 최신 프레임만 유지
        self.result     = []                        # 최신 탐지 결과
        self.result_lock= threading.Lock()
        self._stop      = threading.Event()

    def submit(self, crop, boxes):
        """메인 루프에서 호출 — 큐가 꽉 차면 버림(최신만 처리)"""
        try:
            self.in_q.put_nowait((crop, boxes))
        except queue.Full:
            pass

    def get_result(self):
        with self.result_lock:
            return list(self.result)

    def run(self):
        while not self._stop.is_set():
            try:
                crop, boxes = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            found = []
            for (bx, by, bw, bh) in boxes:
                patch = crop[by:by+bh, bx:bx+bw]
                if patch.size == 0:
                    continue
                processed = preprocess(patch)
                results   = do_ocr(self.ocr_type, self.ocr, processed)
                for (text, conf) in results:
                    if is_adena(text, conf):
                        found.append((bx, by, bw, bh, text.strip(), conf))

            with self.result_lock:
                self.result = found

    def stop(self):
        self._stop.set()


# ── 메인 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ocr_type, ocr         = get_ocr()
    clicker_type, clicker = get_clicker()

    click_status = "ON ✅" if (CLICK_ENABLED and clicker) else "OFF ❌"
    print(f"[설정] 클릭={click_status}  MIN_CONF={MIN_CONFIDENCE}"
          f"  HOVER={HOVER_DELAY}s  CLICK2={CLICK2_DELAY}s")

    # ── 이미지 파일 단발 테스트 ──────────────────────────────────
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        frame    = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] 이미지 로드 실패: {img_path}")
            sys.exit(1)
        print(f"[테스트] {img_path}  크기: {frame.shape}")

        rx = SCAN_REGION["x"]; ry = SCAN_REGION["y"]
        rw = min(SCAN_REGION["width"],  frame.shape[1]-rx)
        rh = min(SCAN_REGION["height"], frame.shape[0]-ry)
        crop  = frame[ry:ry+rh, rx:rx+rw]
        boxes = extract_candidate_boxes(crop)
        print(f"[박스] {len(boxes)}개")

        debug = crop.copy()
        for (bx, by, bw, bh) in boxes:
            patch     = crop[by:by+bh, bx:bx+bw]
            processed = preprocess(patch)
            results   = do_ocr(ocr_type, ocr, processed)
            matched   = False
            for (text, conf) in results:
                if is_adena(text, conf):
                    sx = bx + bw//2 + rx
                    sy = by + bh//2 + ry
                    print(f"  ✅ '{text}'  conf={conf:.2f}  위치=({sx},{sy})")
                    matched = True
            color = (0,255,0) if matched else (0,180,255)
            cv2.rectangle(debug, (bx,by), (bx+bw,by+bh), color, 2)

        if not any(True for _ in boxes):
            print("  ❌ 후보 박스 없음")

        cv2.imshow("Test", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)

    # ── 실시간 루프 ───────────────────────────────────────────────
    worker = OcrWorker(ocr_type, ocr)
    worker.start()

    print(f"\n[실시간] 스캔영역={SCAN_REGION}")
    print(f"  dilation={DILATION_K}×{DILATION_K}×{DILATION_IT}  "
          f"scale={SCALE}  thresh={THRESH_VAL}")
    print("  ESC: 종료  /  C: 클릭 ON/OFF  /  S: 프레임 저장")

    last_click_pos  = None   # 동일 위치 중복 클릭 방지
    last_click_time = 0.0
    RECLICK_INTERVAL = 1.5   # 같은 위치 재클릭 최소 간격 (초)

    frame_count = 0
    while True:
        t0    = time.time()
        frame = capture_screen(SCAN_REGION)

        rx = SCAN_REGION["x"]; ry = SCAN_REGION["y"]
        rw = min(SCAN_REGION["width"],  frame.shape[1]-rx)
        rh = min(SCAN_REGION["height"], frame.shape[0]-ry)
        crop  = frame[ry:ry+rh, rx:rx+rw]

        # 박스 탐지 (빠름)
        boxes = extract_candidate_boxes(crop)

        # OCR 워커에 최신 프레임 제출
        if boxes:
            worker.submit(crop, boxes)

        # OCR 결과 수신 (이전 프레임 결과)
        ocr_hits = worker.get_result()

        # 디버그 오버레이
        debug = crop.copy()
        now   = time.time()
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(debug, (bx,by), (bx+bw,by+bh), (0,180,255), 1)

        for (bx, by, bw, bh, text, conf) in ocr_hits:
            sx = bx + bw//2 + rx
            sy = by + bh//2 + ry

            # 초록 박스 표시
            cv2.rectangle(debug, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
            cv2.putText(debug, text[:10], (bx, by-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            print(f"  ✅ '{text}'  conf={conf:.2f}  위치=({sx},{sy})")

            # 클릭 (중복 방지: 동일 위치 RECLICK_INTERVAL 이내 재클릭 안 함)
            if CLICK_ENABLED and clicker is not None:
                same_pos = (last_click_pos == (sx, sy))
                cooldown_ok = (now - last_click_time) >= RECLICK_INTERVAL
                if not same_pos or cooldown_ok:
                    do_click(clicker_type, clicker, sx, sy)
                    last_click_pos  = (sx, sy)
                    last_click_time = time.time()

        elapsed = (time.time() - t0) * 1000
        if ocr_hits:
            print(f"[프레임 {frame_count}]  탐지: {len(ocr_hits)}개  ({elapsed:.0f}ms)")
        else:
            sys.stdout.write(f"\r[프레임 {frame_count}]  박스={len(boxes)}개  ({elapsed:.0f}ms)   ")
            sys.stdout.flush()

        # 디버그 창 (박스 탐지만 반영 — 즉각 표시)
        scale_v = min(1.0, 1280 / max(debug.shape[1], 1))
        dw = int(debug.shape[1] * scale_v)
        dh = int(debug.shape[0] * scale_v)
        cv2.imshow("AdenaDetector  [ESC=종료 / C=클릭토글 / S=저장]",
                   cv2.resize(debug, (dw, dh)))

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            fname = f"capture_{frame_count}.png"
            cv2.imwrite(fname, frame)
            print(f"\n[저장] {fname}")
        elif key == ord('c'):
            CLICK_ENABLED = not CLICK_ENABLED
            print(f"\n[토글] 클릭 {'ON ✅' if CLICK_ENABLED else 'OFF ❌'}")

    worker.stop()
    cv2.destroyAllWindows()
    print("\n종료")
