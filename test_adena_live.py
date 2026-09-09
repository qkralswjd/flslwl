"""
아데나 탐지 + 자동 클릭 (고속 버전)

전략:
  - OCR 제거 → HSV 흰 테두리 박스만으로 탐지 (매우 빠름)
  - 박스 크기/비율 필터로 오탐 제거
  - 탐지 즉시 2단계 클릭 (이동 → 1차 → 대기 → 2차)
  - OCR은 백그라운드 스레드에서 확인용으로만 실행

사용법:
    python test_adena_live.py              # 실시간
    python test_adena_live.py image.png   # 이미지 테스트
"""

import sys
import cv2
import numpy as np
import time
import threading
import queue
import json
import os

# 피코 모듈 경로 추가
sys.path.insert(0, os.path.dirname(__file__))

# ── config.json에서 detection_zone 읽기 ───────────────────────────
def _load_scan_region():
    """config/config.json의 detection_zone을 SCAN_REGION으로 변환.
    없으면 전체 화면 fallback."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        dz = cfg.get("detection_zone", {})
        if not dz.get("enabled", True):
            raise ValueError("detection_zone disabled")
        cx = dz.get("center_x", 960)
        cy = dz.get("center_y", 540)
        hw = dz.get("half_width",  600)
        hh = dz.get("half_height", 400)
        region = {"x": cx - hw, "y": cy - hh,
                  "width": hw * 2, "height": hh * 2}
        print(f"[설정] detection_zone 로드: x={region['x']}~{region['x']+region['width']}"
              f"  y={region['y']}~{region['y']+region['height']}")
        return region
    except Exception as e:
        print(f"[설정] detection_zone 로드 실패({e}) → 전체화면 사용")
        return {"x": 0, "y": 0, "width": 1920, "height": 850}

# ── 설정 ──────────────────────────────────────────────────────────
SCAN_REGION  = _load_scan_region()   # config.json detection_zone과 동일

# 박스 탐지 파라미터 (흰 테두리 기반)
WHITE_LOWER  = (0,   0,   200)
WHITE_UPPER  = (180, 50,  255)
DILATION_K   = 7
DILATION_IT  = 3
BOX_MIN_W    = 40     # 이름표 최소 너비
BOX_MAX_W    = 300    # 이름표 최대 너비 (너무 크면 노이즈)
BOX_MIN_H    = 15     # 이름표 최소 높이
BOX_MAX_H    = 80     # 이름표 최대 높이
BOX_PAD      = 4

# 클릭 설정
CLICK_ENABLED     = True
HOVER_DELAY       = 0.25   # 1차→2차 클릭 사이 대기 (노란색 변환)
CLICK2_DELAY      = 0.1    # 2차 클릭 후 대기
RECLICK_INTERVAL  = 1.5    # 같은 위치 재클릭 최소 간격 (초)

# OCR (백그라운드 확인용 — 클릭에는 영향 없음)
OCR_ENABLED  = True    # False 면 OCR 스레드 안 띄움
SCALE        = 4
THRESH_VAL   = 135
KEYWORDS     = ["아데나", "데나", "Adena", "adena", "ADENA"]
MIN_CONF     = 0.03
# ─────────────────────────────────────────────────────────────────


def get_clicker():
    """피코 시리얼 클릭 우선 → 없으면 pydirectinput → pyautogui"""
    try:
        from pico.pico_serial import PicoSerialWorker
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        pico_cfg = cfg.get("pico", {})
        port      = pico_cfg.get("serial_port", "COM4")
        baudrate  = pico_cfg.get("baudrate", 115200)
        pulse_ms  = pico_cfg.get("click_pulse_ms", 20)
        worker = PicoSerialWorker(port=port, baudrate=baudrate,
                                  click_pulse_ms=pulse_ms)
        worker.start()
        time.sleep(0.5)
        print(f"[클릭] 피코 시리얼 ({port}, {baudrate}bps)")
        return "pico", worker
    except Exception as e:
        print(f"[클릭] 피코 연결 실패({e}) → pydirectinput 시도")

    try:
        import pydirectinput
        pydirectinput.FAILSAFE = False
        print("[클릭] pydirectinput (fallback)")
        return "pydirectinput", pydirectinput
    except ImportError:
        pass
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        print("[클릭] pyautogui (fallback)")
        return "pyautogui", pyautogui
    except ImportError:
        pass
    print("[경고] 클릭 라이브러리 없음")
    return None, None


def do_click(clicker_type, clicker, x, y):
    """피코: click(x,y) 2회 / pydirectinput: moveTo → click 2회"""
    if clicker is None or not CLICK_ENABLED:
        return
    if clicker_type == "pico":
        clicker.click(x, y)          # 1차 (호버)
        time.sleep(HOVER_DELAY)
        clicker.click(x, y)          # 2차 (줍기)
    elif clicker_type == "pydirectinput":
        clicker.moveTo(x, y)
        time.sleep(0.05)
        clicker.click()
        time.sleep(HOVER_DELAY)
        clicker.click()
    else:
        clicker.click(x, y)
        time.sleep(HOVER_DELAY)
        clicker.click(x, y)
    time.sleep(CLICK2_DELAY)


def extract_boxes(crop_bgr):
    """HSV 흰 테두리 → dilation → 박스 추출 (OCR 없음, 빠름)"""
    hsv     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    kernel  = np.ones((DILATION_K, DILATION_K), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=DILATION_IT)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = crop_bgr.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (BOX_MIN_W <= w <= BOX_MAX_W and BOX_MIN_H <= h <= BOX_MAX_H):
            continue
        x1 = max(0, x-BOX_PAD);     y1 = max(0, y-BOX_PAD)
        x2 = min(w_img, x+w+BOX_PAD); y2 = min(h_img, y+h+BOX_PAD)
        boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes


def capture_screen(region):
    import mss
    with mss.mss() as sct:
        mon = {"left": region["x"], "top": region["y"],
               "width": region["width"], "height": region["height"]}
        shot = sct.grab(mon)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


# ── OCR 백그라운드 스레드 (확인용만) ─────────────────────────────
class OcrWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.in_q  = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.last_text = ""

        try:
            import easyocr
            print("[OCR] easyocr 초기화 중...")
            self._reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
            print("[OCR] 준비 완료 (백그라운드 확인용)")
        except ImportError:
            self._reader = None
            print("[OCR] easyocr 없음 — HSV 탐지만 사용")

    def submit(self, patch):
        if self._reader is None:
            return
        try:
            self.in_q.put_nowait(patch.copy())
        except queue.Full:
            pass

    def run(self):
        while not self._stop.is_set():
            try:
                patch = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._reader is None:
                continue

            h, w = patch.shape[:2]
            big  = cv2.resize(patch, (w*SCALE, h*SCALE),
                              interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            _, t = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
            proc = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

            raw = self._reader.readtext(proc, detail=1, paragraph=False)
            for (_, text, conf) in raw:
                if conf >= MIN_CONF and any(kw.lower() in text.lower()
                                            for kw in KEYWORDS):
                    self.last_text = text
                    print(f"  [OCR확인] '{text}'  conf={conf:.2f}")

    def stop(self):
        self._stop.set()


# ── 메인 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    clicker_type, clicker = get_clicker()
    print(f"[설정] 클릭={'ON ✅' if CLICK_ENABLED and clicker else 'OFF'}"
          f"  방식={clicker_type}  HOVER={HOVER_DELAY}s  RECLICK={RECLICK_INTERVAL}s")

    # ── 이미지 단발 테스트 ────────────────────────────────────────
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        frame    = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] {img_path}")
            sys.exit(1)
        print(f"[테스트] {img_path}  {frame.shape}")

        RX = SCAN_REGION["x"]; RY = SCAN_REGION["y"]
        rw = min(SCAN_REGION["width"],  frame.shape[1]-RX)
        rh = min(SCAN_REGION["height"], frame.shape[0]-RY)
        crop  = frame[RY:RY+rh, RX:RX+rw]
        boxes = extract_boxes(crop)
        print(f"  박스 {len(boxes)}개: {boxes}")

        debug = crop.copy()
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(debug, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
            sx = bx+bw//2+RX; sy = by+bh//2+RY
            print(f"  → 탐지 위치: ({sx},{sy})  박스: {bw}×{bh}")
        cv2.imshow("Test", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit(0)

    # ── 실시간 루프 ───────────────────────────────────────────────
    ocr_worker = None
    if OCR_ENABLED:
        ocr_worker = OcrWorker()
        ocr_worker.start()

    print(f"\n[실시간] ESC=종료  C=클릭토글  S=저장")

    last_click_pos  = None
    last_click_time = 0.0
    frame_count     = 0

    # 스캔 영역 오프셋 (화면 절대좌표 변환용)
    RX = SCAN_REGION["x"]
    RY = SCAN_REGION["y"]

    while True:
        t0   = time.time()
        crop = capture_screen(SCAN_REGION)  # 이미 SCAN_REGION만큼 잘린 이미지

        # HSV 박스 탐지 (빠름)
        boxes = extract_boxes(crop)

        now   = time.time()
        debug = crop.copy()

        for (bx, by, bw, bh) in boxes:
            # 박스 중심 → 화면 절대좌표
            sx = bx + bw//2 + RX
            sy = by + bh//2 + RY

            cv2.rectangle(debug, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
            cv2.putText(debug, f"{bw}x{bh}", (bx, max(0,by-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

            print(f"  ✅ 박스탐지  위치=({sx},{sy})  크기={bw}×{bh}")

            # OCR 워커에 패치 제출 (확인용)
            if ocr_worker:
                ocr_worker.submit(crop[by:by+bh, bx:bx+bw])

            # 클릭
            if CLICK_ENABLED and clicker is not None:
                same   = (last_click_pos == (sx, sy))
                coolok = (now - last_click_time) >= RECLICK_INTERVAL
                if not same or coolok:
                    do_click(clicker_type, clicker, sx, sy)
                    last_click_pos  = (sx, sy)
                    last_click_time = time.time()
                    print(f"  🖱️  클릭: ({sx},{sy})")

        elapsed = (time.time() - t0) * 1000

        if boxes:
            print(f"[{frame_count}] 박스={len(boxes)}  {elapsed:.0f}ms")
        else:
            sys.stdout.write(f"\r[{frame_count}] 탐지없음  {elapsed:.0f}ms   ")
            sys.stdout.flush()

        # 디버그 창
        sc = min(1.0, 1280 / max(debug.shape[1], 1))
        cv2.imshow("AdenaDetector [ESC/C/S]",
                   cv2.resize(debug, (int(debug.shape[1]*sc),
                                      int(debug.shape[0]*sc))))

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

    if ocr_worker:
        ocr_worker.stop()
    if clicker_type == "pico" and clicker is not None:
        clicker.stop()
    cv2.destroyAllWindows()
    print("\n종료")
