"""
아데나 OCR 파이프라인 테스트 (Tesseract 버전)
adena_sample.png → HSV 후보박스 → 전처리 → tesseract OCR → 결과 출력
"""

import sys
sys.path.insert(0, '/home/user/flslwl')

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ────────────────────────────────────────────────────────────────
# 1. 이미지 로드
# ────────────────────────────────────────────────────────────────
img_path = '/home/user/flslwl/config/adena_sample.png'
img = cv2.imread(img_path)
if img is None:
    print("[ERROR] 이미지 로드 실패:", img_path)
    sys.exit(1)

print(f"[INFO] 이미지 크기: {img.shape}  (H×W×C)")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ────────────────────────────────────────────────────────────────
# 2. HSV 범위별 마스크 픽셀 수
# ────────────────────────────────────────────────────────────────
test_ranges = [
    ("노란/주황 텍스트 (기존)",  (15,80,150),  (40,255,255)),
    ("흰색 테두리 (기존)",       (0,0,200),    (180,40,255)),
    ("회색/은색 텍스트",         (0,0,130),    (180,50,210)),
    ("흰+회색 통합",             (0,0,140),    (180,60,255)),
    ("거의 무채색 넓게",         (0,0,100),    (180,80,255)),
    ("아이콘 청록색",            (20,180,180), (35,255,255)),
]

print("\n[마스크 픽셀 수 테스트]")
for name, lower, upper in test_ranges:
    mask = cv2.inRange(hsv_img, lower, upper)
    cnt = int(np.count_nonzero(mask))
    pct = cnt / (img.shape[0] * img.shape[1]) * 100
    print(f"  {name:30s}: {cnt:5d} px  ({pct:.1f}%)")

# ────────────────────────────────────────────────────────────────
# 3. 수정된 HSV 범위로 후보 박스 추출
# ────────────────────────────────────────────────────────────────
print("\n[후보 박스 추출]")

REVISED_RANGES = [
    {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 40, "v_min": 200},  # 흰색 테두리
    {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 50, "v_min": 140},  # 회색/은색 텍스트
]

combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
for rng in REVISED_RANGES:
    s_max = rng.get("s_max", 255)
    m = cv2.inRange(hsv_img,
        (rng["h_min"], rng["s_min"], rng["v_min"]),
        (rng["h_max"], s_max,        255))
    combined_mask = cv2.bitwise_or(combined_mask, m)

kernel = np.ones((3,3), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
print(f"  마스크 픽셀: {int(np.count_nonzero(combined_mask))}")

contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"  컨투어 수: {len(contours)}")

boxes_raw = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 30 or area > 8000:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    h_img, w_img = img.shape[:2]
    x1 = max(0, x-4); y1 = max(0, y-4)
    x2 = min(w_img, x+w+4); y2 = min(h_img, y+h+4)
    boxes_raw.append((x1, y1, x2-x1, y2-y1))

print(f"  필터 후 박스: {len(boxes_raw)}")
for b in boxes_raw:
    print(f"    {b}")

# ────────────────────────────────────────────────────────────────
# 4. 전처리 함수
# ────────────────────────────────────────────────────────────────
def preprocess_patch(patch, scale_target=64):
    h, w = patch.shape[:2]
    scale = max(1, int(np.ceil(scale_target / max(h, 1))))
    scale = min(scale, 8)
    if scale > 1:
        patch = cv2.resize(patch, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)

    _, otsu     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((2,2), np.uint8)
    otsu     = cv2.dilate(otsu,     k, iterations=1)
    otsu_inv = cv2.dilate(otsu_inv, k, iterations=1)

    return gray, otsu, otsu_inv

# ────────────────────────────────────────────────────────────────
# 5. OCR 테스트
# ────────────────────────────────────────────────────────────────
print("\n[OCR 테스트 - tesseract]")

# tesseract 설정
TESS_LANG = 'kor+eng'
TESS_CONFIG = '--psm 7 --oem 1'        # psm7=단일 라인, oem1=LSTM
TESS_CONFIG_BLOCK = '--psm 6 --oem 1'  # psm6=블록

def ocr_patch(name, patch_bgr, config=TESS_CONFIG):
    if patch_bgr is None or patch_bgr.size == 0:
        return
    gray, otsu, otsu_inv = preprocess_patch(patch_bgr)
    variants = [
        ("gray",     gray),
        ("otsu",     otsu),
        ("otsu_inv", otsu_inv),
    ]
    any_result = False
    for vname, var in variants:
        # cv2 → PIL
        pil_img = Image.fromarray(var)
        try:
            text = pytesseract.image_to_string(pil_img, lang=TESS_LANG, config=config).strip()
        except Exception as e:
            text = f"[ERROR:{e}]"
        if text:
            print(f"  [{name}|{vname}] → '{text}'")
            any_result = True
    if not any_result:
        print(f"  [{name}] → (결과 없음 모든 변형)")

# (a) 후보 박스별 OCR
if boxes_raw:
    for i, (bx, by, bw, bh) in enumerate(boxes_raw):
        patch = img[by:by+bh, bx:bx+bw]
        ocr_patch(f"box_{i}({bx},{by},{bw}×{bh})", patch)
else:
    print("  후보 박스 없음 - 전체 이미지로 진행")

# (b) 전체 이미지
print("\n  --- 전체 이미지 ---")
ocr_patch("전체", img, config=TESS_CONFIG_BLOCK)

# (c) 상단 ~50%
h_img = img.shape[0]
print("\n  --- 상단 절반 ---")
ocr_patch("상단절반", img[:h_img//2, :], config=TESS_CONFIG)

# (d) 텍스트 영역 추정 (10~55% 높이)
print("\n  --- 텍스트 영역 (y=10%~55%) ---")
y1_t = int(h_img * 0.10); y2_t = int(h_img * 0.55)
ocr_patch("텍스트영역", img[y1_t:y2_t, :], config=TESS_CONFIG)

# (e) 병합 박스가 있다면 더 넓게 확장한 버전
if boxes_raw:
    print("\n  --- 모든 박스 합친 큰 영역 ---")
    all_x1 = min(b[0] for b in boxes_raw)
    all_y1 = min(b[1] for b in boxes_raw)
    all_x2 = max(b[0]+b[2] for b in boxes_raw)
    all_y2 = max(b[1]+b[3] for b in boxes_raw)
    merged_patch = img[all_y1:all_y2, all_x1:all_x2]
    ocr_patch(f"merged({all_x1},{all_y1}→{all_x2},{all_y2})", merged_patch)

    # 좌우 8px 여유 추가
    pad = 8
    h_img2, w_img2 = img.shape[:2]
    ex1 = max(0, all_x1-pad); ey1 = max(0, all_y1-pad)
    ex2 = min(w_img2, all_x2+pad); ey2 = min(h_img2, all_y2+pad)
    ocr_patch(f"merged+pad({ex1},{ey1}→{ex2},{ey2})", img[ey1:ey2, ex1:ex2])

# ────────────────────────────────────────────────────────────────
# 6. 디버그 이미지 저장
# ────────────────────────────────────────────────────────────────
debug = img.copy()
for (bx, by, bw, bh) in boxes_raw:
    cv2.rectangle(debug, (bx,by), (bx+bw,by+bh), (0,255,0), 1)
cv2.imwrite('/home/user/flslwl/config/debug_boxes.png', debug)
cv2.imwrite('/home/user/flslwl/config/debug_mask.png', combined_mask)
print("\n[저장] debug_boxes.png, debug_mask.png → /home/user/flslwl/config/")
print("[테스트 완료]")
