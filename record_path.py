"""경로 레코딩 툴 — hunt_waypoints / patrol_waypoints 기록.

실행:
    python record_path.py            # hunt_waypoints 기록
    python record_path.py --patrol   # patrol_waypoints 기록

조작:
    게임 화면에서 마우스 우클릭  → 현재 좌표 WP 추가
    'z' 키                      → 마지막 WP 취소 (실수 시)
    'q' 키 / Ctrl+C             → 저장 후 종료
    's' 키                      → 중간 저장 (종료 없이)

저장:
    config_automation.json 의 hunt_waypoints.points
    또는 patrol_waypoints.points 에 덮어씁니다.
    기존 파일은 config_automation.json.bak 으로 백업.

표시:
    OpenCV 창에 게임 화면 + 기록된 WP 경로 오버레이
"""

import json
import logging
import os
import shutil
import sys
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("record_path")

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH      = os.path.join(HERE, "config", "config.json")
AUTO_CONFIG_PATH = os.path.join(HERE, "config", "config_automation.json")
WINDOW_NAME      = "Record Path — 우클릭:WP추가 | z:취소 | s:저장 | q:종료"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    is_patrol = "--patrol" in sys.argv

    config   = load_json(CONFIG_PATH)
    auto_cfg = load_json(AUTO_CONFIG_PATH)

    target_key = "patrol_waypoints" if is_patrol else "hunt_waypoints"
    mode_name  = "patrol_waypoints (사냥터 순찰)" if is_patrol else "hunt_waypoints (사냥터 이동)"

    logger.info("=" * 55)
    logger.info(f"  경로 레코딩 모드: {mode_name}")
    logger.info("  게임 화면에서 우클릭 → WP 추가")
    logger.info("  z: 마지막 WP 취소  |  s: 저장  |  q: 종료")
    logger.info("=" * 55)

    # ── 모니터 오프셋 계산 ─────────────────────────────────────────
    monitor_index = config.get("monitor_index", 2)
    try:
        import mss
        with mss.mss() as sct:
            monitors = sct.monitors
            mon = monitors[monitor_index] if monitor_index < len(monitors) else monitors[1]
            mon_left = mon["left"]
            mon_top  = mon["top"]
    except Exception:
        mon_left, mon_top = 0, 0
    logger.info(f"  모니터 오프셋: left={mon_left}, top={mon_top}")

    cap_region = config.get("capture_region")
    cap_x = cap_region["x"] if cap_region else 0
    cap_y = cap_region["y"] if cap_region else 0

    # ── 화면 캡처 ─────────────────────────────────────────────────
    from capture.screen_capture import ScreenCapturer
    capturer = ScreenCapturer(monitor_index=monitor_index, region=cap_region)

    # ── WP 목록 ───────────────────────────────────────────────────
    waypoints = []   # [{"x": int, "y": int, "label": str}, ...]
    wp_counter = [1]

    # ── 마우스 콜백 ───────────────────────────────────────────────
    # OpenCV 창 좌표 → 게임 화면 좌표 변환
    # (창 크기가 바뀔 수 있으므로 실제 창 크기 기준으로 스케일링)
    frame_size = [1920, 1080]  # 실제 캡처 해상도

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            # 창 크기 → 원본 해상도 스케일링
            win_w = cv2.getWindowImageRect(WINDOW_NAME)[2]
            win_h = cv2.getWindowImageRect(WINDOW_NAME)[3]
            if win_w > 0 and win_h > 0:
                scale_x = frame_size[0] / win_w
                scale_y = frame_size[1] / win_h
                gx = int(x * scale_x) + cap_x
                gy = int(y * scale_y) + cap_y
            else:
                gx = x + cap_x
                gy = y + cap_y

            label = f"WP{wp_counter[0]}"
            waypoints.append({"x": gx, "y": gy, "label": label, "wait_ms": 500})
            wp_counter[0] += 1
            logger.info(f"  + {label}: ({gx}, {gy})  총 {len(waypoints)}개")

    # ── OpenCV 창 설정 ────────────────────────────────────────────
    win_x = config.get("window_x", 0)
    win_y = config.get("window_y", 0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, win_x, win_y)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def draw_overlay(frame):
        out = frame.copy()
        # WP 점 + 선 + 번호 표시
        for i, wp in enumerate(waypoints):
            px = wp["x"] - cap_x
            py = wp["y"] - cap_y
            color = (0, 255, 0) if i < len(waypoints) - 1 else (0, 165, 255)
            cv2.circle(out, (px, py), 8, color, -1)
            cv2.circle(out, (px, py), 8, (255, 255, 255), 2)
            cv2.putText(out, wp["label"], (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if i > 0:
                prev = waypoints[i - 1]
                ppx  = prev["x"] - cap_x
                ppy  = prev["y"] - cap_y
                cv2.arrowedLine(out, (ppx, ppy), (px, py),
                                (0, 200, 255), 2, tipLength=0.03)

        # 상단 안내 텍스트
        cv2.putText(out,
                    f"[{mode_name}]  WP: {len(waypoints)}개  |  우클릭:추가  z:취소  s:저장  q:종료",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2)
        return out

    def save_waypoints():
        if not waypoints:
            logger.warning("  저장할 WP가 없습니다.")
            return

        # 백업
        bak_path = AUTO_CONFIG_PATH + ".bak"
        shutil.copy2(AUTO_CONFIG_PATH, bak_path)
        logger.info(f"  백업: {bak_path}")

        # move_timeout_ms 유지
        existing = auto_cfg.get(target_key, {})
        move_timeout = existing.get("move_timeout_ms", 8000)

        auto_cfg[target_key] = {
            "move_timeout_ms": move_timeout,
            "points": waypoints,
        }
        save_json(AUTO_CONFIG_PATH, auto_cfg)
        logger.info(f"  저장 완료: {target_key} {len(waypoints)}개 WP")
        for i, wp in enumerate(waypoints):
            logger.info(f"    {i+1}. {wp['label']}: ({wp['x']}, {wp['y']})")

    # ── 메인 루프 ─────────────────────────────────────────────────
    try:
        while True:
            frame = capturer.grab()
            frame_size[0] = frame.shape[1]
            frame_size[1] = frame.shape[0]

            display = draw_overlay(frame)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                save_waypoints()
                break

            elif key == ord("s"):
                save_waypoints()
                logger.info("  저장 완료 (계속 기록 중...)")

            elif key == ord("z"):
                if waypoints:
                    removed = waypoints.pop()
                    wp_counter[0] -= 1
                    logger.info(f"  - 취소: {removed['label']} ({removed['x']},{removed['y']})  남은 {len(waypoints)}개")
                else:
                    logger.info("  취소할 WP 없음")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                save_waypoints()
                break

    except KeyboardInterrupt:
        logger.info("  Ctrl+C — 저장 후 종료")
        save_waypoints()
    finally:
        capturer.close()
        cv2.destroyAllWindows()
        logger.info("=== record_path.py 종료 ===")


if __name__ == "__main__":
    main()
