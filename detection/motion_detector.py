"""Motion-based detection via background subtraction (MOG2).

Builds a running model of the "static background" and flags pixels that
changed from it -- i.e. anything moving -- instead of matching a fixed
color. Same get_mask(bgr_frame) -> binary mask interface the color
detector used to have, so it plugs into the same ContourDetector /
tracker / overlay pipeline unchanged.

Caveat: this assumes the background itself is mostly static. If the
whole screen scrolls/pans with camera movement, everything will look
like "motion" -- narrow the ROI to a region that doesn't scroll to
work around that.

추가: SceneMotionFilter
    플레이어가 이동하면 화면 전체가 움직여 MOG2가 오탐을 쏟아냄.
    연속 두 프레임 간 전체 밝기 차이(평균 절댓값)가 임계값을 초과하면
    "카메라 이동 중"으로 판단해 감지를 스킵시킵니다.
"""

import cv2
import numpy as np


class MotionDetector:
    def __init__(
        self,
        history=200,
        var_threshold=16,
        detect_shadows=False,
        blur_kernel=5,
        morph_kernel=5,
    ):
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self._bg_subtractor = self._build_subtractor()

    def _build_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def reset(self):
        """Re-initializes the background model (e.g. after a scene cut)."""
        self._bg_subtractor = self._build_subtractor()

    @staticmethod
    def _odd(value):
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1

    def get_mask(self, bgr_frame, learning_rate: float = -1.0):
        """배경 차분 마스크를 반환합니다.

        Args:
            bgr_frame: 입력 프레임
            learning_rate: MOG2 학습률
                -1.0  → MOG2 자동 결정 (기본)
                 0.0  → 학습 완전 차단 (이동 중 사용)
                 0~1  → 수동 지정
        """
        # Live-tunable without rebuilding the subtractor (history reset is left to reset()).
        self._bg_subtractor.setVarThreshold(self.var_threshold)
        self._bg_subtractor.setDetectShadows(self.detect_shadows)

        blur_k = self._odd(self.blur_kernel)
        blurred = cv2.GaussianBlur(bgr_frame, (blur_k, blur_k), 0)

        fg_mask = self._bg_subtractor.apply(blurred, learningRate=learning_rate)
        # detectShadows marks shadow pixels as gray (127); keep only strong foreground (255).
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        morph_k = max(1, int(self.morph_kernel))
        kernel = np.ones((morph_k, morph_k), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        return fg_mask


class SceneMotionFilter:
    """화면 전체 움직임(카메라 이동)을 감지해 감지 스킵 여부를 알려줍니다.

    동작 원리:
        1. 프레임 중앙 crop_ratio(기본 60%) 구역만 사용 → UI/HUD 제외
        2. 그레이스케일 + 다운샘플 후 이전 프레임과 MAD 계산
        3. MAD > scene_threshold 가 move_confirm_frames 연속 → is_moving = True
           MAD ≤ scene_threshold 가 연속 → settle_frames 대기 후 False
        4. 정지로 전환된 직후 settle_frames 프레임 동안은
           배경 모델이 안정될 때까지 추가 대기

    개선 사항:
        - crop_ratio: 화면 중앙만 비교 (UI 파티클/스킬바 오판 방지)
        - move_confirm_frames: N프레임 연속이어야 이동으로 판정 (순간 파티클 무시)
        - settle_frames 기본값 상향 (5 → 배경 모델 안정화 충분히 대기)

    사용법:
        filter = SceneMotionFilter(scene_threshold=12.0, settle_frames=5)
        skip = filter.update(roi_frame)   # True면 이번 프레임 감지 스킵
    """

    def __init__(
        self,
        scene_threshold: float = 12.0,
        settle_frames: int = 5,
        downscale: int = 4,
        crop_ratio: float = 0.6,
        move_confirm_frames: int = 2,
    ):
        self.scene_threshold     = scene_threshold      # MAD 임계값 (높을수록 둔감)
        self.settle_frames       = settle_frames        # 정지 후 안정화 대기 프레임 수
        self.downscale           = max(1, downscale)
        self.crop_ratio          = max(0.1, min(1.0, crop_ratio))   # 중앙 구역 비율
        self.move_confirm_frames = max(1, move_confirm_frames)       # 연속 이동 판정 프레임

        self._prev_gray: np.ndarray | None = None
        self._moving_streak  = 0   # 연속 이동 프레임 수
        self._settle_counter = 0   # 정지 전환 후 남은 대기 프레임
        self.is_moving       = False
        self.scene_mad       = 0.0  # 마지막 계산된 MAD (디버그용)

    def _center_crop(self, gray: np.ndarray) -> np.ndarray:
        """중앙 crop_ratio 영역만 잘라냅니다 (UI/HUD 가장자리 제외)."""
        h, w = gray.shape
        ch = int(h * self.crop_ratio)
        cw = int(w * self.crop_ratio)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        return gray[y0:y0 + ch, x0:x0 + cw]

    def update(self, bgr_frame: np.ndarray) -> bool:
        """프레임을 받아 현재 이동 중인지 판단합니다.

        Returns:
            True  → 카메라 이동 중 → 감지 스킵 권장
            False → 정지 중 → 정상 감지 가능
        """
        # 그레이스케일 + 중앙 crop + 다운샘플
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        cropped = self._center_crop(gray)
        h, w = cropped.shape
        small = cv2.resize(
            cropped,
            (max(1, w // self.downscale), max(1, h // self.downscale)),
            interpolation=cv2.INTER_AREA,
        )

        if self._prev_gray is None:
            self._prev_gray = small
            return False

        # 프레임 간 평균 절댓값 차이 (MAD)
        diff = cv2.absdiff(small, self._prev_gray)
        self.scene_mad = float(np.mean(diff))
        self._prev_gray = small

        if self.scene_mad > self.scene_threshold:
            # 이동 후보 — move_confirm_frames 연속 초과해야 실제 이동으로 판정
            self._moving_streak += 1
            if self._moving_streak >= self.move_confirm_frames:
                self._settle_counter = self.settle_frames
                self.is_moving       = True
        else:
            # 정지 — 안정화 대기
            self._moving_streak = 0
            if self._settle_counter > 0:
                self._settle_counter -= 1
                self.is_moving = True   # 아직 대기 중
            else:
                self.is_moving = False

        return self.is_moving

    def reset(self) -> None:
        """이전 프레임 버퍼를 초기화합니다."""
        self._prev_gray      = None
        self._moving_streak  = 0
        self._settle_counter = 0
        self.is_moving       = False
        self.scene_mad       = 0.0
