"""core — 탐지·추적·상태머신 핵심 로직 레이어.

하위 모듈:
    detection : Detection 데이터클래스 + RealtimeTemplateDetector
    tracking  : Enemy 데이터클래스 + NearestNeighborTracker
                + SequentialTargetStateMachine (TargetState)
    state     : 상태머신 베이스 클래스 (BaseFSM)
    overlay   : cv2 오버레이 드로잉 헬퍼
                (draw_enemies, draw_hud, draw_roi, draw_detection_zone,
                 draw_mode_state_banner)

사용 측은 각 서브모듈을 직접 import 한다 (cv2/numpy 의존 lazy 분리):

    from core.detection import Detection, RealtimeTemplateDetector
    from core.tracking  import Enemy, NearestNeighborTracker, TargetState
    from core.state     import BaseFSM
    from core.overlay   import draw_enemies, draw_hud, draw_roi
"""

__all__ = [
    # detection
    "Detection",
    "RealtimeTemplateDetector",
    # tracking
    "Enemy",
    "NearestNeighborTracker",
    "SequentialTargetStateMachine",
    "TargetState",
    # state
    "BaseFSM",
    # overlay
    "draw_enemies",
    "draw_hud",
    "draw_roi",
    "draw_detection_zone",
    "draw_mode_state_banner",
]
