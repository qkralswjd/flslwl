"""통합 설정 클래스.

기존 config.json + config_automation.json 두 파일을 하나로 통합합니다.

설계 원칙:
  - Settings 객체 하나가 모든 설정을 담는다
  - 섹션별 중첩 dataclass로 타입 안정성 확보
  - JSON 파일 ↔ Settings 객체 상호 변환
  - 누락된 키는 기본값으로 자동 대체 (KeyError 없음)
  - 런타임에 단일 항목 수정 후 저장 가능

사용법:
    s = Settings.load()               # 기본 경로
    s = Settings.load("my_cfg.json")  # 지정 경로
    port = s.pico.serial_port
    s.pico.serial_port = "COM5"
    s.save()
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_HERE, "config.json")


# ═══════════════════════════════════════════════════════
# 섹션 dataclass
# ═══════════════════════════════════════════════════════

@dataclass
class CaptureSettings:
    monitor_index: int = 2
    fps: int = 60
    region_x: int = 0
    region_y: int = 0
    region_w: int = 1920
    region_h: int = 1080


@dataclass
class DetectionSettings:
    fps: int = 6
    templates_dir: str = "config/templates"
    reject_templates_dir: str = "config/templates_reject"
    match_threshold: float = 0.50
    scale_factors: List[float] = field(default_factory=lambda: [1.0])
    nms_iou_threshold: float = 0.30
    max_templates: int = 60
    # detection_zone
    zone_enabled: bool = True
    zone_cx: int = 960
    zone_cy: int = 490
    zone_half_w: int = 600
    zone_half_h: int = 250


@dataclass
class PicoSettings:
    enabled: bool = True
    serial_port: str = "COM4"
    baudrate: int = 115200
    click_pulse_ms: int = 20
    click_offset_x: int = 0
    click_offset_y: int = 0
    lock_confirm_frames: int = 2
    wait_dead_timeout_ms: int = 5000
    next_target_cooldown_ms: int = 300
    target_priority: str = "nearest_center"
    drag_enabled: bool = True
    drag_dx: int = 80
    drag_dy: int = 0
    drag_steps: int = 8


@dataclass
class TrackingSettings:
    max_missing_frames: int = 30
    max_match_distance: int = 100


@dataclass
class OverlaySettings:
    window_x: int = 0
    window_y: int = 0
    show_debug: bool = True


@dataclass
class HpBarSettings:
    region_x: int = 548
    region_y: int = 842
    region_w: int = 335
    region_h: int = 53
    threshold_pct: float = 50.0
    read_interval_s: float = 0.5


@dataclass
class LevelOcrSettings:
    region_x: int = 238
    region_y: int = 863
    region_w: int = 109
    region_h: int = 30
    read_interval_s: float = 2.0
    min_confidence: float = 0.4   # 구버전 0.1 → 0.4 상향


@dataclass
class LootSettings:
    region_x: int = 360
    region_y: int = 240
    region_w: int = 1200
    region_h: int = 500
    scan_interval_s: float = 0.3
    click_interval_ms: int = 400
    timeout_ms: int = 8000        # 3s → 8s
    rescan_max: int = 3


@dataclass
class KeyBindings:
    potion: str = "F5"
    scroll: str = "F6"
    tp_book: str = "F7"
    return_book: str = "F8"
    speed_potion: str = "F9"
    potion_cooldown_ms: int = 3000


@dataclass
class WaypointPoint:
    x: int = 0
    y: int = 0
    label: str = ""
    wait_ms: int = 500
    clicks: int = 1
    click_delay_ms: int = 0


@dataclass
class WaypointSettings:
    move_timeout_ms: int = 8000
    idle_timeout_s: float = 3.0
    stuck_start_ratio: float = 0.60
    stuck_ratio: float = 0.25
    stuck_min_baseline: float = 0.30
    stuck_max: int = 3
    points: List[WaypointPoint] = field(default_factory=list)


@dataclass
class DummySettings:
    """허수아비 공격 (leveling 모드)."""
    drag_from_x: int = 1184
    drag_from_y: int = 171
    drag_to_x: int = 1109
    drag_to_y: int = 201
    drag_steps: int = 8
    attack_interval_ms: int = 800
    move_timeout_ms: int = 3000


@dataclass
class TeleportSettings:
    dest_region_x: int = 400
    dest_region_y: int = 150
    dest_region_w: int = 300
    dest_region_h: int = 400
    destination_text: str = "허수아비"
    wait_after_key_ms: int = 800
    wait_after_click_ms: int = 3000
    max_retries: int = 3


@dataclass
class PatrolCombatSettings:
    kill_wait_ms: int = 2500
    attack_interval_ms: int = 800
    max_attacks_per_wp: int = 3


@dataclass
class LevelGoals:
    target_level_dummy: int = 999
    target_level_hunt: int = 999


# ═══════════════════════════════════════════════════════
# 최상위 Settings
# ═══════════════════════════════════════════════════════

@dataclass
class Settings:
    """전체 봇 설정을 담는 단일 객체.

    config.json + config_automation.json 두 파일을 하나로 통합.
    모든 섹션은 기본값이 있으므로 JSON 없이도 동작합니다.
    """
    capture:          CaptureSettings      = field(default_factory=CaptureSettings)
    detection:        DetectionSettings    = field(default_factory=DetectionSettings)
    pico:             PicoSettings         = field(default_factory=PicoSettings)
    tracking:         TrackingSettings     = field(default_factory=TrackingSettings)
    overlay:          OverlaySettings      = field(default_factory=OverlaySettings)
    hp_bar:           HpBarSettings        = field(default_factory=HpBarSettings)
    level_ocr:        LevelOcrSettings     = field(default_factory=LevelOcrSettings)
    loot:             LootSettings         = field(default_factory=LootSettings)
    keys:             KeyBindings          = field(default_factory=KeyBindings)
    dummy:            DummySettings        = field(default_factory=DummySettings)
    teleport:         TeleportSettings     = field(default_factory=TeleportSettings)
    level_goals:      LevelGoals           = field(default_factory=LevelGoals)
    hunt_waypoints:   WaypointSettings     = field(default_factory=WaypointSettings)
    patrol_waypoints: WaypointSettings     = field(default_factory=WaypointSettings)
    patrol_combat:    PatrolCombatSettings = field(default_factory=PatrolCombatSettings)

    # 저장 경로 (외부 비공개)
    _path: str = field(default=DEFAULT_CONFIG_PATH, compare=False, repr=False)

    # ── 로드 / 저장 ─────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "Settings":
        """JSON → Settings. 파일 없으면 기본값으로 생성 후 저장."""
        s = cls()
        s._path = path
        if not os.path.exists(path):
            s.save()
            return s
        with open(path, "r", encoding="utf-8") as f:
            raw: dict = json.load(f)
        _apply(s, raw)
        return s

    def save(self, path: str = "") -> None:
        """Settings → JSON."""
        target = path or self._path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        raw = {
            "capture":          asdict(self.capture),
            "detection":        asdict(self.detection),
            "pico":             asdict(self.pico),
            "tracking":         asdict(self.tracking),
            "overlay":          asdict(self.overlay),
            "hp_bar":           asdict(self.hp_bar),
            "level_ocr":        asdict(self.level_ocr),
            "loot":             asdict(self.loot),
            "keys":             asdict(self.keys),
            "dummy":            asdict(self.dummy),
            "teleport":         asdict(self.teleport),
            "level_goals":      asdict(self.level_goals),
            "hunt_waypoints":   _wp_to_dict(self.hunt_waypoints),
            "patrol_waypoints": _wp_to_dict(self.patrol_waypoints),
            "patrol_combat":    asdict(self.patrol_combat),
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=4)

    # ── 편의 프로퍼티 ────────────────────────────────────

    @property
    def capture_region(self) -> Tuple[int, int, int, int]:
        c = self.capture
        return (c.region_x, c.region_y, c.region_w, c.region_h)

    @property
    def detection_zone_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """(x0,y0,x1,y1). zone 비활성이면 None."""
        d = self.detection
        if not d.zone_enabled:
            return None
        return (
            d.zone_cx - d.zone_half_w,
            d.zone_cy - d.zone_half_h,
            d.zone_cx + d.zone_half_w,
            d.zone_cy + d.zone_half_h,
        )

    @property
    def hp_bar_region(self) -> Tuple[int, int, int, int]:
        h = self.hp_bar
        return (h.region_x, h.region_y, h.region_w, h.region_h)

    @property
    def loot_region(self) -> Tuple[int, int, int, int]:
        lt = self.loot
        return (lt.region_x, lt.region_y, lt.region_w, lt.region_h)


# ═══════════════════════════════════════════════════════
# 내부 헬퍼
# ═══════════════════════════════════════════════════════

def _g(d: dict, key: str, default):
    """dict.get의 짧은 별칭."""
    return d.get(key, default)


def _apply(s: Settings, raw: dict) -> None:
    """raw dict 내용을 Settings 객체에 적용 (누락 시 기존 기본값 유지)."""
    c = raw.get("capture", {})
    s.capture = CaptureSettings(
        monitor_index=_g(c,"monitor_index",s.capture.monitor_index),
        fps=_g(c,"fps",s.capture.fps),
        region_x=_g(c,"region_x",s.capture.region_x),
        region_y=_g(c,"region_y",s.capture.region_y),
        region_w=_g(c,"region_w",s.capture.region_w),
        region_h=_g(c,"region_h",s.capture.region_h),
    )

    d = raw.get("detection", {})
    s.detection = DetectionSettings(
        fps=_g(d,"fps",s.detection.fps),
        templates_dir=_g(d,"templates_dir",s.detection.templates_dir),
        reject_templates_dir=_g(d,"reject_templates_dir",s.detection.reject_templates_dir),
        match_threshold=_g(d,"match_threshold",s.detection.match_threshold),
        scale_factors=_g(d,"scale_factors",s.detection.scale_factors),
        nms_iou_threshold=_g(d,"nms_iou_threshold",s.detection.nms_iou_threshold),
        max_templates=_g(d,"max_templates",s.detection.max_templates),
        zone_enabled=_g(d,"zone_enabled",s.detection.zone_enabled),
        zone_cx=_g(d,"zone_cx",s.detection.zone_cx),
        zone_cy=_g(d,"zone_cy",s.detection.zone_cy),
        zone_half_w=_g(d,"zone_half_w",s.detection.zone_half_w),
        zone_half_h=_g(d,"zone_half_h",s.detection.zone_half_h),
    )

    p = raw.get("pico", {})
    s.pico = PicoSettings(
        enabled=_g(p,"enabled",s.pico.enabled),
        serial_port=_g(p,"serial_port",s.pico.serial_port),
        baudrate=_g(p,"baudrate",s.pico.baudrate),
        click_pulse_ms=_g(p,"click_pulse_ms",s.pico.click_pulse_ms),
        click_offset_x=_g(p,"click_offset_x",s.pico.click_offset_x),
        click_offset_y=_g(p,"click_offset_y",s.pico.click_offset_y),
        lock_confirm_frames=_g(p,"lock_confirm_frames",s.pico.lock_confirm_frames),
        wait_dead_timeout_ms=_g(p,"wait_dead_timeout_ms",s.pico.wait_dead_timeout_ms),
        next_target_cooldown_ms=_g(p,"next_target_cooldown_ms",s.pico.next_target_cooldown_ms),
        target_priority=_g(p,"target_priority",s.pico.target_priority),
        drag_enabled=_g(p,"drag_enabled",s.pico.drag_enabled),
        drag_dx=_g(p,"drag_dx",s.pico.drag_dx),
        drag_dy=_g(p,"drag_dy",s.pico.drag_dy),
        drag_steps=_g(p,"drag_steps",s.pico.drag_steps),
    )

    t = raw.get("tracking", {})
    s.tracking = TrackingSettings(
        max_missing_frames=_g(t,"max_missing_frames",s.tracking.max_missing_frames),
        max_match_distance=_g(t,"max_match_distance",s.tracking.max_match_distance),
    )

    o = raw.get("overlay", {})
    s.overlay = OverlaySettings(
        window_x=_g(o,"window_x",s.overlay.window_x),
        window_y=_g(o,"window_y",s.overlay.window_y),
        show_debug=_g(o,"show_debug",s.overlay.show_debug),
    )

    hp = raw.get("hp_bar", {})
    s.hp_bar = HpBarSettings(
        region_x=_g(hp,"region_x",s.hp_bar.region_x),
        region_y=_g(hp,"region_y",s.hp_bar.region_y),
        region_w=_g(hp,"region_w",s.hp_bar.region_w),
        region_h=_g(hp,"region_h",s.hp_bar.region_h),
        threshold_pct=_g(hp,"threshold_pct",s.hp_bar.threshold_pct),
        read_interval_s=_g(hp,"read_interval_s",s.hp_bar.read_interval_s),
    )

    lv = raw.get("level_ocr", {})
    s.level_ocr = LevelOcrSettings(
        region_x=_g(lv,"region_x",s.level_ocr.region_x),
        region_y=_g(lv,"region_y",s.level_ocr.region_y),
        region_w=_g(lv,"region_w",s.level_ocr.region_w),
        region_h=_g(lv,"region_h",s.level_ocr.region_h),
        read_interval_s=_g(lv,"read_interval_s",s.level_ocr.read_interval_s),
        min_confidence=_g(lv,"min_confidence",s.level_ocr.min_confidence),
    )

    lt = raw.get("loot", {})
    s.loot = LootSettings(
        region_x=_g(lt,"region_x",s.loot.region_x),
        region_y=_g(lt,"region_y",s.loot.region_y),
        region_w=_g(lt,"region_w",s.loot.region_w),
        region_h=_g(lt,"region_h",s.loot.region_h),
        scan_interval_s=_g(lt,"scan_interval_s",s.loot.scan_interval_s),
        click_interval_ms=_g(lt,"click_interval_ms",s.loot.click_interval_ms),
        timeout_ms=_g(lt,"timeout_ms",s.loot.timeout_ms),
        rescan_max=_g(lt,"rescan_max",s.loot.rescan_max),
    )

    k = raw.get("keys", {})
    s.keys = KeyBindings(
        potion=_g(k,"potion",s.keys.potion),
        scroll=_g(k,"scroll",s.keys.scroll),
        tp_book=_g(k,"tp_book",s.keys.tp_book),
        return_book=_g(k,"return_book",s.keys.return_book),
        speed_potion=_g(k,"speed_potion",s.keys.speed_potion),
        potion_cooldown_ms=_g(k,"potion_cooldown_ms",s.keys.potion_cooldown_ms),
    )

    dm = raw.get("dummy", {})
    s.dummy = DummySettings(
        drag_from_x=_g(dm,"drag_from_x",s.dummy.drag_from_x),
        drag_from_y=_g(dm,"drag_from_y",s.dummy.drag_from_y),
        drag_to_x=_g(dm,"drag_to_x",s.dummy.drag_to_x),
        drag_to_y=_g(dm,"drag_to_y",s.dummy.drag_to_y),
        drag_steps=_g(dm,"drag_steps",s.dummy.drag_steps),
        attack_interval_ms=_g(dm,"attack_interval_ms",s.dummy.attack_interval_ms),
        move_timeout_ms=_g(dm,"move_timeout_ms",s.dummy.move_timeout_ms),
    )

    tp = raw.get("teleport", {})
    s.teleport = TeleportSettings(
        dest_region_x=_g(tp,"dest_region_x",s.teleport.dest_region_x),
        dest_region_y=_g(tp,"dest_region_y",s.teleport.dest_region_y),
        dest_region_w=_g(tp,"dest_region_w",s.teleport.dest_region_w),
        dest_region_h=_g(tp,"dest_region_h",s.teleport.dest_region_h),
        destination_text=_g(tp,"destination_text",s.teleport.destination_text),
        wait_after_key_ms=_g(tp,"wait_after_key_ms",s.teleport.wait_after_key_ms),
        wait_after_click_ms=_g(tp,"wait_after_click_ms",s.teleport.wait_after_click_ms),
        max_retries=_g(tp,"max_retries",s.teleport.max_retries),
    )

    lg = raw.get("level_goals", {})
    s.level_goals = LevelGoals(
        target_level_dummy=_g(lg,"target_level_dummy",s.level_goals.target_level_dummy),
        target_level_hunt=_g(lg,"target_level_hunt",s.level_goals.target_level_hunt),
    )

    s.hunt_waypoints   = _wp_from_dict(raw.get("hunt_waypoints",   {}), s.hunt_waypoints)
    s.patrol_waypoints = _wp_from_dict(raw.get("patrol_waypoints", {}), s.patrol_waypoints)

    pc = raw.get("patrol_combat", {})
    s.patrol_combat = PatrolCombatSettings(
        kill_wait_ms=_g(pc,"kill_wait_ms",s.patrol_combat.kill_wait_ms),
        attack_interval_ms=_g(pc,"attack_interval_ms",s.patrol_combat.attack_interval_ms),
        max_attacks_per_wp=_g(pc,"max_attacks_per_wp",s.patrol_combat.max_attacks_per_wp),
    )


def _wp_from_dict(raw: dict, default: WaypointSettings) -> WaypointSettings:
    points = [
        WaypointPoint(
            x=p.get("x",0), y=p.get("y",0),
            label=p.get("label",""),
            wait_ms=p.get("wait_ms",500),
            clicks=p.get("clicks",1),
            click_delay_ms=p.get("click_delay_ms",0),
        )
        for p in raw.get("points", [])
    ]
    return WaypointSettings(
        move_timeout_ms=_g(raw,"move_timeout_ms",default.move_timeout_ms),
        idle_timeout_s=_g(raw,"idle_timeout_s",default.idle_timeout_s),
        stuck_start_ratio=_g(raw,"stuck_start_ratio",default.stuck_start_ratio),
        stuck_ratio=_g(raw,"stuck_ratio",default.stuck_ratio),
        stuck_min_baseline=_g(raw,"stuck_min_baseline",default.stuck_min_baseline),
        stuck_max=_g(raw,"stuck_max",default.stuck_max),
        points=points,
    )


def _wp_to_dict(ws: WaypointSettings) -> dict:
    return {
        "move_timeout_ms":   ws.move_timeout_ms,
        "idle_timeout_s":    ws.idle_timeout_s,
        "stuck_start_ratio": ws.stuck_start_ratio,
        "stuck_ratio":       ws.stuck_ratio,
        "stuck_min_baseline":ws.stuck_min_baseline,
        "stuck_max":         ws.stuck_max,
        "points": [
            {"x":p.x,"y":p.y,"label":p.label,
             "wait_ms":p.wait_ms,"clicks":p.clicks,
             "click_delay_ms":p.click_delay_ms}
            for p in ws.points
        ],
    }
