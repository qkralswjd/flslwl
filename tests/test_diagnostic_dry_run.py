"""TEST: Diagnostic Dry-Run 핵심 속성 검증.

검증 범위:
1. NullPicoWorker — click/drag/key_tap 차단 (실제 명령 미전송)
2. NullPicoWorker — is_connected=True (SM 연결 체크 우회)
3. NullPicoWorker — 모든 메서드 호출 후 예외 없음
4. DiagRecord — to_log_lines() 포맷 구조 검증
5. _infer_decision() — 각 perception 조건별 decision 추론
6. NullPico가 PicoSerialWorker 인터페이스를 완전히 대체 가능한지

환경: Windows 실행 불필요. 순수 로직 검증.
"""

import sys
import os
import unittest

# 프로젝트 루트를 PYTHONPATH에 추가
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# NullPicoWorker 테스트
# ──────────────────────────────────────────────────────────────────────────────

class TestNullPicoWorker(unittest.TestCase):
    """NullPicoWorker가 실제 명령을 전송하지 않고 인터페이스를 만족하는지 검증."""

    def setUp(self):
        from pico.null_pico import NullPicoWorker
        self.pico = NullPicoWorker()

    # ── 접속 상태 ────────────────────────────────────────────────────────────
    def test_is_connected_true(self):
        """is_connected는 항상 True — SM의 연결 체크를 우회한다."""
        self.assertTrue(self.pico.is_connected)

    # ── 마우스 명령 차단 ──────────────────────────────────────────────────────
    def test_click_does_not_raise(self):
        """click()이 예외 없이 반환된다."""
        self.pico.click(960, 540, pulse_ms=20)  # 예외 없으면 통과

    def test_click_returns_none(self):
        """click()은 None 반환 (실제 Pico 응답 없음)."""
        result = self.pico.click(0, 0)
        self.assertIsNone(result)

    def test_drag_does_not_raise(self):
        """drag()가 예외 없이 반환된다."""
        self.pico.drag(100, 200, 300, 400, steps=8)

    def test_drag_returns_none(self):
        """drag()는 None 반환."""
        result = self.pico.drag(0, 0, 100, 100)
        self.assertIsNone(result)

    # ── 키보드 명령 차단 ──────────────────────────────────────────────────────
    def test_key_tap_name_does_not_raise(self):
        """key_tap_name()이 예외 없이 반환된다."""
        self.pico.key_tap_name("F5", hold_ms=80)

    def test_key_tap_name_returns_none(self):
        """key_tap_name()은 None 반환."""
        result = self.pico.key_tap_name("F6")
        self.assertIsNone(result)

    def test_key_tap_does_not_raise(self):
        self.pico.key_tap(0x74, hold_ms=50)

    def test_key_down_up_does_not_raise(self):
        self.pico.key_down("F9")
        self.pico.key_up("F9")

    # ── 생명 주기 ────────────────────────────────────────────────────────────
    def test_start_stop_does_not_raise(self):
        """start()/stop()이 예외 없이 동작한다."""
        self.pico.start()
        self.pico.stop()

    def test_stop_target_does_not_raise(self):
        self.pico.stop_target()

    # ── 기타 ─────────────────────────────────────────────────────────────────
    def test_move_to_does_not_raise(self):
        self.pico.move_to(500, 600)

    def test_scroll_does_not_raise(self):
        self.pico.scroll(1)

    def test_all_commands_combined(self):
        """전체 명령 시퀀스가 예외 없이 실행된다."""
        self.pico.start()
        self.pico.key_tap_name("F6", hold_ms=80)   # 두루마리
        self.pico.click(700, 400)                   # 목적지 클릭
        self.pico.drag(960, 600, 960, 400, steps=8) # 공격 드래그
        self.pico.key_tap_name("F5", hold_ms=50)   # 물약
        self.pico.stop_target()
        self.pico.stop()


# ──────────────────────────────────────────────────────────────────────────────
# DiagRecord 포맷 테스트
# ──────────────────────────────────────────────────────────────────────────────

class TestDiagRecord(unittest.TestCase):
    """DiagRecord.to_log_lines() 출력 포맷 검증."""

    def _make_record(self, **kwargs):
        from diagnostic.models import DiagRecord
        rec = DiagRecord()
        for k, v in kwargs.items():
            setattr(rec, k, v)
        return rec

    def test_log_lines_contain_perception_block(self):
        rec = self._make_record(hp_pct=75.5, level=5, is_moving=False,
                                enemy_count=2, tracked_count=2, loot_count=0)
        lines = rec.to_log_lines()
        text = "\n".join(lines)
        self.assertIn("[PERCEPTION]", text)
        self.assertIn("hp=75.5", text)
        self.assertIn("level=5", text)
        self.assertIn("moving=False", text)
        self.assertIn("enemies=2", text)
        self.assertIn("tracked=2", text)
        self.assertIn("loot=0", text)

    def test_log_lines_contain_state_block(self):
        rec = self._make_record(
            hunting_state="HUNTING_10",
            target_id=3,
            target_state="LOCKING",
        )
        lines = rec.to_log_lines()
        text = "\n".join(lines)
        self.assertIn("[STATE]", text)
        self.assertIn("hunting=HUNTING_10", text)
        self.assertIn("target=3", text)
        self.assertIn("target_state=LOCKING", text)

    def test_log_lines_contain_decision_block(self):
        rec = self._make_record(decision="ATTACK", decision_reason="enemy_detected",
                                target_id=7)
        lines = rec.to_log_lines()
        text = "\n".join(lines)
        self.assertIn("[DECISION]", text)
        self.assertIn("ATTACK", text)
        self.assertIn("reason=enemy_detected", text)
        self.assertIn("target_id=7", text)

    def test_log_lines_none_target(self):
        """target_id가 None일 때 '-' 로 표시된다."""
        rec = self._make_record(target_id=None)
        text = "\n".join(rec.to_log_lines())
        self.assertIn("target=-", text)

    def test_log_lines_none_level(self):
        """level이 None일 때 '?' 로 표시된다."""
        rec = self._make_record(level=None)
        text = "\n".join(rec.to_log_lines())
        self.assertIn("level=?", text)


# ──────────────────────────────────────────────────────────────────────────────
# _infer_decision() 테스트
# ──────────────────────────────────────────────────────────────────────────────

class TestInferDecision(unittest.TestCase):
    """perception 조건 → decision 추론 검증."""

    def _rec(self, **kwargs):
        from diagnostic.models import DiagRecord
        rec = DiagRecord()
        rec.hp_pct        = 100.0
        rec.is_moving     = False
        rec.tracked_count = 0
        rec.loot_count    = 0
        rec.hunting_state = "HUNTING_10"
        rec.target_id     = None
        for k, v in kwargs.items():
            setattr(rec, k, v)
        return rec

    def _decide(self, **kwargs):
        from diagnostic.models import infer_decision
        rec = self._rec(**kwargs)
        return infer_decision(rec)

    def test_moving_returns_wait(self):
        decision, reason = self._decide(is_moving=True)
        self.assertEqual(decision, "WAIT")
        self.assertEqual(reason, "scene_moving")

    def test_low_hp_returns_use_potion(self):
        decision, reason = self._decide(hp_pct=40.0, is_moving=False)
        self.assertEqual(decision, "USE_POTION")
        self.assertIn("40.0", reason)

    def test_hp_exactly_50_not_low(self):
        """HP=50.0은 경계값 — 50% 미만이 아니므로 USE_POTION이 아니어야 한다."""
        decision, _ = self._decide(hp_pct=50.0, tracked_count=0,
                                   hunting_state="IDLE", loot_count=0)
        self.assertNotEqual(decision, "USE_POTION")

    def test_enemy_detected_returns_attack(self):
        decision, reason = self._decide(tracked_count=2, hunting_state="HUNTING_10")
        self.assertEqual(decision, "ATTACK")
        self.assertIn("enemy_detected", reason)

    def test_loot_detected_returns_loot(self):
        decision, reason = self._decide(loot_count=1, tracked_count=0,
                                        hunting_state="HUNTING_10")
        self.assertEqual(decision, "LOOT")
        self.assertIn("loot_detected", reason)

    def test_looting_state_returns_loot(self):
        decision, _ = self._decide(hunting_state="LOOTING", loot_count=0)
        self.assertEqual(decision, "LOOT")

    def test_looting_state_with_loot(self):
        decision, reason = self._decide(hunting_state="LOOTING", loot_count=3)
        self.assertEqual(decision, "LOOT")
        self.assertIn("3", reason)

    def test_hunting_no_enemy_returns_move(self):
        decision, reason = self._decide(tracked_count=0, hunting_state="HUNTING_10",
                                        loot_count=0)
        self.assertEqual(decision, "MOVE")
        self.assertIn("hunting_no_enemy", reason)

    def test_idle_returns_wait(self):
        decision, reason = self._decide(hunting_state="IDLE", tracked_count=0,
                                        loot_count=0)
        self.assertEqual(decision, "WAIT")
        self.assertIn("IDLE", reason)

    def test_moving_takes_priority_over_low_hp(self):
        """이동 중이면 HP가 낮아도 WAIT 우선."""
        decision, _ = self._decide(is_moving=True, hp_pct=10.0)
        self.assertEqual(decision, "WAIT")

    def test_low_hp_takes_priority_over_attack(self):
        """HP 낮음은 적 감지보다 우선."""
        decision, _ = self._decide(hp_pct=30.0, tracked_count=5,
                                   hunting_state="HUNTING_10", is_moving=False)
        self.assertEqual(decision, "USE_POTION")


# ──────────────────────────────────────────────────────────────────────────────
# NullPico가 HuntingStateMachine에 주입 가능한지 (구조 호환성)
# ──────────────────────────────────────────────────────────────────────────────

class TestNullPicoSmCompatibility(unittest.TestCase):
    """NullPicoWorker가 HuntingStateMachine의 pico_worker 파라미터로 동작하는지."""

    def test_null_pico_has_required_sm_interface(self):
        """SM이 사용하는 메서드들이 모두 존재한다."""
        from pico.null_pico import NullPicoWorker
        pico = NullPicoWorker()
        # SM 내부에서 호출하는 메서드 목록
        required = [
            "click", "drag", "key_tap_name",
            "start", "stop", "stop_target",
            "is_connected",
        ]
        for attr in required:
            self.assertTrue(
                hasattr(pico, attr),
                f"NullPicoWorker에 '{attr}' 없음",
            )

    def test_null_pico_is_connected_property(self):
        """is_connected가 property로 접근 가능하다."""
        from pico.null_pico import NullPicoWorker
        pico = NullPicoWorker()
        # property 접근 — 예외 없으면 OK
        val = pico.is_connected
        self.assertIsInstance(val, bool)


if __name__ == "__main__":
    unittest.main()
