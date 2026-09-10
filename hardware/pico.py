"""hardware/pico.py — Pico HID 워커 (실제 + 더미).

이 모듈 하나에 두 클래스를 함께 둔다:
  PicoWorker     — threading 기반 실제 시리얼 통신 워커
  NullPicoWorker — duck-typing 더미 (Dry-Run / 테스트 격리)

프로토콜 (텍스트, 줄바꿈 구분):
  PC → Pico : PING
  Pico → PC : PONG
  PC → Pico : MOVE:<dx>:<dy>
  PC → Pico : CLICK:<pulse_ms>
  PC → Pico : PRESS / RELEASE      (드래그 버튼 유지/해제)
  PC → Pico : KEYDOWN:<keycode>
  PC → Pico : KEYUP:<keycode>
  PC → Pico : STOP
  Pico → PC : OK:<CMD> | ERR:<CMD>

기존 pico_serial.py 대비 개선점:
  - PicoWorker / NullPicoWorker 단일 모듈로 통합
  - from_settings() 팩토리 추가
  - click_current_pos() 공개 메서드 추가 (NullPicoWorker에도 존재)
  - is_idle 프로퍼티 NullPicoWorker에도 구현
  - 연결 상태 is_connected 타입 힌트 명확화
  - KEY_NAME_MAP 이 모듈에서 직접 제공 (외부에서 import 가능)
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger("pico")

# ── 타임아웃 상수 ──────────────────────────────────────────────────────────
PING_INTERVAL_S  = 1.0
PONG_TIMEOUT_S   = 30.0   # _do_drag 최대 실행시간(~6.4 s) 대응
ACK_TIMEOUT_S    = 1.0    # MOVE 응답 대기

# ── 폐루프 이동 보정 ────────────────────────────────────────────────────────
CORRECTION_TOLERANCE_PX = 4
MAX_CORRECTION_ITERS    = 20
CORRECTION_DAMPING      = 0.35


# ════════════════════════════════════════════════════════════════════════════
# HID 키코드 매핑
# ════════════════════════════════════════════════════════════════════════════

KEY_NAME_MAP: dict[str, int] = {
    # 기능키
    "F1": 0x3A, "F2": 0x3B, "F3": 0x3C, "F4": 0x3D,
    "F5": 0x3E, "F6": 0x3F, "F7": 0x40, "F8": 0x41,
    "F9": 0x42, "F10": 0x43, "F11": 0x44, "F12": 0x45,
    # 특수키
    "ENTER": 0x28, "ESCAPE": 0x29, "ESC": 0x29,
    "SPACE": 0x2C, "BACKSPACE": 0x2A, "TAB": 0x2B,
    "DELETE": 0x4C, "INSERT": 0x49,
    "HOME": 0x4A, "END": 0x4D,
    "PAGEUP": 0x4B, "PAGEDOWN": 0x4E,
    # 방향키
    "UP": 0x52, "DOWN": 0x51, "LEFT": 0x50, "RIGHT": 0x4F,
    # 알파벳
    "A": 0x04, "B": 0x05, "C": 0x06, "D": 0x07,
    "E": 0x08, "F": 0x09, "G": 0x0A, "H": 0x0B,
    "I": 0x0C, "J": 0x0D, "K": 0x0E, "L": 0x0F,
    "M": 0x10, "N": 0x11, "O": 0x12, "P": 0x13,
    "Q": 0x14, "R": 0x15, "S": 0x16, "T": 0x17,
    "U": 0x18, "V": 0x19, "W": 0x1A, "X": 0x1B,
    "Y": 0x1C, "Z": 0x1D,
    # 숫자
    "1": 0x1E, "2": 0x1F, "3": 0x20, "4": 0x21, "5": 0x22,
    "6": 0x23, "7": 0x24, "8": 0x25, "9": 0x26, "0": 0x27,
}


def list_serial_ports() -> list[str]:
    """현재 연결된 COM 포트 목록을 반환한다."""
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()]
    except ImportError:
        return []


def _get_cursor_pos() -> tuple[int, int]:
    """OS 커서 위치 조회 (Windows ctypes). 마우스를 직접 움직이지 않는다."""
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# ════════════════════════════════════════════════════════════════════════════
# PicoWorker — 실제 시리얼 통신 워커
# ════════════════════════════════════════════════════════════════════════════

class PicoWorker:
    """threading 기반 Pico HID 시리얼 워커.

    콜백:
        on_connected()                        — 연결 성공
        on_disconnected(reason: str)          — 연결 끊김
        on_log(level: str, message: str)      — 로그 메시지
        on_command_result(cmd: str, ok: bool) — 명령 응답

    사용 예::

        worker = PicoWorker.from_settings(settings)
        worker.start()
        worker.click(640, 400)
        worker.key_tap_name("F1")
        worker.stop()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        monitor_offset_x: int = 0,
        monitor_offset_y: int = 0,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
        on_command_result: Optional[Callable[[str, bool], None]] = None,
    ) -> None:
        self.port     = port
        self.baudrate = baudrate
        self._monitor_offset_x = monitor_offset_x
        self._monitor_offset_y = monitor_offset_y

        self._on_connected      = on_connected      or (lambda: None)
        self._on_disconnected   = on_disconnected   or (lambda r: None)
        self._on_log            = on_log            or (lambda lv, msg: logger.info("[%s] %s", lv, msg))
        self._on_command_result = on_command_result or (lambda cmd, ok: None)

        self._running        = False
        self._ser            = None
        self._out_queue: queue.Queue[str] = queue.Queue()
        self._rx_buffer      = b""
        self._last_ping_sent = 0.0
        self._last_pong_recv = 0.0
        self._is_connected   = False
        self._thread: Optional[threading.Thread] = None

    # ── 팩토리 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
        on_command_result: Optional[Callable[[str, bool], None]] = None,
    ) -> "PicoWorker":
        """Settings에서 PicoWorker를 생성한다."""
        p = settings.pico
        c = settings.capture
        return cls(
            port=p.serial_port,
            baudrate=p.baudrate,
            monitor_offset_x=c.region_x,
            monitor_offset_y=c.region_y,
            on_connected=on_connected,
            on_disconnected=on_disconnected,
            on_log=on_log,
            on_command_result=on_command_result,
        )

    # ── 공개 API ───────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_idle(self) -> bool:
        """출력 큐가 비면 True (모든 명령 처리 완료)."""
        return self._out_queue.empty()

    def start(self) -> None:
        """워커 스레드를 시작한다."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="PicoSerial", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """워커 스레드 중지를 요청한다."""
        self._running = False

    def enqueue(self, command: str) -> None:
        """명령을 큐에 넣는다 (스레드 안전)."""
        self._out_queue.put(command)

    # ── 마우스 명령 ────────────────────────────────────────────────────────

    def click(self, target_x: int, target_y: int, pulse_ms: int = 20) -> None:
        """절대 화면 좌표로 이동 후 클릭한다."""
        self.enqueue(f"__MOVECLICK__:{int(target_x)}:{int(target_y)}:{int(pulse_ms)}")

    def click_current_pos(self, pulse_ms: int = 20) -> None:
        """현재 커서 위치에서 이동 없이 클릭만 한다."""
        self.enqueue(f"CLICK:{int(pulse_ms)}")

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        steps: int = 8,
    ) -> None:
        """드래그 명령을 큐에 넣는다."""
        self.enqueue(
            f"__DRAG__:{int(from_x)}:{int(from_y)}:{int(to_x)}:{int(to_y)}:{int(steps)}"
        )

    def stop_target(self) -> None:
        """Pico에 STOP 명령을 보낸다 (비상 정지)."""
        self.enqueue("STOP")

    # ── 키보드 명령 ────────────────────────────────────────────────────────

    def key_tap(self, keycode: int, hold_ms: int = 50) -> None:
        """HID 키코드로 키를 눌렀다 뗀다.

        Args:
            keycode: HID 키코드 (예: 0x3A=F1, 0x28=Enter)
            hold_ms: 누름 유지 시간(ms)
        """
        self.enqueue(f"__KEYTAP__:{int(keycode)}:{int(hold_ms)}")

    def key_down(self, keycode: int) -> None:
        """키를 누른 상태로 유지한다 (해제 없음)."""
        self.enqueue(f"KEYDOWN:{int(keycode)}")

    def key_up(self, keycode: int) -> None:
        """누른 키를 해제한다."""
        self.enqueue(f"KEYUP:{int(keycode)}")

    def key_tap_name(self, key_name: str, hold_ms: int = 50) -> None:
        """키 이름으로 탭한다 (예: 'F1', 'ENTER', 'SPACE').

        Args:
            key_name: KEY_NAME_MAP 키 (대소문자 무시)
            hold_ms: 누름 유지 시간(ms)
        """
        keycode = KEY_NAME_MAP.get(key_name.upper())
        if keycode is None:
            logger.warning("[Pico] 알 수 없는 키 이름: %s", key_name)
            return
        self.key_tap(keycode, hold_ms)

    # ── 내부 워커 루프 ─────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import serial
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0)
        except Exception as exc:
            self._on_disconnected(f"포트 열기 실패 {self.port}: {exc}")
            return

        self._is_connected   = True
        self._last_pong_recv = time.monotonic()
        self._on_connected()
        self._on_log("INFO", f"Pico 연결됨: {self.port}")

        try:
            while self._running:
                now = time.monotonic()

                # heartbeat PING
                if now - self._last_ping_sent >= PING_INTERVAL_S:
                    self._write_line("PING")
                    self._last_ping_sent = now

                # heartbeat 타임아웃
                if now - self._last_pong_recv >= PONG_TIMEOUT_S:
                    self._on_disconnected("Pico heartbeat timeout")
                    break

                # 큐 명령 처리
                try:
                    while True:
                        cmd = self._out_queue.get_nowait()
                        self._on_log("DEBUG", f"큐 꺼냄: {cmd}")
                        if cmd.startswith("__MOVECLICK__:"):
                            _, x_s, y_s, pulse_s = cmd.split(":")
                            self._do_move_click(int(x_s), int(y_s), int(pulse_s))
                        elif cmd.startswith("__DRAG__:"):
                            # 최신 drag만 사용 (오래된 것 드레인)
                            latest = cmd
                            while not self._out_queue.empty():
                                peek = self._out_queue.get_nowait()
                                if peek.startswith("__DRAG__:"):
                                    self._on_log("DEBUG", f"오래된 drag 버림: {latest}")
                                    latest = peek
                                else:
                                    self._write_line(peek)
                            cmd = latest
                            _, fx_s, fy_s, tx_s, ty_s, steps_s = cmd.split(":")
                            self._do_drag(int(fx_s), int(fy_s), int(tx_s), int(ty_s), int(steps_s))
                        elif cmd.startswith("__KEYTAP__:"):
                            _, kc_s, hold_s = cmd.split(":")
                            self._do_key_tap(int(kc_s), int(hold_s))
                        else:
                            self._write_line(cmd)
                except queue.Empty:
                    pass

                # 수신 처리
                try:
                    import serial
                    waiting = self._ser.in_waiting
                    if waiting:
                        self._rx_buffer += self._ser.read(waiting)
                        while b"\n" in self._rx_buffer:
                            line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                            self._handle_line(line.decode("utf-8", errors="ignore").strip())
                except Exception as exc:
                    self._on_disconnected(f"수신 오류: {exc}")
                    break

                time.sleep(0.01)
        finally:
            self._is_connected = False
            if self._ser and self._ser.is_open:
                try:
                    self._ser.close()
                except Exception:
                    pass

    def _write_line(self, line: str) -> None:
        if not self._ser:
            return
        try:
            self._ser.write((line + "\n").encode("utf-8"))
        except Exception as exc:
            self._on_disconnected(f"전송 오류: {exc}")
            self._running = False

    def _move_to(
        self,
        target_x: int,
        target_y: int,
        tolerance: int = CORRECTION_TOLERANCE_PX,
        max_iters: int = MAX_CORRECTION_ITERS,
    ) -> bool:
        """폐루프 절대 좌표 이동 (GetCursorPos + 댐핑 보정)."""
        abs_x = target_x + self._monitor_offset_x
        abs_y = target_y + self._monitor_offset_y
        for i in range(max_iters):
            try:
                cur_x, cur_y = _get_cursor_pos()
            except Exception as exc:
                self._on_log("ERROR", f"GetCursorPos 실패: {exc}")
                return False
            dx = abs_x - cur_x
            dy = abs_y - cur_y
            self._on_log(
                "DEBUG",
                f"_move_to iter={i} cur=({cur_x},{cur_y}) "
                f"target=({abs_x},{abs_y}) delta=({dx},{dy})",
            )
            if abs(dx) <= tolerance and abs(dy) <= tolerance:
                return True
            send_dx = round(dx * CORRECTION_DAMPING) or (1 if dx > 0 else -1)
            send_dy = round(dy * CORRECTION_DAMPING) or (1 if dy > 0 else -1)
            self._write_line(f"MOVE:{send_dx}:{send_dy}")
            self._wait_for_ack("OK:MOVE", "ERR:MOVE")
            time.sleep(0.02)
        self._on_log("ERROR", f"_move_to 미수렴: target=({target_x},{target_y})")
        return False

    def _do_move_click(self, target_x: int, target_y: int, pulse_ms: int) -> None:
        converged = self._move_to(target_x, target_y)
        if not converged:
            self._on_log("ERROR", f"이동 보정 미수렴: ({target_x},{target_y})")
        self._write_line(f"CLICK:{pulse_ms}")
        ok = self._wait_for_ack("OK:CLICK", "ERR:CLICK")
        self._on_command_result("CLICK", bool(ok))

    def _do_drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        steps: int,
    ) -> None:
        if not self._move_to(from_x, from_y):
            self._on_log("ERROR", f"드래그 시작점 미도달: ({from_x},{from_y})")
            self._on_command_result("DRAG", False)
            return
        self._write_line("PRESS")
        if not self._wait_for_ack("OK:PRESS", "ERR:PRESS"):
            self._on_command_result("DRAG", False)
            return
        time.sleep(0.05)
        steps = max(1, steps)
        for i in range(1, steps + 1):
            wx = round(from_x + (to_x - from_x) * i / steps)
            wy = round(from_y + (to_y - from_y) * i / steps)
            is_last = i == steps
            self._move_to(
                wx, wy,
                tolerance=CORRECTION_TOLERANCE_PX if is_last else 8,
                max_iters=MAX_CORRECTION_ITERS if is_last else 5,
            )
            time.sleep(0.03)
        self._write_line("RELEASE")
        ok = self._wait_for_ack("OK:RELEASE", "ERR:RELEASE")
        self._on_command_result("DRAG", bool(ok))

    def _do_key_tap(self, keycode: int, hold_ms: int) -> None:
        self._write_line(f"KEYDOWN:{keycode}")
        ok = self._wait_for_ack("OK:KEYDOWN", "ERR:KEYDOWN")
        if ok:
            time.sleep(hold_ms / 1000.0)
            self._write_line(f"KEYUP:{keycode}")
            self._wait_for_ack("OK:KEYUP", "ERR:KEYUP")
        self._on_command_result("KEY", bool(ok))

    def _wait_for_ack(
        self,
        ok_line: str,
        err_line: str,
    ) -> bool | None:
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                waiting = self._ser.in_waiting
            except Exception as exc:
                self._on_disconnected(f"수신 오류: {exc}")
                return None
            if waiting:
                self._rx_buffer += self._ser.read(waiting)
                while b"\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="ignore").strip()
                    if text == "PONG":
                        self._last_pong_recv = time.monotonic()
                    elif text == ok_line:
                        return True
                    elif text == err_line:
                        return False
                    elif text:
                        self._handle_line(text)
            time.sleep(0.005)
        return None

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        if line == "PONG":
            self._last_pong_recv = time.monotonic()
        elif line.startswith("OK:"):
            self._on_command_result(line[3:], True)
        elif line.startswith("ERR:"):
            self._on_command_result(line[4:], False)
            self._on_log("ERROR", f"Pico 오류: {line}")


# ════════════════════════════════════════════════════════════════════════════
# NullPicoWorker — duck-typing 더미 워커 (Dry-Run / 테스트 격리)
# ════════════════════════════════════════════════════════════════════════════

class NullPicoWorker:
    """PicoWorker 인터페이스와 동일하지만 실제 명령을 전송하지 않는다.

    Dry-Run 모드 또는 유닛 테스트에서 PicoWorker 대신 주입한다.
    모든 메서드는 호출 사실을 로그에 기록하고 즉시 반환한다.

    기존 null_pico.py 대비 개선:
      - is_idle 프로퍼티 추가 (항상 True)
      - click_current_pos() 추가
      - from_settings() 팩토리 추가
    """

    # ── 팩토리 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: "Settings") -> "NullPicoWorker":  # noqa: ARG003
        return cls()

    # ── 상태 ───────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """항상 True — 상태머신이 연결 체크를 우회하도록."""
        return True

    @property
    def is_idle(self) -> bool:
        """항상 True — 큐 없음."""
        return True

    # ── 생명 주기 ──────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("[NullPico] start() — no-op (dry-run)")

    def stop(self) -> None:
        logger.info("[NullPico] stop() — no-op (dry-run)")

    def stop_target(self) -> None:
        logger.info("[NullPico] stop_target() — no-op (dry-run)")

    def enqueue(self, command: str) -> None:
        logger.debug("[NullPico] BLOCKED enqueue(%r)", command)

    # ── 마우스 명령 ────────────────────────────────────────────────────────

    def click(self, target_x: int, target_y: int, pulse_ms: int = 20) -> None:
        logger.debug(
            "[NullPico] BLOCKED click(%d, %d, pulse_ms=%d)",
            target_x, target_y, pulse_ms,
        )

    def click_current_pos(self, pulse_ms: int = 20) -> None:
        logger.debug("[NullPico] BLOCKED click_current_pos(pulse_ms=%d)", pulse_ms)

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        steps: int = 8,
    ) -> None:
        logger.debug(
            "[NullPico] BLOCKED drag(%d,%d → %d,%d, steps=%d)",
            from_x, from_y, to_x, to_y, steps,
        )

    # ── 키보드 명령 ────────────────────────────────────────────────────────

    def key_tap(self, keycode: int, hold_ms: int = 50) -> None:
        logger.debug(
            "[NullPico] BLOCKED key_tap(keycode=0x%02X, hold_ms=%d)",
            keycode, hold_ms,
        )

    def key_down(self, keycode: int) -> None:
        logger.debug("[NullPico] BLOCKED key_down(0x%02X)", keycode)

    def key_up(self, keycode: int) -> None:
        logger.debug("[NullPico] BLOCKED key_up(0x%02X)", keycode)

    def key_tap_name(self, key_name: str, hold_ms: int = 50) -> None:
        logger.debug(
            "[NullPico] BLOCKED key_tap_name(%r, hold_ms=%d)",
            key_name, hold_ms,
        )
