"""Pico와의 시리얼 통신 워커 스레드 (threading 기반, PySide6 미사용).

프로토콜 (텍스트, 줄바꿈 구분):
    PC -> Pico : PING
    Pico -> PC : PONG
    PC -> Pico : MOVE:<dx>:<dy>       (상대 HID 마우스 이동)
    PC -> Pico : CLICK:<pulse_ms>     (좌클릭 pulse)
    PC -> Pico : PRESS / RELEASE      (드래그용 버튼 유지/해제)
    PC -> Pico : KEY:<keycode>        (키보드 단일 키 입력)
    PC -> Pico : KEYDOWN:<keycode>    (키보드 키 누름)
    PC -> Pico : KEYUP:<keycode>      (키보드 키 해제)
    PC -> Pico : STOP                 (비상 정지)
    Pico -> PC : OK:<CMD> | ERR:<CMD>

키코드 예시:
    0x04=a, 0x05=b ... 0x1D=z
    0x3A=F1, 0x3B=F2 ... 0x45=F12
    0x28=Enter, 0x29=Escape, 0x2C=Space

PC는 마우스를 직접 클릭하지 않습니다.
절대 좌표를 받으면 GetCursorPos로 현재 위치를 읽어
델타(dx, dy)만 Pico에 전송합니다.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import queue
import threading
import time
from typing import Callable, Optional

import serial
import serial.tools.list_ports

logger = logging.getLogger("pico_serial")

PING_INTERVAL_S  = 1.0
PONG_TIMEOUT_S   = 30.0  # _do_drag 최대 실행시간(~6.4s) 대응 (기존 3.0에서 상향)
ACK_TIMEOUT_S    = 0.3

CORRECTION_TOLERANCE_PX = 4
MAX_CORRECTION_ITERS     = 20
CORRECTION_DAMPING       = 0.35


def _get_cursor_pos() -> tuple[int, int]:
    """OS 커서 위치를 읽기 전용으로 조회 (마우스를 직접 움직이지 않음)."""
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def list_serial_ports() -> list[str]:
    """현재 연결된 COM 포트 목록을 반환합니다."""
    return [p.device for p in serial.tools.list_ports.comports()]


class PicoSerialWorker:
    """threading 기반 Pico 시리얼 워커.

    콜백:
        on_connected()                        - 연결 성공
        on_disconnected(reason: str)          - 연결 끊김
        on_log(level: str, message: str)      - 로그 메시지
        on_command_result(cmd: str, ok: bool) - 명령 응답
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        on_connected: Optional[Callable] = None,
        on_disconnected: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
        on_command_result: Optional[Callable[[str, bool], None]] = None,
    ):
        self.port     = port
        self.baudrate = baudrate

        self._on_connected      = on_connected      or (lambda: None)
        self._on_disconnected   = on_disconnected   or (lambda r: None)
        self._on_log            = on_log            or (lambda lv, msg: logger.info("[%s] %s", lv, msg))
        self._on_command_result = on_command_result or (lambda cmd, ok: None)

        self._running        = False
        self._ser: Optional[serial.Serial] = None
        self._out_queue: queue.Queue[str]  = queue.Queue()
        self._rx_buffer      = b""
        self._last_ping_sent = 0.0
        self._last_pong_recv = 0.0
        self._is_connected   = False
        self._thread: Optional[threading.Thread] = None

    # ── 공개 API (다른 스레드에서 안전하게 호출 가능) ──────────────────────

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def start(self) -> None:
        """워커 스레드를 시작합니다."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="PicoSerial", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """워커 스레드 중지를 요청합니다."""
        self._running = False

    def enqueue(self, command: str) -> None:
        """명령을 큐에 넣습니다 (스레드 안전)."""
        self._out_queue.put(command)

    def click(self, target_x: int, target_y: int, pulse_ms: int = 20) -> None:
        """절대 화면 좌표로 이동 후 클릭합니다."""
        self.enqueue(f"__MOVECLICK__:{int(target_x)}:{int(target_y)}:{int(pulse_ms)}")

    def click_current_pos(self, pulse_ms: int = 20) -> None:
        """현재 커서 위치에서 그냥 클릭만 합니다 (이동 없음)."""
        self.enqueue(f"CLICK:{int(pulse_ms)}")

    def drag(self, from_x: int, from_y: int, to_x: int, to_y: int, steps: int = 8) -> None:
        """드래그 명령을 큐에 넣습니다."""
        self.enqueue(f"__DRAG__:{int(from_x)}:{int(from_y)}:{int(to_x)}:{int(to_y)}:{int(steps)}")

    def stop_target(self) -> None:
        """Pico에 STOP 명령을 보냅니다 (비상 정지)."""
        self.enqueue("STOP")

    # ── 키보드 API ────────────────────────────────────────────────────

    def key_tap(self, keycode: int, hold_ms: int = 50) -> None:
        """키보드 키를 눌렀다 뗍니다 (HID keycode).
        
        Args:
            keycode: HID 키코드 (예: 0x3A=F1, 0x28=Enter, 0x04=a)
            hold_ms: 키를 누르고 있는 시간 (ms)
        """
        self.enqueue(f"__KEYTAP__:{int(keycode)}:{int(hold_ms)}")

    def key_down(self, keycode: int) -> None:
        """키보드 키를 누릅니다 (해제 없음)."""
        self.enqueue(f"KEYDOWN:{int(keycode)}")

    def key_up(self, keycode: int) -> None:
        """키보드 키를 해제합니다."""
        self.enqueue(f"KEYUP:{int(keycode)}")

    def key_tap_name(self, key_name: str, hold_ms: int = 50) -> None:
        """키 이름으로 탭합니다 (예: 'F1', 'enter', 'escape', 'space').
        
        Args:
            key_name: 키 이름 문자열
            hold_ms: 누름 유지 시간 (ms)
        """
        keycode = KEY_NAME_MAP.get(key_name.upper())
        if keycode is None:
            logger.warning(f"[Pico] 알 수 없는 키 이름: {key_name}")
            return
        self.key_tap(keycode, hold_ms)

    # ── 내부 워커 루프 ────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0)
        except serial.SerialException as exc:
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

                # heartbeat 타임아웃 감지
                if now - self._last_pong_recv >= PONG_TIMEOUT_S:
                    self._on_disconnected("Pico heartbeat timeout")
                    break

                # 큐에 쌓인 명령 처리
                try:
                    while True:
                        cmd = self._out_queue.get_nowait()
                        self._on_log("DEBUG", f"큐 꺼냄: {cmd}")
                        if cmd.startswith("__MOVECLICK__:"):
                            _, x_s, y_s, pulse_s = cmd.split(":")
                            self._do_move_click(int(x_s), int(y_s), int(pulse_s))
                        elif cmd.startswith("__DRAG__:"):
                            _, fx_s, fy_s, tx_s, ty_s, steps_s = cmd.split(":")
                            self._on_log("DEBUG", f"_do_drag 시작: ({fx_s},{fy_s})→({tx_s},{ty_s}) steps={steps_s}")
                            self._do_drag(int(fx_s), int(fy_s), int(tx_s), int(ty_s), int(steps_s))
                        elif cmd.startswith("__KEYTAP__:"):
                            _, kc_s, hold_s = cmd.split(":")
                            self._do_key_tap(int(kc_s), int(hold_s))
                        else:
                            self._write_line(cmd)
                except queue.Empty:
                    pass

                # 수신 데이터 처리
                try:
                    waiting = self._ser.in_waiting
                    if waiting:
                        self._rx_buffer += self._ser.read(waiting)
                        while b"\n" in self._rx_buffer:
                            line, self._rx_buffer = self._rx_buffer.split(b"\n", 1)
                            self._handle_line(line.decode("utf-8", errors="ignore").strip())
                except serial.SerialException as exc:
                    self._on_disconnected(f"수신 오류: {exc}")
                    break

                time.sleep(0.01)
        finally:
            self._is_connected = False
            if self._ser and self._ser.is_open:
                try:
                    self._ser.close()
                except serial.SerialException:
                    pass

    def _write_line(self, line: str) -> None:
        if not self._ser:
            return
        try:
            self._ser.write((line + "\n").encode("utf-8"))
        except serial.SerialException as exc:
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
        for i in range(max_iters):
            try:
                cur_x, cur_y = _get_cursor_pos()
            except Exception as exc:
                self._on_log("ERROR", f"GetCursorPos 실패: {exc}")
                return False
            dx = target_x - cur_x
            dy = target_y - cur_y
            self._on_log("DEBUG", f"_move_to iter={i} cur=({cur_x},{cur_y}) target=({target_x},{target_y}) dx={dx} dy={dy}")
            if abs(dx) <= tolerance and abs(dy) <= tolerance:
                self._on_log("DEBUG", f"_move_to 수렴 → ({target_x},{target_y})")
                return True
            send_dx = round(dx * CORRECTION_DAMPING) or (1 if dx > 0 else -1)
            send_dy = round(dy * CORRECTION_DAMPING) or (1 if dy > 0 else -1)
            self._write_line(f"MOVE:{send_dx}:{send_dy}")
            ack = self._wait_for_ack("OK:MOVE", "ERR:MOVE")
            self._on_log("DEBUG", f"_move_to MOVE ACK={ack}")
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

    def _do_drag(self, from_x: int, from_y: int, to_x: int, to_y: int, steps: int) -> None:
        if not self._move_to(from_x, from_y):
            self._on_log("ERROR", f"드래그 시작점 미도달: ({from_x},{from_y})")
            self._on_command_result("DRAG", False)
            return
        self._write_line("PRESS")
        if not self._wait_for_ack("OK:PRESS", "ERR:PRESS"):
            self._on_log("ERROR", "PRESS 거부됨")
            self._on_command_result("DRAG", False)
            return
        time.sleep(0.05)
        steps = max(1, steps)
        for i in range(1, steps + 1):
            wx = round(from_x + (to_x - from_x) * i / steps)
            wy = round(from_y + (to_y - from_y) * i / steps)
            is_last = i == steps
            self._move_to(wx, wy,
                          tolerance=CORRECTION_TOLERANCE_PX if is_last else 8,
                          max_iters=MAX_CORRECTION_ITERS if is_last else 5)
            time.sleep(0.03)
        self._write_line("RELEASE")
        ok = self._wait_for_ack("OK:RELEASE", "ERR:RELEASE")
        self._on_command_result("DRAG", bool(ok))

    def _wait_for_ack(self, ok_line: str, err_line: str) -> bool | None:
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                waiting = self._ser.in_waiting
            except serial.SerialException as exc:
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

    def _do_key_tap(self, keycode: int, hold_ms: int) -> None:
        """키보드 키를 눌렀다 뗍니다."""
        self._write_line(f"KEYDOWN:{keycode}")
        ok = self._wait_for_ack("OK:KEYDOWN", "ERR:KEYDOWN")
        if ok:
            time.sleep(hold_ms / 1000.0)
            self._write_line(f"KEYUP:{keycode}")
            self._wait_for_ack("OK:KEYUP", "ERR:KEYUP")
        self._on_command_result("KEY", bool(ok))

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


# ── HID 키코드 이름 매핑 ───────────────────────────────────────────────
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
    # 알파벳 (소문자로도 인식)
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
