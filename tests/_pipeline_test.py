"""파이프라인 전체 dry-run 테스트."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.settings import Settings

s = Settings.load()

class FakePico:
    def click(self, x, y, pulse_ms=20): pass
    def drag(self, fx, fy, tx, ty, steps=8): pass
    def key_tap_name(self, name, hold_ms=50): pass

def fake_grab():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

ok = True

# 1. NearestNeighborTracker.from_settings
try:
    from core.tracking import NearestNeighborTracker
    t = NearestNeighborTracker.from_settings(s)
    print('[OK] NearestNeighborTracker.from_settings')
except Exception as e:
    print(f'[FAIL] NearestNeighborTracker: {e}')
    ok = False

# 2. RealtimeTemplateDetector.from_settings
try:
    from core.detection import RealtimeTemplateDetector
    d = RealtimeTemplateDetector.from_settings(s, base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f'[OK] RealtimeTemplateDetector templates={len(d._prepared)}')
except Exception as e:
    print(f'[FAIL] RealtimeTemplateDetector: {e}')
    ok = False

# 3. build_hunt_loop
loop = None
try:
    from modes.hunt_loop import build_hunt_loop
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loop = build_hunt_loop(s, FakePico(), fake_grab, base_dir=base)
    print(f'[OK] build_hunt_loop detector={len(loop._detector._prepared)} loot={loop._loot is not None} hp={loop._hp_reader is not None}')
except Exception as e:
    import traceback
    print(f'[FAIL] build_hunt_loop: {e}')
    traceback.print_exc()
    ok = False

# 4. detect()
if loop:
    try:
        dets = loop._detector.detect(fake_grab())
        print(f'[OK] detect() dets={len(dets)} (검은 프레임=0 정상)')
    except Exception as e:
        print(f'[FAIL] detect(): {e}')
        ok = False

    # 5. tracker.update()
    try:
        enemies = loop._tracker.update([], 0.1)
        print(f'[OK] tracker.update() enemies={len(enemies)} SM={loop._tracker.target_state.name}')
    except Exception as e:
        print(f'[FAIL] tracker.update(): {e}')
        ok = False

    # 6. loot.find()
    if loop._loot:
        try:
            items = loop._loot.find(fake_grab())
            print(f'[OK] loot.find() items={len(items)}')
        except Exception as e:
            print(f'[FAIL] loot.find(): {e}')
            ok = False

    # 7. hp_reader.read()
    if loop._hp_reader:
        try:
            ratio = loop._hp_reader.read(fake_grab())
            print(f'[OK] hp_reader.read() ratio={ratio}')
        except Exception as e:
            print(f'[FAIL] hp_reader.read(): {e}')
            ok = False

print()
print('=== RESULT:', 'ALL OK' if ok else 'FAILED ===')
sys.exit(0 if ok else 1)
