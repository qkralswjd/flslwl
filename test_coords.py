"""좌표 테스트 — HP바 / 레벨 / 텔레포트창 영역 캡처 확인"""
import mss
import cv2
import numpy as np

with mss.MSS() as sct:
    mon = sct.monitors[2]
    shot = sct.grab(mon)
    frame = np.array(shot)[:, :, :3]

# HP 바 영역
hp = frame[841:841+52, 553:553+327]
cv2.imwrite('test_hp.png', hp)

# 레벨 영역
lv = frame[858:858+41, 240:240+86]
cv2.imwrite('test_level.png', lv)

# 텔레포트 창 영역
tp = frame[49:49+564, 264:264+387]
cv2.imwrite('test_teleport.png', tp)

print('저장 완료!')
print('  test_hp.png        <- HP 바 영역')
print('  test_level.png     <- 레벨 표시 영역')
print('  test_teleport.png  <- 텔레포트 목적지창 영역')
print('')
print('이미지 3개를 열어서 올바른 위치인지 확인하세요.')
