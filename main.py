from contextlib import contextmanager
import typing as tp
import time

import numpy as np
import cv2

from median_filter import MedianFilter

CHANGE_THRESHOLD = 0.5

FILTER_SEC = 0.0
FILTER_N_FRAMES = 3
TAPE_SEC = 5.0
STAY_AWAKE_SEC = 30.0
STAY_AWAKE_SEC = 3.0

# FOUR_CC, EXT = 'XVID', 'avi'
# FOUR_CC, EXT = 'H264', 'mp4'
FOUR_CC, EXT = 'mp4v', 'mp4'

@contextmanager
def VideoCapture(*args, **kw):
    cap = cv2.VideoCapture(*args, **kw)
    try:
        yield cap
    finally:
        cap.release()

@contextmanager
def VideoWriter(*args, **kw):
    out = cv2.VideoWriter(*args, **kw)
    try:
        yield out
    finally:
        out.release()

def LastFramesInCap(
    cap: cv2.VideoCapture, fps: float, 
    out: tp.List[np.ndarray], 
):
    assert not out
    tape_len = round(TAPE_SEC * fps)
    while True:
        try:
            feed_is_alive, frame = cap.read()
            assert feed_is_alive
            for thickness, color in ((3, 0), (2, 255)):
                cv2.putText(
                    frame, time.ctime(), (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, 
                    color=(color, color, color), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA, 
                )
            out.append(frame)
            if len(out) > tape_len:
                out.pop(0)
                yield
        except KeyboardInterrupt:
            print('bye')
            break

def main():
    with VideoCapture(0) as cap:
        assert cap.isOpened()
        width  = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = round(cap.get(cv2.CAP_PROP_FPS))
        size = (width, height)
        medFilter = MedianFilter(FILTER_N_FRAMES + round(FILTER_SEC * fps))
        fourcc: int = cv2.VideoWriter_fourcc(*FOUR_CC)
        with VideoWriter(f'./out.{EXT}', fourcc, fps, size) as out:
            sleeping, inactivity = False, 0.0
            tape: tp.List[np.ndarray] = []
            print('Starting...')
            for _ in LastFramesInCap(cap, fps, tape):
                change = (np.abs(tape[-1] - tape[0]).astype(np.float32) / 256.0).mean()
                smoothed_change = medFilter.next(change)
                if smoothed_change is None:
                    continue
                # print(f'\r {smoothed_change = :.5f} {inactivity = :.1f}', end='', flush=True)
                if sleeping:
                    if smoothed_change > CHANGE_THRESHOLD:
                        # print('\n waking up')
                        sleeping, inactivity = False, 0.0
                        for frame in tape:
                            out.write(frame)
                else:
                    out.write(tape[-1])
                    if smoothed_change > CHANGE_THRESHOLD:
                        inactivity = 0.0
                    else:
                        inactivity += 1 / fps
                        if inactivity > STAY_AWAKE_SEC:
                            # print('\n going to sleep')
                            sleeping, inactivity = True, -1.0

if __name__ == "__main__":
    main()
