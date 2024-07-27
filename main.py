
from contextlib import contextmanager
import numpy as np
import cv2

from median_filter import MedianFilter

FILTER_SEC = 1.0
TAPE_SEC = 5.0
STAY_AWAKE_SEC = 5.0

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

class CircularBuffer:
    def __init__(self, n_frames: int, width: int, height: int):
        self.n_frames = n_frames
        self.buffer = np.zeros((
            n_frames, height, width, 
        ), dtype=np.uint8)
        self.push_i = 0
        self.pop_i = 0
    
    def push(self, frame: np.ndarray):
        self.buffer[self.push_i] = frame
        self.push_i = (self.push_i + 1) % self.n_frames
        if self.push_i == self.pop_i:
            self.pop_i = (self.pop_i + 1) % self.n_frames
    
    def Harvest(self):
        while self.pop_i != self.push_i:
            yield self.buffer[self.pop_i]
            self.pop_i = (self.pop_i + 1) % self.n_frames

def FramesInCap(cap: cv2.VideoCapture, apart: int):
    assert isinstance(apart, int) and apart >= 1
    buf = []
    while True:
        try:
            feed_is_alive, frame = cap.read()
            assert feed_is_alive
            buf.append(frame)
            if len(buf) == apart + 1:
                yield frame, buf.pop(0)
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
        medFilter = MedianFilter(round(FILTER_SEC * fps))
        contextBuf = CircularBuffer(round(CONTEXT_PAD_SEC * fps), *size)
        fourcc: int = cv2.VideoWriter_fourcc(*FOUR_CC)
        with VideoWriter(f'./out.{EXT}', fourcc, fps, size) as out:
            sleeping, inactivity = False, 0.0
            last_frame = np.zeros((height, width, 3), dtype=np.uint8)
            for frame in FramesInCap(cap):
                change = (np.abs(frame - last_frame).astype(np.float32) / 256.0).mean()
                print(f'{change : .2e}')
                last_frame = frame

if __name__ == "__main__":
    main()
