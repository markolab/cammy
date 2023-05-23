import logging
import copy
import numpy as np
import queue
from cammy.camera.base import CammyCamera
from typing import Optional


# TODO:
# 1) Get data from counters and append to timestamp file
class SpoofCamera(CammyCamera):
    def __init__(
        self,
        id: Optional[str],
        save_queue=None,
        width=640,
        height=480,
        **kwargs,
    ):
        super(CammyCamera, self).__init__()
  

        self.camera = id
        self.id = id
        self.logger = logging.getLogger(self.__class__.__name__)

        self.frame_count = 0
        self._last_framegrab = np.nan
        self.fps = None
        self.save_queue = save_queue # where data goes to recover
        self.recv_queue = queue.Queue() # transfer from primary camera
        self.missed_frames = 0
        self.total_frames = 0
        self.buffer = None
        self._width = width
        self._height = height  # stage stream
        

    # WE GET COPIES FROM OTHER CAMERAS, SIMPLY PUBLISH THEM
    def try_pop_frame(self):
        # don't grab twice!
        buffer = None
        try:
            buffer = self.recv_queue.get_nowait()
        except queue.Empty:
            pass
        # buffer = copy.deepcopy(self.buffer)
        # self.buffer = None
    
        if (self.save_queue is not None) and (buffer is not None):
            self.total_frames += 1
            self.frame_count += 1
            frame, timestamps = buffer
            self.save_queue.put((frame, timestamps))
            return frame, timestamps
        elif buffer is not None:
            self.total_frames += 1
            self.frame_count += 1
            frame, timestamps = buffer
            return frame, timestamps
        else:
            return None, None
    

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

     





