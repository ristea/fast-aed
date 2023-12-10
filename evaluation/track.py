from enum import Enum


class TrackState(Enum):
    CREATED = "created"
    UPDATED = "updated"
    CLOSED = "closed"


class Track:

    def __init__(self, start_idx=0, end_idx=None, mask=0, video_name=""):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.bboxes = {}
        self.mask = mask
        self.state = TrackState.CREATED
        self.video_name = video_name

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)