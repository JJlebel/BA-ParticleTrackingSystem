from enum import Enum


class EventType(Enum):
    MERGE = 1
    SPLIT = 2
    NONE = 3


class Event:
    def __init__(self):
        self.frame = None
        self.first_particle = None
        self.second_particle = None
        self.event_type = EventType.NONE
