from dataclasses import dataclass, asdict
from abc import abstractmethod
from traceback import walk_stack
from turtle import window_width
from typing import Protocol, List

@dataclass
class Point:
    x: int
    y: int


@dataclass
class Border:
    start: Point
    end: Point

@dataclass
class Wall(Border):
    pass


@dataclass
class Window(Border):
    illumination: float


class OptimizedObject:
    def __init__(self, ):
        pass



class Area:
    def __init__(self, walls: List[Wall], windows: List[Window], *, surface=None):
        self.walls = walls
        self.windows = windows

    def append_object(self, object):
        pass

    