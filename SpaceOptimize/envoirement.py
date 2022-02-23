from dataclasses import dataclass, asdict
from typing import List

from turtle import window_width
import torch

from .interfaces import OptimizedObject

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


class Table(OptimizedObject):
    def get_coords(self) -> List[Point]:
        return super().get_coords()


class Area:
    """
    Получить освещенность окон. (Координаты, Ширина окон, Осввещенность, Нормаль) - для одной границы. 
    Для всей Area отдавать матрицу освещенности помещения, как surface
    """

    def __init__(self, walls: List[Wall], windows: List[Window], *, surface=None):
        self.walls = walls
        self.windows = windows

    def append_object(self, object: OptimizedObject):
        pass

    def get_surface_lumen(self):
        pass


