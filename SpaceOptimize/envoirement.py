from contextlib import AbstractAsyncContextManager
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


class OptimizedObject(Protocol):  # Проработать API объекта
    @abstractmethod
    def get_coords(self, ):
        pass


class GroupObject(): # расчет лосса для группы, так и между группами
    """
        Все интересующие параметры будут извлекаться из групп
        Произвести операции с коордианатами объектов, принадлежащих группе
        Габариты объектов - матрица габаритов
        Освещенность - вектор для группы, для объекта число
    """
    def __init__(self, objects: List[OptimizedObject]):
        pass




class Table(OptimizedObject):
    def get_coords(self) -> List[Point]:
        return super().get_coords()

t = Table()

c = t.get_coords()
p = c[0]

p.y




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

    

