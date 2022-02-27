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
    Для удобства расчет необходимо чтобы полная осещенность (от окон и от ламп) была описана именно в этом класса
    Делить следует по постоянной (от окон) и динамической (от ламп). Для второй должена быть предусмотрена функция
    обновления как временного, так и постоянного с учетом батчей.
    """

    def __init__(self, walls: List[Wall], windows: List[Window], *, surface=None):
        self.walls = walls
        self.windows = windows


    def quante_area(self):
        """
        Нарезает область на атомарные единицы, для которых будут определяться параметры.
        Думаю пока это сделать через библиотеку полигонов. С одной стороны, это делается только один
        раз, поэтому времени много теряться не будет, с дрогой, можно будет строить этажи со стенами не
        под 90 градусов.
        :return:
        """
        pass

    def calculate_constant_illumination(self):
        """
        Расчитывает для каждой квантованной точки уровень освещенности от окон.
        Расчитывается только один раз.
        :return:
        """
        pass

    def append_object(self, object: OptimizedObject):
        pass

    def get_surface_lumen(self):
        pass

    def update_temp(self, object: SpaceObjects):
        """
        Пересчитывает параметры Area исходя их параметров групп обектов. Создает временные пераметры
        :param object:
        :return:
        """
        pass

    def update_perm(self, object: SpaceObjects):
        """
        Пересчитывает параметры Area исходя их параметров групп обектов. Создает временные пераметры
        :param object:
        :return:
        """
        pass



