from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, asdict
from abc import abstractmethod
from traceback import walk_stack
from turtle import window_width
from typing import Protocol, List
import torch


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


class GroupObject():  # расчет лосса для группы, так и между группами
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


def density(group):
    """
    Function returns sum of distances between points and centre of points' cloud.
    :param group: object with Group type.
    :return: batch list of float with sum distances,
             batch list with list of distances for each objects,
             float (sum of distance) for statistic.
    """

    coordinates = group.get_parameters("coordinates")  # torch tensor with size (batch_size, N_objects, 2) !!!! COPY
    centre = torch.unsqueeze(torch.mean(coordinates * 1.0, 1), 1)
    distant = torch.sqrt(torch.sum(torch.square(coordinates - centre), 2))

    return distant.sum(1), distant, distant.sum()


def illumination(group):
    """
    Function compares illumination parameters of objects with required values.
    :param group: object with Group type.
    :return: batch list with sums of objects' illumination
             batch list with suns of objects' excess illumination
             batch list with shortage illumination for each objects
             batch list with excess illumination for each objects
             shortage and excess sums for statistics
    """
    illuminations = group.get_parameters("illumination")  # torch tensor with size (batch_size, N_objects) !!!COPY
    lighting_requirements = group.get_parameters("illumination_requirements")  # torch tensor with size as higher

    differences_ill_req = lighting_requirements - illuminations
    shortage_illuminations = differences_ill_req * (differences_ill_req > 0)
    excess_illumination = differences_ill_req * (differences_ill_req <= 0) * (-1)

    return shortage_illuminations.sum(1), excess_illumination.sum(
        1), shortage_illuminations, excess_illumination, shortage_illuminations.sum() + excess_illumination.sum()
