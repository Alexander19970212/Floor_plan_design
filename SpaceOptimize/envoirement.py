from dataclasses import dataclass, asdict
from typing import List

from turtle import window_width
import torch
from shapely.geometry import Point, LineString, Polygon
import numpy as np

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

        self.quantization_step = 0.01

    def find_border(self):
        """
        Функция находит замкнутый полигон контура путем перебора стен.
        Следует учесть, что реализация не подходит, если контур состоит из нескольких полгигов
        :return:
        """

        border = []

        start_point = [self.walls[0].start.x, self.walls[0].start.y]
        end_point = [self.walls[0].end.x, self.walls[0].end.y]
        border.append(start_point)

        while end_point != start_point:
            border.append(end_point)
            find_point_flag = False
            for wall in self.walls:
                if [wall.start.x, wall.start.y] == end_point:
                    end_point = [wall.end.x, wall.end.y]
                    find_point_flag = True
            assert find_point_flag, "Border is not closed"

        return np.array(border)

    def quantization(self):
        border = self.find_border()
        min_x_y = border.min(axis=0)
        max_x_y = border.max(axis=0)

        step = min((max_x_y[0] - min_x_y[0]) * self.quantization_step,
                   (max_x_y[1] - min_x_y[1]) * self.quantization_step)

        x_coords = np.arange(min_x_y[0], max_x_y[0], step)
        y_coords = np.arange(min_x_y[1], max_x_y[1], step)

        # combine coords
        x_size = x_coords.shape[0]
        y_size = y_coords.shape[0]
        y_coords = y_coords[np.newaxis].T
        y_coords = np.tile(y_coords, x_size)[:, :, np.newaxis]
        x_coords = x_coords[np.newaxis].T
        x_coords = np.tile(x_coords, y_size).T[:, :, np.newaxis]
        coords = np.concatenate((x_coords, y_coords), axis=2)

        countur_polygon = Polygon(border)

        coords = coords.reshape((coords.shape[0] * coords.shape[1], coords.shape[2]))
        contained_points = np.zeros(coords.shape[0])

        for point, coord_cont in zip(coords, contained_points):
            coord_cont = countur_polygon.contains(point)

        return coords[coord_cont]

    def illuminate_calculation(self):
        """
        функции для каждой квантованной точки плоцщади расчитывает ее освещенность.
        Расчет проводится по следующему алгоритму:
        В цикле для каждой квантованной точки:
          - строятся лучи от точки во всех направлениях с заданным шагом.
          - находятся те лучи, которые достигают окон
          - из лучей, найденных вышу вычисляются отрезеки
          - проверяется, пересекаются ли отрезке с стенами
          - их тех что не пересейкаются составляются списки + к ним с в соответсвие мощность излучения от
          конкретного окна
          - суммируются произвдение количество точек окна на мощность окна
        :return:
        """
        pass

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


def magnitude(vector):
    return np.sqrt(np.dot(np.array(vector), np.array(vector)))


def norm(vector):
    return np.array(vector) / magnitude(np.array(vector))


def lineRayIntersectionPoint2(rayOrigin, rayDirection, point1, point2):
    # Convert to numpy arrays
    rayOrigin = np.array(rayOrigin, dtype=np.float)
    rayDirection = np.array(norm(rayDirection), dtype=np.float)
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)

    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.array([-rayDirection[1], rayDirection[0]])

    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)

    inters_points = t1

    intersection_points = np.tile(rayOrigin, (v1.shape[0], 1)) + t1[:, np.newaxis] * np.tile(rayDirection,
                                                                                             (v1.shape[0], 1))

    intersect_matrix = (t1 >= 0.0) * (t2 >= 0.0) * (t2 <= 1.0)

    return intersection_points, intersect_matrix



