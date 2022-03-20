from dataclasses import dataclass, asdict
from typing import List

from turtle import window_width
from geometry_lib import check_intersect, lineRayIntersectionPoint2, norm, check_intersect_batch
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

    def __init__(self, walls: List[Wall], windows: List[Window], septum: List[Wall], windows_koeff, *, surface=None):
        self.walls = walls
        self.windows = windows
        self.septum = septum
        self.windows_weights = windows_koeff

        self.quantization_step = 0.002
        self.grid_coord_step = 0.01
        self.border = self.find_border()
        self.quant_coord = self.quantization(self.quantization_step)
        self.quant_coord_torch = torch.from_numpy(self.quant_coord)
        self.constant_illumination = self.calculate_constant_illumination()
        self.constant_illumination_torch = torch.from_numpy(np.array(self.constant_illumination))
        self.artificial_natural_illum_koeff = 0.25
        self.obj_grid = self.calculate_vailable_coord()

    def find_border(self):
        """
        Функция находит замкнутый полигон контура путем перебора стен.
        Следует учесть, что реализация не подходит, если контур состоит из нескольких полгигов
        :return: координаты полигона (координаты точек, (не повторяются, как бы это было при описании набором отрезков))
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

    def quantization(self, quant_step):
        # border numpy array with coords of points of border. Size is (N, 2)
        min_x_y = self.border.min(axis=0)
        max_x_y = self.border.max(axis=0)

        step = min((max_x_y[0] - min_x_y[0]) * quant_step,
                   (max_x_y[1] - min_x_y[1]) * quant_step)

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

        countur_polygon = Polygon(self.border)

        coords = coords.reshape((coords.shape[0] * coords.shape[1], coords.shape[2]))
        contained_points = np.zeros(coords.shape[0])

        for i, point in enumerate(coords):
            contained_points[i] = countur_polygon.contains(Point(point))

        # print(contained_points)

        return coords[contained_points == 1]


    def calculate_constant_illumination(self):
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
        :return: float list with value of illumination for each quanted point.
        """
        list_illumination = []

        angles = np.arange(0, 360, 1.0)
        norms = np.array([np.cos(angles * np.pi / 180), np.sin(angles * np.pi / 180)]).T
        d = norm(norms)

        all_sep = np.append(self.walls, self.septum, axis=0)

        for r in self.quant_coord:
            points, intersect_mat = lineRayIntersectionPoint2(r, d, self.windows[:, 0, :], self.windows[:, 1, :])
            secondary_intersection = np.zeros_like(intersect_mat)
            foundet_points = points[intersect_mat]

            rays = np.zeros((foundet_points.shape[0], 2, 2))
            rays[:, 0] = foundet_points
            rays[:, 1] = r

            result_intersections = check_intersect(walls=all_sep, rays=rays) * 1
            print(result_intersections.shape)
            result_intersections = result_intersections.sum(1)
            result_intersections = (result_intersections <= 1) * 1

            secondary_intersection[intersect_mat] = result_intersections
            # real_points = points[secondary_intersection]

            point_illumination = self.windows_weights * secondary_intersection.sum(0)
            list_illumination.append(point_illumination.sum())

        return list_illumination

    def calculate_vailable_coord(self):
        """
        The function create grids on floor polygon. Every edge of grid could be used for locating objects.
        : return: numpy array with size (N_edges, 2(x and y).
        """
        return self.quantization(self.grid_coord_step)


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

    def get_illum_distribution(self):
        """
        Функция возворащает распределение освещенности в каком-то виде (решить позже). Необходама для
        расчета целевых функций.
        к этой фонкции обращается целевая функция через посредника с целью обеспечения универсализации.
        Повредниками могут быть Space_obj, group_obj, builder. Собственно тот, кто будет выполнять расчет ФФ.

        Фонкция выполняет запрос с целью получения освещенности от объектов.
        Запрос может проводиться у group_obj, space_obj. Возможно данный объект стоит подавать на вход (надо подумать)
        запрос производится через стандартную функцию get_param.
         Хорошо бы было реализовать эту функцию у всех типов объектов.
         : return: torch tensor with size (N_batches, N_quant_points)
        """
        light_koeff, light_point = virt_obj.get_param('Illumination', crd=True)

        # coord is numpy array and has size (N_batches, N_light_points, 2)
        # N_points in batches have to have the same lengths. If it are different, it should be added random new
        # and have 0 for them in light_koeff.
        # light_koeff is numpy array with size (N_light_points)

        light_points = torch.from_numpy(light_points)

        tracks = torch.zeros(light_points.size(0), self.quant_coord_torch.size(0), light_points.size(1), light_points.size(2),
                             self.quant_coord_torch.size(1))
        tracks[:, :, :, 0, :] = light_points.unsqueeze(1)
        tracks[:, :, :, 1, :] = self.quant_coord_torch.unsqueeze(1)

        all_sep = np.append(self.walls, self.septum, axis=0)
        barrier = torch.from_numpy(all_sep)
        track_for_walls = torch.zeros(tracks.size(0), tracks.size(1), tracks.size(2), barrier.size(0), 2, 2)
        walls_for_tracks = torch.zeros(tracks.size(0), tracks.size(1), tracks.size(2), barrier.size(0), 2, 2)
        track_for_walls[:, :, :, :, :, :] = tracks.unsqueeze(3).repeat(1, 1, 1, barrier.size(0), 1, 1)
        walls_for_tracks = barrier.repeat(tracks.size(0), tracks.size(1), tracks.size(2), 1, 1, 1)

        result_inter = check_intersect_batch(track_for_walls, walls_for_tracks)
        lighted_q_pints = result_inter.sum(3) <= 1
        distances_sqr = torch.sqrt(torch.square(tracks[:, :, :, 0, 0] - tracks[:, :, :, 1, 0]) + torch.square(
            tracks[:, :, :, 0, 1] - tracks[:, :, :, 1, 1]))
        power_for_points = lighted_q_pints / distances_sqr * light_koeff
        power_for_points = power_for_points.sum(2)

        points_power = self.constant_illumination + power_for_points[0] * self.artificial_natural_illum_koeff

        return points_power

    def get_param(self, param):
        """
        The function return parameters of Area as class.
        : param: str arg - name of parameter. Available options are : "constant_illumination", "coord_grid", "barrier",
                                                                "windows", "septum", (doors), "current_illumination",
        """
        if param == "constant_illumination":
            """
            type = numpy array;
            size = (N_quants, 2 (x, y))
            """
            return self.constant_illumination

        elif param == "coord_grid":
            """
            type = numpy array;
            size = (N_edges, 2(x, y))
            """
            return np.array(self.obj_grid)

        elif param == "barrier":
            """
            type = numpy array;
            size = (N_vertex, 2(x, y)). Vertex are not repeated including start and end points.
            """
            return self.border

        elif param == "windows":
            """
            type = numpy array;
            size = (N_windows, 2(start, end), 2(x, y))
            """
            return np.array(self.windows)

        elif param == "septum":
            """
            type = numpy array;
            size = (N_septum, 2(start, end), 2(x, y))
            """
            return np.array(self.septum)

        elif param == "current_illumination":
            """
            type = torch tensor
            size = (N_batch, N_quant)
            """
            return self.get_illum_distribution()






