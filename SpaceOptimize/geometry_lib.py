import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import torch


def check_intersect(walls, rays):
    """
    The function checks intersections between two sets of line sections.
    :param walls: numpy array with size (N_lines_in_set_1, 2 (start and end points), 2 (coord))
    :param rays: numpy array with size (N_lines_in_set_2, 2 (start and end points), 2 (coord))
    :return: numpy boolean array with size (N_lines_in_set_2, N_lines_in_set_1). True means intersection exist.
    """
    walls_start_x = np.tile(walls[:, 0, 0], (rays.shape[0], 1))
    walls_start_y = np.tile(walls[:, 0, 1], (rays.shape[0], 1))
    walls_end_x = np.tile(walls[:, 1, 0], (rays.shape[0], 1))
    walls_end_y = np.tile(walls[:, 1, 1], (rays.shape[0], 1))

    rays_start_x = np.tile(rays[:, 0, 0], (walls.shape[0], 1)).T
    rays_start_y = np.tile(rays[:, 0, 1], (walls.shape[0], 1)).T
    rays_end_x = np.tile(rays[:, 1, 0], (walls.shape[0], 1)).T
    rays_end_y = np.tile(rays[:, 1, 1], (walls.shape[0], 1)).T

    denom = (rays_end_y - rays_start_y) * (walls_end_x - walls_start_x) - (rays_end_x - rays_start_x) * (
                walls_end_y - walls_start_y)
    ua = ((rays_end_x - rays_start_x) * (walls_start_y - rays_start_y) - (rays_end_y - rays_start_y) * (
                walls_start_x - rays_start_x)) / denom
    ub = ((walls_end_x - walls_start_x) * (walls_start_y - rays_start_y) - (walls_end_y - walls_start_y) * (
                walls_start_x - rays_start_x)) / denom

    result_inter = np.logical_not(
        np.logical_or(np.logical_or(np.logical_or(ua < 0, ua > 1), np.logical_or(ub < 0, ub > 1)), denom == 0))

    return result_inter


def magnitude(vector):
    """
    This function is additional.
    """
    return np.sqrt(np.diag(np.tensordot(np.array(vector), np.array(vector), axes=[1, 1])))


def norm(vector):
    """
    This function is additional.
    """
    return np.array(vector) / magnitude(np.array(vector))[:, np.newaxis]


def lineRayIntersectionPoint2(rayOrigin, rayDirection, point1, point2):

    """
    The function check intersection between set of lines sections and set of rays from 1 point (in 2d).
    Ray-Line Segment Intersection Test in 2D
    http://bit.ly/1CoxdrG
    :param rayOrigin: Origin, numpy array with size 2
    :param rayDirection: set with normals, numpy array with size (N_rays, 2)
    :param point1: set with start points of line sections, numpy array with size (N_line_sections, 2)
    :param point2: set with end points of line sections, numpy array with size (N_line_sections, 2)
    :return:
            intersection points: numpy array with size (N_rays, N_line_sections, 2)
            intersect_matrix: boolean numpy array with size (N_rays, N_line_sections)
    """
    # Convert to numpy arrays
    rayOrigin = np.array(rayOrigin, dtype=np.float)
    rayDirection = np.array(norm(rayDirection), dtype=np.float)
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)

    v1 = rayOrigin - point1
    v2 = point2 - point1

    v3 = np.zeros_like(rayDirection)
    v3[:, 0] = rayDirection[:, 1] * (-1)
    v3[:, 1] = rayDirection[:, 0]

    size_v1 = v1.shape[0]
    size_v2 = v2.shape[0]
    size_v3 = v3.shape[0]

    v2_v1_cross = np.cross(v2, v1)

    v1 = np.repeat(v1, size_v3, axis=0)
    v2 = np.repeat(v2, size_v3, axis=0)
    v3 = np.tile(v3, (size_v1, 1))

    v2_v3_dot = v2 * v3
    v1_v3_dot = v1 * v3

    v1_v3_dot = v1_v3_dot.sum(1)
    v2_v3_dot = v2_v3_dot.sum(1)
    v1_v3_dot = v1_v3_dot.reshape((size_v1, size_v3))
    v2_v3_dot = v2_v3_dot.reshape((size_v2, size_v3))

    t1 = v2_v1_cross / (v2_v3_dot.T)
    t2 = (v1_v3_dot / v2_v3_dot).T

    tile1 = np.tile(rayOrigin, (size_v1, 1))
    tile_2 = np.tile(rayDirection[:, np.newaxis], (1, size_v1, 1))
    t1_ = t1[:, :, np.newaxis]

    intersection_points = tile1 + t1_ * tile_2

    intersect_matrix = (t1 >= 0.0) * (t2 >= 0.0) * (t2 <= 1.0)

    return intersection_points, intersect_matrix


def check_intersect_batch(walls, rays):
    """
    The check intersections between two sets of lines. This was made for pytorch.
    Args walls and rays must be torch tensors and have the same size.
    : param walls: torch tensor with size (N_batch, N_qiunt_points, N_light sources, N_walls, 2, 2)
    : param rays: torch tensor with size (N_batch, N_quant_point, N_light_sources, N_walls, 2, 2)
    : return: boolean torch tensor with size (N_batch, N_quant_points, N_light_cources, N_walls)
    """
    walls_start_x = walls[:, :, :, :, 0, 0]  # x1
    walls_start_y = walls[:, :, :, :, 0, 1]  # y1
    walls_end_x = walls[:, :, :, :, 1, 0]  # x2
    walls_end_y = walls[:, :, :, :, 1, 1]  # y2

    rays_start_x = rays[:, :, :, :, 0, 0]  # x3
    rays_start_y = rays[:, :, :, :, 0, 1]  # y3
    rays_end_x = rays[:, :, :, :, 1, 0]  # x4
    rays_end_y = rays[:, :, :, :, 1, 1]  # y4

    denom = (rays_end_y - rays_start_y) * (walls_end_x - walls_start_x) - (rays_end_x - rays_start_x) * (
                walls_end_y - walls_start_y)
    # ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ua = ((rays_end_x - rays_start_x) * (walls_start_y - rays_start_y) - (rays_end_y - rays_start_y) * (
                walls_start_x - rays_start_x)) / denom
    # ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    ub = ((walls_end_x - walls_start_x) * (walls_start_y - rays_start_y) - (walls_end_y - walls_start_y) * (
                walls_start_x - rays_start_x)) / denom

    # result_inter = np.logical_not(np.logical_or(np.logical_or(np.logical_or(ua<0, ua>1), np.logical_or(ub<0, ub>1)), denom==0))
    result_inter = torch.logical_not(
        torch.logical_or(torch.logical_or(torch.logical_or(ua <= 0, ua >= 1), torch.logical_or(ub <= 0, ub >= 1)),
                         denom == 0))

    return result_inter