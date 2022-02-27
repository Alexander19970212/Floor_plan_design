from turtle import window_width
import torch


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


def sum_sqr_shortest_path(acquired_obj_group):
    """
    Functions returns the sum of sqr path for objects of group_1 to the nearest object form group_2. Groups were defined
    in acquired_obj_group.
    :param acquired_obj_group: Object with type Acquired_group
    :return: torch vector of sum for batches
    """

    shortest_path = acquired_obj_group.get_parametrs("Path_lengths")  # torch tensor with size (batch_size, n_ob_group1)

    return torch.sqrt(shortest_path).sum(1)
