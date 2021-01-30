import numpy as np
import random


class Optimizer:
    def __init__(self, classes):
        self.Classes = classes
        self.bounds = []
        self.probability_mask = []
        self.bounds_constructor()
        self.probability_mask_constructor()
        # self.bounds = np.array(self.bounds)
        # self.probability_mask = np.array(self.probability_mask)
        print(self.probability_mask)
        self.gen_constructor()

    def bounds_constructor(self):
        for kind in self.Classes:
            layer_parameters = [[self.Classes[kind]['Environment_x'], self.Classes[kind]['Environment_x'] * 2],
                                [self.Classes[kind]['Environment_y'], self.Classes[kind]['Environment_y'] * 2],
                                [0, self.Classes[kind]['Environment_x']], [0, self.Classes[kind]['Environment_y']],
                                [1, 10], [1, 10],
                                [self.Classes[kind]['Environment_x'] * 2, self.Classes[kind]['Environment_x'] * 10],
                                [self.Classes[kind]['Environment_y'] * 2, self.Classes[kind]['Environment_y'] * 10],
                                [0, 45, 90], [0, 1]]
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0, 45, 90])
            self.bounds.append(layer_parameters)

    def probability_mask_constructor(self):
        for kind in self.Classes:
            layer_parameters = [0, 0, 0, 0, 0.5, 0.5,
                                'Equally_distributed', 'Equally_distributed',
                                'Equally_distributed', 'Equally_distributed',
                                [0.4, 0.2, 0.4], 0]
            for drop in range(0, self.Classes[kind]["Amount"]):
                layer_parameters.append([0.4, 0.2, 0.4])
            self.probability_mask.append(layer_parameters)

    def gen_constructor(self):
        big_gen = []
        for net_mask, net_doubt in zip(self.probability_mask, self.bounds):
            net_gen = []
            for probability, value_slot in zip(net_mask, net_doubt):

                if type(probability) is int:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                                     value_slot[0] + (value_slot[1] - value_slot[0]) * probability))
                elif type(probability) is float:
                    net_gen.append(random.triangular(value_slot[0], value_slot[1],
                                            value_slot[0] + (value_slot[1] - value_slot[0]) * probability))
                elif type(probability) is str:
                    net_gen.append(random.uniform(value_slot[0], value_slot[1]))
                else:
                    #print(value_slot, probability)
                    net_gen.append(random.choices(value_slot, cum_weights=probability, k=1)[0])

                # elif type(probability) == 'str':
                # else:
            big_gen.append(net_gen)
        print(big_gen)


if __name__ == "__main__":
    Classes = {
        'Workplace': {"Amount": 12, "rectangular_x": 2, "rectangular_y": 1, 'Environment_x': 4, "Environment_y": 3,
                      "Need_lighting": 9,
                      "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                      "Classes_for_distant": {"Machine_tool": 8}},
        'Printers': {"Amount": 3, "rectangular_x": 1, "rectangular_y": 1, 'Environment_x': 3, "Environment_y": 3,
                     "Need_lighting": 9,
                     "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
                     "Classes_for_distant": {"Machine_tool": 9}},
        'Cabinets': {"Amount": 4, "rectangular_x": 0.5, "rectangular_y": 2, 'Environment_x': 1.5, "Environment_y": 1,
                     "Need_lighting": 6,
                     "Classes_for_short_path": ["Workplace"], "Classes_ignored_intersections": ["lamp"],
                     "Classes_for_distant": {"Machine_tool": 5}},
        'lamp': {"Amount": 60, "rectangular_x": 0.4, "rectangular_y": 0.4, 'Environment_x': 0.4, "Environment_y": 0.4,
                 "Need_lighting": "reverse",
                 "Classes_for_short_path": [None],
                 "Classes_ignored_intersections": ["Workplace", "Printers", "Cabinets", "Machine_tool"],
                 "Classes_for_distant": {None}},
        'Machine_tool': {"Amount": 4, "rectangular_x": 3, "rectangular_y": 4, 'Environment_x': 8, "Environment_y": 8,
                         "Need_lighting": 8,
                         "Classes_for_short_path": ["Printers", "Cabinets"], "Classes_ignored_intersections": ["lamp"],
                         "Classes_for_distant": {"Workplace": 8, "Printers": 9, "Cabinets": 5}}
    }

    Opt = Optimizer(Classes)
