import numpy as np
import matplotlib.pyplot as plt


class Particle_wsarm:
    # def __init__(self, Points_obj, Points_block, obj_distance, koef_obj, block_distance, koef_block, indexs_cl, step):
    #     self.points_block = Points_block
    #     self.points_obj = Points_obj
    #     self.obj_distance = obj_distance
    #     self.koef_obj = koef_obj
    #     self.block_distance = block_distance
    #     self.koef_block = koef_block
    #     self.step_value = step
    #     self.step_save = step
    #     self.indexs_cl = indexs_cl
    #
    #     self.points_obj_x = self.points_obj[:, 0].flatten()
    #     self.points_obj_y = self.points_obj[:, 1].flatten()
    #
    #     self.points_block_x = self.points_block[:, 0].flatten()
    #     self.points_block_y = self.points_block[:, 1].flatten()

    def __init__(self, cl_att_cl, Num_cl, Points_block, block_cl, bl_att_cl):
        self.cl_att_cl = cl_att_cl
        self.Num_cl = Num_cl
        self.Points_block = Points_block
        self.block_cl = block_cl
        self.bl_att_cl = bl_att_cl

    def intialize(self, Points_obj, Points_block, obj_distance, koef_obj, block_distance, koef_block, indexs_cl, step):

        self.points_block = Points_block
        self.points_obj = Points_obj
        self.obj_distance = obj_distance
        self.koef_obj = koef_obj
        self.block_distance = block_distance
        self.koef_block = koef_block
        self.step_value = step
        self.step_save = step
        self.indexs_cl = indexs_cl

        self.points_obj_x = self.points_obj[:, 0].flatten()
        self.points_obj_y = self.points_obj[:, 1].flatten()

        self.points_block_x = self.points_block[:, 0].flatten()
        self.points_block_y = self.points_block[:, 1].flatten()

    def searching_distance_loc(self):
        self.loc_t_block_x = np.array([self.points_obj_x, ] * self.points_block_x.shape[0])
        self.loc_t_block_y = np.array([self.points_obj_y, ] * self.points_block_y.shape[0])
        self.block_t_loc_x = np.array([self.points_block_x, ] * self.points_obj_x.shape[0])
        self.block_t_loc_y = np.array([self.points_block_y, ] * self.points_obj_y.shape[0])

        points_obj_x = np.array([self.points_obj_x, ] * self.points_obj_x.shape[0])
        points_obj_y = np.array([self.points_obj_y, ] * self.points_obj_y.shape[0])

        self.dist_loc_x = points_obj_x - points_obj_x.transpose()
        self.dist_loc_y = points_obj_y - points_obj_y.transpose()

        self.distance_loc = np.sqrt(np.square(self.dist_loc_x) + np.square(self.dist_loc_y))

        # print(self.distance_loc)

    def searching_distance_block(self):
        self.dist_block_x = self.loc_t_block_x.transpose() - self.block_t_loc_x
        self.dist_block_y = self.loc_t_block_y.transpose() - self.block_t_loc_y

        self.distance_block = np.sqrt(np.square(self.dist_block_x) + np.square(self.dist_block_y))

        # print(self.distance_block.shape)

    def searching_forces_loc(self):
        loc_cas_1 = self.distance_loc * ((self.distance_loc < self.obj_distance[:, :, 0]) * 1)
        loc_cas_2 = self.distance_loc * ((self.distance_loc > self.obj_distance[:, :, 0]) * 1)
        loc_cas_2 = self.distance_loc * ((loc_cas_2 < self.obj_distance[:, :, 1]) * 1)

        loc_cas_3 = self.distance_loc * ((self.distance_loc > self.obj_distance[:, :, 1]) * 1)

        force_loc_1 = loc_cas_1 * self.koef_obj[:, :, 0] - self.koef_obj[:, :, 0] * (
                (loc_cas_1 > 0) * 1) * self.obj_distance[:, :, 0]
        force_loc_2 = loc_cas_2 * self.koef_obj[:, :, 1] - self.koef_obj[:, :, 1] * (
                (loc_cas_2 > 0) * 1) * self.obj_distance[:, :, 0]
        force_loc_3 = loc_cas_3 * self.koef_obj[:, :, 2] - self.koef_obj[:, :, 2] * (
                (loc_cas_3 > 0) * 1) * self.obj_distance[:, :, 1]

        force_loc = force_loc_1 + force_loc_2 + force_loc_3
        self.force_loc_x = force_loc * ((self.dist_loc_x > 0) * 2 - 1)
        self.force_loc_y = force_loc * ((self.dist_loc_y > 0) * 2 - 1)

        self.force_loc_x = self.force_loc_x.sum(axis=1).transpose()
        self.force_loc_y = self.force_loc_y.sum(axis=1).transpose()

        # print(self.force_loc_y)

    def searching_forces_block(self):
        block_cas_1 = self.distance_block * ((self.distance_block < self.block_distance[:, :, 0]) * 1)
        block_cas_2 = self.distance_block * ((self.distance_block > self.block_distance[:, :, 0]) * 1)
        block_cas_2 = self.distance_block * ((block_cas_2 < self.block_distance[:, :, 1]) * 1)
        block_cas_3 = self.distance_block * ((self.distance_block > self.block_distance[:, :, 1]) * 1)

        # print(np.argwhere(block_cas_1))
        # print(np.argwhere(block_cas_2))
        # print(np.argwhere(block_cas_3))

        force_block_1 = block_cas_1 * self.koef_block[:, :, 0] - self.koef_block[:, :, 0] * (
                (block_cas_1 > 0) * 1) * self.block_distance[:, :, 0]
        force_block_2 = block_cas_2 * self.koef_block[:, :, 1] - self.koef_block[:, :, 1] * (
                (block_cas_2 > 0) * 1) * self.block_distance[:, :, 0]
        force_block_3 = block_cas_3 * self.koef_block[:, :, 2] - self.koef_block[:, :, 2] * (
                (block_cas_3 > 0) * 1) * self.block_distance[:, :, 1]

        force_block = force_block_1 + force_block_2 + force_block_3
        self.force_block = force_block
        # print(np.sum(force_block))
        # print(force_block)
        self.force_block_x = force_block * ((self.dist_block_x < 0) * 2 - 1)
        self.force_block_y = force_block * ((self.dist_block_y < 0) * 2 - 1)
        # print(self.force_block_x.shape)

        self.force_block_x = self.force_block_x.sum(axis=1).transpose()
        self.force_block_y = self.force_block_y.sum(axis=1).transpose()
        # print(self.force_block_y)

    def searching_forces_block_mindist(self):
        block_cas_1 = self.distance_block * ((self.distance_block < self.block_distance[:, :, 0]) * 1)
        block_cas_2 = self.distance_block * ((self.distance_block > self.block_distance[:, :, 0]) * 1)
        block_cas_2 = self.distance_block * ((block_cas_2 < self.block_distance[:, :, 1]) * 1)
        block_cas_3 = self.distance_block * ((self.distance_block > self.block_distance[:, :, 1]) * 1)

        # print(np.argwhere(block_cas_1))
        # print(np.argwhere(block_cas_2))
        # print(np.argwhere(block_cas_3))

        force_block_1 = block_cas_1 * self.koef_block[:, :, 0] - self.koef_block[:, :, 0] * (
                (block_cas_1 > 0) * 1) * self.block_distance[:, :, 0]
        force_block_2 = block_cas_2 * self.koef_block[:, :, 1] - self.koef_block[:, :, 1] * (
                (block_cas_2 > 0) * 1) * self.block_distance[:, :, 0]
        force_block_3 = block_cas_3 * self.koef_block[:, :, 2] - self.koef_block[:, :, 2] * (
                (block_cas_3 > 0) * 1) * self.block_distance[:, :, 1]

        force_grav_max_dist = force_block_3.max() + 1

        grav_mask = (force_block_3 == 0) * force_grav_max_dist
        force_block_3 += grav_mask
        # print(force_block_3.shape)

        ind_min_dist_grav = np.unravel_index(np.argmin(force_block_3, axis=1), force_block_3.shape)
        # print(ind_min_dist_grav)
        # force_grav_min_dist = force_block_3.min()
        s_mask = np.zeros_like(force_block_3)
        for i, loc in enumerate(ind_min_dist_grav[1]):
            s_mask[i, loc] = force_block_3[i, loc]

        force_block = force_block_1 + force_block_2 + s_mask
        # force_block[ind_min_dist_grav[0], ind_min_dist_grav[1]] += force_grav_min_dist
        self.force_block = force_block
        # print(np.sum(force_block))
        # print(force_block)
        self.force_block_x = force_block * ((self.dist_block_x < 0) * 2 - 1)
        self.force_block_y = force_block * ((self.dist_block_y < 0) * 2 - 1)
        # print(self.force_block_x.shape)

        self.force_block_x = self.force_block_x.sum(axis=1).transpose()
        self.force_block_y = self.force_block_y.sum(axis=1).transpose()
        # print(self.force_block_y)

    def step(self):
        # print(self.force_block_x)
        # print(self.force_block_y)
        self.points_obj_x = self.points_obj_x + self.step_value * (self.force_block_x + self.force_loc_x)
        self.points_obj_y = self.points_obj_y + self.step_value * (self.force_block_y + self.force_loc_y)

        self.max_force.append(
            max(max(self.force_block_x + self.force_loc_x), max(self.force_block_y + self.force_loc_y)))
        self.x_plot_force.append(self.count_for_img)
        self.max_force_block.append(max(max(self.force_block_x), max(self.force_block_y)))
        self.max_force_lock.append(max(max(self.force_loc_x), max(self.force_loc_y)))
        # print(self.points_obj_x)

    def preparing_data(self, cl_att_cl, Num_cl, bloks, blok_cl, bl_att_cl, Points_obj):
        import random
        N = np.sum(Num_cl)
        bl_att_cl[:, :, 0] = bl_att_cl[:, :, 0] * N * 2
        # bl_att_cl[0, 1, 2] = bl_att_cl[0, 1, 2] * N

        Points_obj = np.array(Points_obj)
        addit_count_ = 0
        indexs_cl = []
        for cl in Num_cl:
            indexs_cl.append([addit_count_, addit_count_ + cl - 1])
            addit_count_ += cl

        print(indexs_cl)

        obj_mask = np.ones((N, N), int)
        obj_dist = np.repeat(np.repeat(cl_att_cl, Num_cl, axis=0), Num_cl, axis=1)
        np.fill_diagonal(obj_mask, 0)
        obj_mask = obj_mask[:, :, np.newaxis]
        koef_obj = obj_dist[:, :, :3] * obj_mask
        obj_dist = obj_dist[:, :, 3:5] * obj_mask

        block_distance = np.array([[[0, 0, 0, 0, 0], ] * Points_block.shape[0], ] * Points_obj.shape[0])

        for i, bl_cl in enumerate(block_cl):
            for case in enumerate(bl_cl):
                # print(i)
                for j, val in enumerate(bl_att_cl[i]):
                    # print(int(indexs_cl[j][0]), int(indexs_cl[j][1]), int(case[1][0]), int(case[1][1]), val)
                    block_distance[int(indexs_cl[j][0]):int(indexs_cl[j][1]), int(case[1][0]): int(case[1][1]), :] = val

                # block_distance[]
                # print(case[0], case[1])

        # print(Points_obj)
        return Points_obj, bloks, obj_dist, koef_obj, np.array(block_distance[:, :, 3:5]), np.array(
            block_distance[:, :, :3]), indexs_cl

    def process(self):
        for t in range(0, 200):
            self.step_value = self.step_save * t / 200
            self.searching_distance_loc()
            self.searching_distance_block()
            self.searching_forces_loc()
            self.searching_forces_block_mindist()
            self.step()
            # print(self.points_obj_x.shape)
            # print(self.points_obj_y.shape)
            print(t)

            fig, ax = plt.subplots(2)
            ax[0].scatter(self.points_block_x, self.points_block_y)
            for cl_raw in self.indexs_cl:
                ax[0].scatter(self.points_obj_x[cl_raw[0]:cl_raw[1]], self.points_obj_y[cl_raw[0]:cl_raw[1]])
            ax[1].plot(self.x_plot_force, self.max_force)
            fig.savefig(f"band{t}.jpg", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        # plt.show()

    def micro_process(self):
        for t in range(0, 200):
            self.step_value = self.step_save * t / 200
            self.searching_distance_loc()
            self.searching_distance_block()
            self.searching_forces_loc()
            self.searching_forces_block_mindist()
            self.step()
            # print(self.points_obj_x.shape)
            # print(self.points_obj_y.shape)
            print(t)

            fig, ax = plt.subplots(2)
            ax[0].scatter(self.points_block_x, self.points_block_y)
            for cl_raw in self.indexs_cl:
                ax[0].scatter(self.points_obj_x[cl_raw[0]:cl_raw[1]], self.points_obj_y[cl_raw[0]:cl_raw[1]])
            ax[1].set_xlim(0, 620)
            ax[1].plot(self.x_plot_force, self.max_force_lock)
            ax[1].plot(self.x_plot_force, self.max_force_block)

            fig.savefig(f"band{self.count_for_img}.jpg", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            self.count_for_img += 1

    def process2(self):
        import random
        print('Rice')
        self.max_force = []
        self.max_force_block = []
        self.max_force_lock = []
        self.avg_force = []
        self.x_plot_force = []
        self.count_for_img = 0
        step = 0.001
        addit_count_ = 0
        self.Points_obj = []
        for i, cl in enumerate(self.Num_cl):
            step_Num_cl = self.Num_cl[0:i + 1]
            step_cl_att_cl = self.cl_att_cl[0:i + 1, 0:i + 1, :]
            step_bl_att_cl = self.bl_att_cl[:, 0:i + 1, :]
            new_Points_obj = []

            for i in range(addit_count_, addit_count_ + cl):
                self.Points_obj.append([random.randrange(22, 32), random.randrange(17, 32)])

            Points_obj, Points_block, obj_dist, koef_obj, block_distance, koef_block, indexs_cl = self.preparing_data(
                step_cl_att_cl, step_Num_cl, self.Points_block, self.block_cl, step_bl_att_cl, self.Points_obj)

            self.intialize(Points_obj, Points_block, obj_dist, koef_obj, block_distance, koef_block, indexs_cl,
                           step)
            self.micro_process()


def get_data(cl_att_cl, Num_cl, bloks, blok_cl, bl_att_cl):
    import random
    N = np.sum(Num_cl)
    bl_att_cl[:, :, 0] = bl_att_cl[:, :, 0] * N ** 2
    # bl_att_cl[0, 1, 2] = bl_att_cl[0, 1, 2] * N
    Points_obj = []
    for i in range(N):
        Points_obj.append([random.randrange(22, 32), random.randrange(17, 32)])

    Points_obj = np.array(Points_obj)
    addit_count_ = 0
    indexs_cl = []
    for cl in Num_cl:
        indexs_cl.append([addit_count_, addit_count_ + cl - 1])
        addit_count_ += cl

    print(indexs_cl)

    obj_mask = np.ones((N, N), int)
    obj_dist = np.repeat(np.repeat(cl_att_cl, Num_cl, axis=0), Num_cl, axis=1)
    np.fill_diagonal(obj_mask, 0)
    obj_mask = obj_mask[:, :, np.newaxis]
    koef_obj = obj_dist[:, :, :3] * obj_mask
    obj_dist = obj_dist[:, :, 3:5] * obj_mask

    block_distance = np.array([[[0, 0, 0, 0, 0], ] * Points_block.shape[0], ] * Points_obj.shape[0])

    for i, bl_cl in enumerate(block_cl):
        for case in enumerate(bl_cl):
            # print(i)
            for j, val in enumerate(bl_att_cl[i]):
                # print(int(indexs_cl[j][0]), int(indexs_cl[j][1]), int(case[1][0]), int(case[1][1]), val)
                block_distance[int(indexs_cl[j][0]):int(indexs_cl[j][1]), int(case[1][0]): int(case[1][1]), :] = val

            # block_distance[]
            # print(case[0], case[1])

    # print(Points_obj)
    return Points_obj, bloks, obj_dist, koef_obj, np.array(block_distance[:, :, 3:5]), np.array(
        block_distance[:, :, :3]), indexs_cl


if __name__ == "__main__":
    T_barrier = 10
    T_window_grav = 4
    Num_cl = [15, 16, 17]
    cl_att_cl = np.array([[[15, 0, 3, 4, 15], [5, 0, 1, 10, 65], [5, 0, 1, 13, 60]],
                          [[5, 0, 1, 10, 65], [15, 0, 3, 4, 15], [5, 0, 1, 10, 60]],
                          [[5, 0, 1, 13, 60], [5, 0, 1, 10, 60], [15, 0, 3, 4, 15]]
                          ])

    Points_block = np.array([[8, 7],
                             [9, 7],
                             [10, 7],
                             [11, 7],
                             [12, 7],
                             [13, 7],
                             [14, 7],
                             [15, 7],
                             [16, 7],
                             [17, 7],
                             [18, 7],
                             [19, 7],
                             [20, 7],
                             [20, 8],
                             [20, 9],
                             [20, 10],
                             [20, 11],
                             [20, 12],
                             [20, 13],
                             [20, 14],
                             [20, 15],
                             [21, 15],
                             [22, 15],
                             [23, 15],
                             [24, 15],
                             [25, 15],
                             [26, 15],
                             [27, 15],
                             [28, 15],
                             [29, 15],
                             [30, 15],
                             [31, 15],
                             [32, 15],
                             [33, 15],
                             [34, 15],
                             [35, 15],
                             [35, 14],
                             [35, 13],
                             [35, 12],
                             [35, 11],
                             [35, 10],
                             [35, 9],
                             [35, 8],
                             [35, 7],
                             [36, 7],
                             [37, 7],
                             [38, 7],
                             [39, 7],
                             [40, 7],
                             [41, 7],
                             [42, 8],
                             [43, 8],
                             [44, 9],
                             [45, 10],
                             [45, 11],
                             [46, 12],
                             [46, 13],
                             [47, 14],
                             [48, 15],
                             [48, 16],
                             [48, 17],
                             [49, 18],
                             [50, 19],
                             [50, 20],
                             [50, 21],
                             [51, 22],
                             [52, 23],
                             [52, 24],
                             [52, 25],
                             [52, 26],
                             [52, 27],
                             [52, 28],
                             [52, 29],
                             [52, 30],
                             [52, 31],
                             [52, 32],
                             [52, 33],
                             [51, 34],
                             [50, 35],
                             [49, 35],
                             [48, 35],
                             [47, 35],
                             [46, 35],
                             [45, 35],
                             [44, 35],
                             [43, 35],
                             [42, 35],
                             [41, 35],
                             [40, 35],
                             [39, 35],
                             [38, 35],
                             [37, 35],
                             [36, 35],
                             [35, 35],
                             [34, 35],
                             [33, 35],
                             [32, 35],
                             [31, 35],
                             [30, 35],
                             [29, 35],
                             [28, 35],
                             [27, 35],
                             [26, 35],
                             [25, 35],
                             [24, 35],
                             [23, 35],
                             [22, 35],
                             [21, 35],
                             [20, 35],
                             [19, 35],
                             [18, 35],
                             [17, 35],
                             [16, 35],
                             [15, 35],
                             [14, 35],
                             [13, 35],
                             [12, 35],
                             [11, 35],
                             [10, 35],
                             [9, 35],
                             [8, 35],
                             [8, 34],
                             [8, 33],
                             [8, 32],
                             [8, 31],
                             [8, 30],
                             [8, 29],
                             [8, 28],
                             [8, 27],
                             [8, 26],
                             [8, 25],
                             [8, 24],
                             [8, 23],
                             [8, 22],
                             [8, 21],
                             [8, 20],
                             [8, 19],
                             [8, 18],
                             [8, 17],
                             [8, 16],
                             [8, 15],
                             [8, 14],
                             [8, 13],
                             [8, 12],
                             [8, 11],
                             [8, 10],
                             [8, 9],
                             [8, 8],
                             [8, 9],
                             [8, 4],
                             [13, 4],
                             [18, 4],
                             [23, 4],
                             [23, 9],
                             [23, 12],
                             [28, 12],
                             [33, 12],
                             [33, 7],
                             [36, 4],
                             [41, 4],
                             [45, 6],
                             [48, 9],
                             [50, 12],
                             [52, 16],
                             [53, 19],
                             [55, 22],
                             [55, 27],
                             [55, 31],
                             [55, 38],
                             [50, 38],
                             [45, 38],
                             [40, 38],
                             [35, 38],
                             [30, 38],
                             [25, 38],
                             [20, 38],
                             [15, 38],
                             [10, 38],
                             [5, 38],
                             [5, 32],
                             [5, 27],
                             [5, 22],
                             [5, 17],
                             [5, 12],
                             [5, 7]
                             ])
    block_cl = [[[50, 70], [72, 74]], [[0, 50], [70, 72], [74, 149]], [[149, Points_block.shape[0]]]]
    bl_att_cl = np.array([[[T_barrier, 0, T_window_grav, 2, 4], [T_barrier, 0, 0, 2, 4], [T_barrier, 0, 0, 2, 4]],
                          [[T_barrier, 0, 0, 2, 4], [T_barrier, 0, 0, 2, 4], [T_barrier, 0, 0, 2, 4]],
                          [[T_barrier, 0, 0, 5, 7], [T_barrier, 0, 0, 5, 7], [T_barrier, 0, 0, 5, 7]]
                          ])

    Points_obj, Points_block, obj_dist, koef_obj, block_distance, koef_block, indexs_cl = get_data(cl_att_cl, Num_cl,
                                                                                                   Points_block,
                                                                                                   block_cl, bl_att_cl)
    # print(Points_obj.shape)
    print(Points_block.shape)
    # print(obj_dist.shape)
    # print(koef_obj.shape)
    # print(block_distance.shape)
    # print(koef_block)
    step = 0.001
    P2_w = Particle_wsarm(cl_att_cl, Num_cl, Points_block, block_cl, bl_att_cl)
    P2_w.process2()
    # P_w = Particle_wsarm(Points_obj, Points_block, obj_dist, koef_obj, block_distance, koef_block, indexs_cl, step)
    # P_w.process()
