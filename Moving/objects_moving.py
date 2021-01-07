import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
import networkx as nx  # import module for founding the shortest path
from PIL import Image


class Particle_wsarm:
    def __init__(self, cl_att_cl, Num_cl, bloks, blok_cl, bl_att_cl, Points_obj, step):
        self.cl_att_cl = cl_att_cl
        self.Num_cl = Num_cl
        self.bloks = bloks
        self.blok_cl = blok_cl
        self.bl_att_cl = bl_att_cl
        self.Points_obj = Points_obj
        self.step_value = step
        self.step_save = step
        self.Points_obj = np.array(self.Points_obj)
        self.preparing_data()

        self.force_loc_max = []
        self.force_block_max = []
        self.x_axis = []

        self.points_obj_x = self.Points_obj[:, 0].flatten()
        self.points_obj_y = self.Points_obj[:, 1].flatten()
        #
        self.points_block_x = self.bloks[:, 0].flatten()
        self.points_block_y = self.bloks[:, 1].flatten()

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
    #     self.force_loc_max = []
    #     self.force_block_max = []
    #     self.x_axis = []
    #
    #
    #     self.points_obj_x = self.points_obj[:, 0].flatten()
    #     self.points_obj_y = self.points_obj[:, 1].flatten()
    #
    #     self.points_block_x = self.points_block[:, 0].flatten()
    #     self.points_block_y = self.points_block[:, 1].flatten()

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

        force_block_3_copy = force_block_3 + grav_mask
        # print(force_block_3.shape)

        ind_min_dist_grav = np.unravel_index(np.argmin(force_block_3_copy, axis=1), force_block_3_copy.shape)
        # print(ind_min_dist_grav)
        # force_grav_min_dist = force_block_3.min()
        s_mask = np.zeros_like(force_block_3)
        for i, loc in enumerate(ind_min_dist_grav[1]):
            s_mask[i, loc] = force_block_3[i, loc]

        force_block = force_block_1 + force_block_2 + s_mask
        # force_block[ind_min_dist_grav[0], ind_min_dist_grav[1]] += force_grav_min_dist
        self.force_block = force_block
        # print((force_block<0).sum(), (force_block==0).sum(), (force_block>0).sum())
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
        bufer_x = self.points_obj_x + self.step_value * (self.force_block_x + self.force_loc_x)
        bufer_y = self.points_obj_y + self.step_value * (self.force_block_y + self.force_loc_y)
        bufer_x_mask = (((bufer_x < 8) * 1 + (bufer_x > 52) * 1) == 0)
        bufer_y_mask = (((bufer_y < 7) * 1 + (bufer_y > 35) * 1) == 0)

        general_mask = np.logical_and(bufer_y_mask, bufer_x_mask) * 1

        self.points_obj_x = self.points_obj_x + self.step_value * (self.force_block_x + self.force_loc_x) * general_mask
        self.points_obj_y = self.points_obj_y + self.step_value * (self.force_block_y + self.force_loc_y) * general_mask

        self.force_block_max.append(max(max(self.force_block_x), max(self.force_block_y)))
        self.force_loc_max.append(max(max(self.force_loc_x), max(self.force_loc_y)))

        # print(self.points_obj_x)

    def preparing_data(self):
        import random
        N = np.sum(self.Num_cl)
        # self.bl_att_cl[:, :, 0] = self.bl_att_cl[:, :, 0] * N
        # bl_att_cl[0, 1, 2] = bl_att_cl[0, 1, 2] * N

        addit_count_ = 0
        indexs_cl = []
        for cl in self.Num_cl:
            indexs_cl.append([addit_count_, addit_count_ + cl - 1])
            addit_count_ += cl

        # print(indexs_cl)

        obj_mask = np.ones((N, N), int)
        obj_dist = np.repeat(np.repeat(self.cl_att_cl, self.Num_cl, axis=0), self.Num_cl, axis=1)
        np.fill_diagonal(obj_mask, 0)
        obj_mask = obj_mask[:, :, np.newaxis]
        koef_obj = obj_dist[:, :, :3] * obj_mask
        obj_dist = obj_dist[:, :, 3:5] * obj_mask

        block_distance = np.array([[[0, 0, 0, 0, 0], ] * Points_block.shape[0], ] * self.Points_obj.shape[0])

        for i, bl_cl in enumerate(block_cl):
            for case in enumerate(bl_cl):
                # print(i)
                for j, val in enumerate(self.bl_att_cl[i]):
                    # print(int(indexs_cl[j][0]), int(indexs_cl[j][1]), int(case[1][0]), int(case[1][1]), val)
                    block_distance[int(indexs_cl[j][0]):int(indexs_cl[j][1] + 1), int(case[1][0]): int(case[1][1]) + 1,
                    :] = val

                # block_distance[]
                # print(case[0], case[1])

        # print(Points_obj)
        self.obj_distance = obj_dist
        self.koef_obj = koef_obj
        self.block_distance = block_distance[:, :, 3:5]
        self.koef_block = block_distance[:, :, :3]
        self.indexs_cl = indexs_cl

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    def update_progress(self, progress):
        print('\r[{0}] {1}%'.format('#' * int(progress / 10), progress))

    def process(self, type_draw='end', length=3000):
        bar = pb.ProgressBar().start()
        gif_time = 30000
        from matplotlib.legend_handler import HandlerLine2D
        self.cl_att_cl[:, :, 0] = self.cl_att_cl[:, :, 0] / 5
        # self.cl_att_cl[:, :, 0] = self.cl_att_cl[:, :, 0] / 5
        for raw in range(self.cl_att_cl.shape[0]):
            self.cl_att_cl[raw, raw, 2] = self.cl_att_cl[raw, raw, 2] * 20
            self.cl_att_cl[raw, raw, 0] = self.cl_att_cl[raw, raw, 0] * 20
        self.preparing_data()

        Bound_step = 0.3
        for t in range(0, length):
            if t == int(length * 0.6):
                for raw in range(self.cl_att_cl.shape[0]):
                    self.cl_att_cl[raw, raw, 2] = self.cl_att_cl[raw, raw, 2] / 20
                    self.cl_att_cl[raw, raw, 0] = self.cl_att_cl[raw, raw, 0] / 20
                self.cl_att_cl[:, :, 0] = self.cl_att_cl[:, :, 0] * 20
                self.preparing_data()

            if t == int(length * 0.75):
                for raw in range(self.cl_att_cl.shape[0]):
                    self.cl_att_cl[raw, raw, 2] = 0
                    # self.cl_att_cl[raw, raw, 3] = self.cl_att_cl[raw, raw, 3] + 10
                max_gravity_block = np.amax(self.bl_att_cl[:, :, 2])
                self.bl_att_cl[1, :, 2] = max_gravity_block * 5
                self.preparing_data()

            self.step_value = self.step_save * t / length
            if self.step_value >= self.step_save * Bound_step: self.step_value = self.step_save * Bound_step
            self.searching_distance_loc()
            self.searching_distance_block()
            self.searching_forces_loc()
            self.searching_forces_block_mindist()
            self.step()
            # print(self.points_obj_x.shape)
            # print(self.points_obj_y.shape)
            # self.printProgressBar(t, 1000, prefix='Progress: ', suffix='Complete')
            # print(t)
            # self.update_progress(t/1000)
            bar.update(t * 100 / length)
            self.x_axis.append(t)

            if type_draw == 'all':

                fig, ax = plt.subplots(2)
                ax[1].set_xlim(0, length)
                for i, bl_cl in enumerate(self.blok_cl):
                    color_set = 'b'
                    if i == 1:
                        color_set = 'k'
                    elif i == 2:
                        color_set = 'c'
                    for cs in bl_cl:
                        ax[0].scatter(self.points_block_x[cs[0]:cs[1]], self.points_block_y[cs[0]:cs[1]],
                                      color=color_set)

                # ax[0].scatter(self.points_block_x, self.points_block_y)
                for i, cl_raw in enumerate(self.indexs_cl):
                    color_set = 'y'
                    if i == 1:
                        color_set = 'g'
                    elif i == 2:
                        color_set = 'r'
                    ax[0].scatter(self.points_obj_x[cl_raw[0]:cl_raw[1] + 1],
                                  self.points_obj_y[cl_raw[0]:cl_raw[1] + 1], color=color_set)
                ax[1].plot(self.x_axis, self.force_block_max, label="Force_block_max", color='g')
                ax[1].plot(self.x_axis, self.force_loc_max, label="Force_loc_max", color='r')

                # ax[1].text(480, 10, t)
                ax[1].legend()

                fig.savefig(f"Scrins/band{t}.jpg", dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            if type_draw == 'end':
                if t == length - 1:
                    fig, ax = plt.subplots(2)
                    ax[1].set_xlim(0, length)
                    ax[1].set_ylim(0, 21000)
                    for i, bl_cl in enumerate(self.blok_cl):
                        color_set = 'b'
                        if i == 1:
                            color_set = 'k'
                        elif i == 2:
                            color_set = 'c'
                        for cs in bl_cl:
                            ax[0].scatter(self.points_block_x[cs[0]:cs[1]], self.points_block_y[cs[0]:cs[1]],
                                          color=color_set)

                    # ax[0].scatter(self.points_block_x, self.points_block_y)
                    for i, cl_raw in enumerate(self.indexs_cl):
                        color_set = 'y'
                        if i == 1:
                            color_set = 'g'
                        elif i == 2:
                            color_set = 'r'
                        ax[0].scatter(self.points_obj_x[cl_raw[0]:cl_raw[1] + 1],
                                      self.points_obj_y[cl_raw[0]:cl_raw[1] + 1], color=color_set)
                    ax[1].plot(self.x_axis, self.force_block_max, label="Force_block_max", color='g')
                    ax[1].plot(self.x_axis, self.force_loc_max, label="Force_loc_max", color='r')

                    # ax[1].text(480, 10, t)
                    ax[1].legend()

                    fig.savefig(f"Result.jpg", dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

        # plt.show()

        bar.finish()
        if type_draw == 'all':
            print('Udate Images')
            names = [f"Scrins/band{band}.jpg" for band in range(0, length, 10)]
            images = [Image.open(f) for f in names]
            images = [image.convert("P", palette=Image.ADAPTIVE) for image in images]
            fp_out = "image.gif"
            print('Creating GIF')

            img = images[0]
            img.save(fp=fp_out, format="GIF", append_images=images[1:], save_all=True, duration=int(gif_time / length),
                     loop=0)

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

            fig, ax = plt.subplots()
            ax.scatter(self.points_block_x, self.points_block_y)
            for cl_raw in self.indexs_cl:
                ax.scatter(self.points_obj_x[cl_raw[0]:cl_raw[1]], self.points_obj_y[cl_raw[0]:cl_raw[1]])
            fig.savefig(f"band{self.count_for_img}.jpg", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            self.count_for_img += 1

    def process2(self):
        import random
        print('Rice')
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

    def show_short_path(self, x_path, y_path, x_points, y_points, length=3000):
        fig, ax = plt.subplots(2)
        ax[1].set_xlim(0, length)
        ax[1].set_ylim(0, 21000)
        for i, bl_cl in enumerate(self.blok_cl):
            color_set = 'b'
            if i == 1:
                color_set = 'k'
            elif i == 2:
                color_set = 'c'
            for cs in bl_cl:
                ax[0].scatter(self.points_block_x[cs[0]:cs[1]], self.points_block_y[cs[0]:cs[1]],
                              color=color_set)

        # ax[0].scatter(self.points_block_x, self.points_block_y)
        for i, cl_raw in enumerate(self.indexs_cl):
            color_set = 'y'
            if i == 1:
                color_set = 'g'
            elif i == 2:
                color_set = 'r'
            ax[0].scatter(self.points_obj_x[cl_raw[0]:cl_raw[1] + 1],
                          self.points_obj_y[cl_raw[0]:cl_raw[1] + 1], color=color_set)
        ax[1].plot(self.x_axis, self.force_block_max, label="Force_block_max", color='g')
        ax[1].plot(self.x_axis, self.force_loc_max, label="Force_loc_max", color='r')

        for x_p, y_p in zip(x_path, y_path):
            ax[0].plot(x_p, y_p)

        #ax[0].scatter(x_points, y_points)
        # ax[1].text(480, 10, t)
        ax[1].legend()

        fig.savefig(f"Result_short_path_3.jpg", dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


class Corridors:
    '''
    This class have been created for founding the shortest path between drops, that is connecting inputed drops.
    Input:
    - x, y of objects' drops;
    - matrix of attention between classes;
    - amounts objects in classes;
    - x, y of drops for connection.
    '''

    def __init__(self, x_obj, y_obj, cl_att_cl, Num_cl, x_doors, y_doors):
        '''
        This object is constructor.
        :param x_obj: vector x of objects' drops;
        :param y_obj: vector y of objects' drops;
        :param cl_att_cl: matrix of attention between classes;
        :param Num_cl: amounts objects in classes;
        :param x_doors: vector x of drops for connection;
        :param y_doors: vector y of drops for connection;
        '''

        self.G = nx.Graph()  # inition of Graph

        self.x_obj = x_obj  # inition of input data
        self.y_obj = y_obj
        self.cl_att_cl = cl_att_cl
        self.Num_cl = Num_cl
        self.x_doors = np.array(x_doors)
        self.y_doors = np.array(y_doors)

        self.x_arr = []
        self.y_arr = []
        self.w_dist = []
        self.qu_w_dist = []
        self.qu_mask_bt_cl = []
        self.mask_bt_cl = []
        self.w_dist_bw = []
        self.weights = []
        self.indexes = []
        self.permited_points_x = []
        self.permited_points_y = []

    def find_ar_point(self):
        '''
        Function for founding the points between inputed drops
        '''
        qu_obj_x = np.array([self.x_obj, ] * self.x_obj.shape[0])  # transforming vec x to square matrix
        qu_obj_y = np.array([self.y_obj, ] * self.y_obj.shape[0])  # transforming vec y to square matrix
        np.fill_diagonal(qu_obj_x, 0)  # removing diagonal elements from x matrix
        np.fill_diagonal(qu_obj_y, 0)  # removing diagonal elements from y matrix

        # FOR DEBUGING: signs might be changed
        x_arr = qu_obj_x - (qu_obj_x - qu_obj_x.transpose()) / 2  # founding average points x
        y_arr = qu_obj_y - (qu_obj_y - qu_obj_y.transpose()) / 2  # founding average points y

        '''
        Average points matrix are transformed to list as the elements above diagonal 
        '''

        x_arr_list = []  # creating void list for finished average points x
        y_arr_list = []  # creating void list for finished average points y

        # FOR DEBUGING: Points could be lost due to offset
        for raw in range(0, self.x_obj.shape[0] - 1):  # indexes of element above diagonal
            for colomn in range(raw + 1, self.x_obj.shape[0]):
                x_arr_list.append(x_arr[raw, colomn])  # adding point's x to list
                y_arr_list.append(y_arr[raw, colomn])  # adding point's y to list

        self.x_arr = np.array(x_arr_list)  # adding x list into class
        self.y_arr = np.array(y_arr_list)  # adding y list into class



    def between_class_mask(self):
        """
        Function founds attention between drops: if points are in one class, value will be 1, otherwise - 0
        """
        pr_mask = np.eye(len(self.Num_cl))  # creating matrix with 1 in diagonal indexes
        qu_pr_mask = np.repeat(np.repeat(pr_mask, self.Num_cl, axis=0), self.Num_cl, axis=1)  # expanding one-matrix
        # to like attention between drops matrix
        np.fill_diagonal(qu_pr_mask, 0)  # removing elements from diagonal

        mask_between_class = []  # void list for saving attention between drops

        '''
        Attention between points matrix are transformed to list as the elements above diagonal 
        '''

        # FOR DEBUGGING: Points could be lost due to offset
        for raw in range(0, self.x_obj.shape[0] - 1):
            for column in range(raw + 1, self.x_obj.shape[0]):
                mask_between_class.append(qu_pr_mask[raw, column])  # adding 0 or 1 into attention list

        self.mask_bt_cl = np.array(mask_between_class)  # saving attention list

        qu_mask_bt_cl = np.array([self.mask_bt_cl, ] * self.mask_bt_cl.shape[0])  # transforming attention list to sq_m
        np.fill_diagonal(qu_mask_bt_cl, 0)  # removing elements from diagonal

        self.qu_mask_bt_cl = qu_mask_bt_cl  # saving weights

    def find_dist_obj(self):
        """
        Function founds weights of average points as the distant between drops.
        """
        qu_obj_x = np.array([self.x_obj, ] * self.x_obj.shape[0])  # transforming vec x to square matrix
        qu_obj_y = np.array([self.y_obj, ] * self.y_obj.shape[0])  # transforming vec y to square matrix
        np.fill_diagonal(qu_obj_x, 0)  # removing diagonal elements from x matrix
        np.fill_diagonal(qu_obj_y, 0)  # removing diagonal elements from y matrix

        pr_d_x = qu_obj_x - qu_obj_x.transpose()  # founding x distant matrix
        pr_d_y = qu_obj_y - qu_obj_y.transpose()  # founding y distant matrix

        # FOR DEBUGGING: Possibility to find distant as square root of sum square
        w_dist = np.sqrt(np.square(pr_d_x) + np.square(pr_d_y))  # common distant

        '''
        Average points matrix are transformed to list as the elements above diagonal 
        '''
        pr_dist_list = []  # void list for weights

        for raw in range(0, self.x_obj.shape[0] - 1):
            for column in range(raw + 1, self.x_obj.shape[0]):
                pr_dist_list.append(w_dist[raw, column])

        self.w_dist = np.array(pr_dist_list)  # saving distant list

        qu_w_dist = np.array([self.w_dist, ] * self.w_dist.shape[0])  # transforming distant list to square matrix
        np.fill_diagonal(qu_w_dist, 0)  # removing diagonal elements

        self.qu_w_dist = qu_w_dist  # saving weight distant matrix

    def add_points(self):
        """
        The function adds points for connection as new drops.
        That have to be used after using function for founding weights (find_dist_obj, between_class_mask), but before
        (find_dist_ar_p).
        Otherwise the find_dist_ar_p have to be recalculated.
        """
        self.x_arr = np.append(self.x_arr, self.x_doors)  # Adding x of doors
        self.y_arr = np.append(self.y_arr, self.y_doors)  # Adding y of doors

        qu_w_dist_zeros = np.zeros(
            [self.x_arr.shape[0], self.x_arr.shape[0]]) + np.max(self.qu_w_dist)  # creating the zero matrix size with new points
        qu_mask_bt_cl_zeros = np.zeros([self.x_arr.shape[0], self.x_arr.shape[0]]) + np.max(self.qu_mask_bt_cl)

        qu_w_dist_zeros[:self.qu_w_dist.shape[0], :self.qu_w_dist.shape[1]] = self.qu_w_dist  # coping the old result
        # into expended matrix
        qu_mask_bt_cl_zeros[:self.qu_mask_bt_cl.shape[0], :self.qu_mask_bt_cl.shape[1]] = self.qu_mask_bt_cl

        self.qu_mask_bt_cl = qu_mask_bt_cl_zeros  # saving the matrix of attention weights
        self.qu_w_dist = qu_w_dist_zeros  # saving the matrix of distant weights

    def find_min_dist_point_drops(self):
        a_source = []
        for x, y in zip(self.x_arr, self.y_arr):
            if np.amin(self.x_obj - x) == 0:
                print(np.min(self.y_obj - y))
            #print('Here', x, y, np.min(np.sqrt(np.square(self.x_obj - x) + np.square(self.y_obj - y))))
            a_min = np.min(np.sqrt(np.square(self.x_obj - x) + np.square(self.y_obj - y)))
            a_source.append(a_min)

        #print(a_source)
        a_source = np.array(a_source)
        min_dist_mtr = np.array([a_source, ] * a_source.shape[0])  # transforming distant list to square matrix
        np.fill_diagonal(min_dist_mtr, 0)  # removing diagonal elements

        self.min_dist_mtr = min_dist_mtr  # saving weight distant matrix


    def find_dist_ar_p(self):
        """
        Function founds weights of average points as the distant between average points.
        """
        qu_obj_x = np.array([self.x_arr, ] * self.x_arr.shape[0])  # transforming vec average x to square matrix
        qu_obj_y = np.array([self.y_arr, ] * self.y_arr.shape[0])  # transforming vec average y to square matrix
        np.fill_diagonal(qu_obj_x, 0)  # removing diagonal elements from x matrix
        np.fill_diagonal(qu_obj_y, 0)  # removing diagonal elements from x matrix

        pr_d_x = qu_obj_x - qu_obj_x.transpose()  # founding x distant matrix
        pr_d_y = qu_obj_y - qu_obj_y.transpose()  # founding y distant matrix

        # FOR DEBUGGING: Distant between points could be saved as list
        self.w_dist_bw = np.sqrt(np.square(pr_d_x) + np.square(pr_d_y))  # Getting and saving the distant between points

    def summing_w(self, k_w_dist=1, k_mask_bt_cl=2, k_dist_bw=100):
        """
        Function sums the weights (distant between average points, distant between drops, attention between drops)
        """
        # FOR DEBUGGING: The coefficient have to be rejected
        self.weights = self.qu_w_dist * k_w_dist + self.qu_mask_bt_cl * k_mask_bt_cl \
                       + self.w_dist_bw * k_dist_bw  # Summing weights

    def exception_weight(self, thr=0.5):
        """
        Function founds the points where the distant between points less then threshold
        """
        # FOR DEBUGGING: The threshold have to be rejected
        max_dist = np.max(self.min_dist_mtr)  # Founding the maximal distant
        min_dist = np.max(self.w_dist_bw)
        self.indexes = np.argwhere((self.min_dist_mtr >= max_dist * thr) & (self.w_dist_bw <= min_dist*0.25))
        # Founding the indexes less then threshold
        print('Shape', self.indexes.shape)
        #print('Indexes', self.indexes)
        for ind in self.indexes:
            if ind[0] != ind[1]:
                self.permited_points_x.append(self.x_arr[ind[0]])
                self.permited_points_y.append(self.y_arr[ind[0]])
                #self.permited_points_y.append(self.y_arr[ind[1]])
                #self.permited_points_x.append(self.x_arr[ind[0]])

        print('Swept', len(self.permited_points_x) / self.x_arr.shape[0])

    def set_graph(self):
        """
        Function fill graph
        """
        nodes = np.array(np.arange(0, self.x_arr.shape[0]))  # Creating range list for nodes names
        self.G.add_nodes_from(nodes)  # Initialling the nodes of graph
        for edge in self.indexes:
            if edge[0] == edge[1]:  # exception diagonal points
                continue
            elif edge[0] >= 1128 and edge[1] >= 1128:
                continue
            self.G.add_edge(edge[0], edge[1], weight=self.weights[edge[0], edge[1]])  # Adding edge weights

    def shortest_path(self, source_door, target_door):
        """
        The function finds the shortest path between two last points
        """
        path = nx.shortest_path(self.G, source=self.x_arr.shape[0] - source_door - 1,
                                target=self.x_arr.shape[0] - target_door - 1, weight='weight')
        path_x = self.x_arr[path]
        path_y = self.y_arr[path]
        return path_x, path_y, path

    def first_test(self):
        """
        The function tests all functions
        """
        self.find_ar_point()
        self.between_class_mask()
        self.find_dist_obj()
        self.add_points()
        self.find_dist_ar_p()
        self.summing_w()
        self.find_min_dist_point_drops()
        self.exception_weight()
        self.set_graph()
        s = (self.x_doors.shape[0], self.x_doors.shape[0])
        print(s)
        max_length = 0
        pathes_x = []
        pathes_y = []
        pathes = np.array([])
        accountant_doors = np.arange(0, self.x_doors.shape[0])
        matrix_short_pathes = np.zeros(s)
        for i in range(self.x_doors.shape[0]):
            for j in range(self.x_doors.shape[0]):
                if i != j:
                    # path_x, path_y = self.shortest_path(i, j)
                    matrix_short_pathes[i, j] = nx.shortest_path_length(self.G, source=self.x_arr.shape[0] - i - 1,
                                                                        target=self.x_arr.shape[0] - j - 1,
                                                                        weight='weight')

        print(matrix_short_pathes)
        i, j = np.unravel_index(matrix_short_pathes.argmax(), matrix_short_pathes.shape)
        max_length = matrix_short_pathes[i, j]
        print(i, j)
        path_x, path_y, path = self.shortest_path(i, j)
        pathes = np.append(pathes, path)
        pathes_x.append(path_x)
        pathes_y.append(path_y)
        accountant_doors = accountant_doors[(accountant_doors != i) & (accountant_doors != j)]
        for point in accountant_doors:
            lengthes = np.array([])
            for node in pathes:
                if self.x_arr.shape[0] - point - 1 != node:
                    length = nx.shortest_path_length(self.G, source=self.x_arr.shape[0] - point - 1, target=node,
                                                     weight='weight')
                    lengthes = np.append(lengthes, length)
                else:
                    lengthes = np.append(lengthes, max_length)

            print('Pathes', pathes)
            print('Leghtes', lengthes)
            n_nodes = np.unravel_index(lengthes.argmin(), lengthes.shape)
            n_nodes = pathes[n_nodes]
            path_2 = nx.shortest_path(self.G, source=self.x_arr.shape[0] - point - 1,
                                    target=n_nodes, weight='weight')
            print("Path", path_2, 'indexes', self.x_arr.shape[0] - point - 1, n_nodes)
            path_x = self.x_arr[path_2]
            path_y = self.y_arr[path_2]
            np.append(pathes, path_2)
            pathes_x.append(path_x)
            pathes_y.append(path_y)

        print(pathes_x)
        fig, ax = plt.subplots()
        ax.scatter(self.x_arr, self.y_arr, color='blue')
        ax.scatter(self.permited_points_x, self.permited_points_y, color='black', marker="x")
        for p_x, p_y in zip(pathes_x, pathes_y):
            ax.plot(p_x, p_y)
        fig.savefig(f"Sqr.jpg", dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        #for p_x, p_y in
        return pathes_x, pathes_y


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
    T_window_grav = 15
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
    bl_att_cl = np.array([[[T_barrier, 0, 0, 2, 4], [T_barrier, 0, T_window_grav, 2, 4], [T_barrier, 0, 0, 2, 4]],
                          [[T_barrier, 0, 0, 2, 4], [T_barrier, 0, 0, 1, 4], [T_barrier, 0, 0, 2, 4]],
                          [[T_barrier * 2, 0, 0, 5, 7], [T_barrier * 2, 0, 0, 5, 7], [T_barrier * 2, 0, 0, 5, 7]]
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
    P_w = Particle_wsarm(cl_att_cl, Num_cl, Points_block, block_cl, bl_att_cl, Points_obj, step)
    # P_w = Particle_wsarm(Points_obj, Points_block, obj_dist, koef_obj, block_distance, koef_block, indexs_cl, step)
    P_w.process()
    current_x_obj = P_w.points_obj_x
    current_y_obj = P_w.points_obj_y
    P_corridors = Corridors(current_x_obj, current_y_obj, cl_att_cl, Num_cl, [52, 8, 35, 35, 20], [31, 22, 10, 35, 10])
    result_path_x, result_path_y = P_corridors.first_test()
    potential_x_path, potential_y_path = P_corridors.permited_points_x, P_corridors.permited_points_y
    P_w.show_short_path(result_path_x, result_path_y, potential_x_path, potential_y_path)
