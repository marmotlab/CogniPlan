import matplotlib.pyplot as plt
import os
import random
import skimage.io
from skimage.measure import block_reduce
import torch
from PIL import Image
import torchvision.transforms as transforms

from .sensor import *
from .graph_generator import *
from .node import *


class Env:
    def __init__(self, map_index, predictor, k_size=20, plot=False, test=False):
        self.test = test
        if self.test:
            self.map_dir = f'dataset/maps_eval_nav'
        else:
            self.map_dir = f'dataset/maps_train_nav'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.start_position, self.target_position = self.import_ground_truth(
            self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth)
        self.robot_belief = np.ones(self.ground_truth_size) * 127  # unexplored
        self.pred_mean_belief = np.ones(self.ground_truth_size) * 255
        self.pred_max_belief = np.ones(self.ground_truth_size) * 255
        self.downsampled_belief = None
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.resolution = 2  # downsample belief
        self.sensor_range = 40
        self.explored_rate = 0
        self.frontiers = None
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, target_position = self.target_position, plot=plot)
        self.graph_generator.route_node.append(self.start_position)
        self.real_node_coords, self.real_node_utility = None, None
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector, self.pred_prob, self.pred_signal = None, None, None, None, None, None, None
        self.predictor = predictor
        
        self.begin()

        self.plot = plot
        self.frame_files = []
        if self.plot:
            self.xPoints = [self.start_position[0]]
            self.yPoints = [self.start_position[1]]
            self.xTarget = [self.target_position[0]]
            self.yTarget = [self.target_position[1]]

    def find_index_from_coords(self, position, index_in_real=False):
        if index_in_real:
            index = np.argmin(np.linalg.norm(self.real_node_coords - position, axis=1))
        else:
            index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index

    def pre_process_input(self):
        width_in, height_in, _ = self.predictor.config['image_shape']
        width_map, height_map = self.robot_belief.shape

        pad = width_map < width_in and height_map < height_in
        if pad:
            pad_left = (width_in - width_map) // 2
            pad_top = (height_in - height_map) // 2
            pad_right = width_in - width_map - pad_left
            pad_bottom = height_in - height_map - pad_top
            belief = np.pad(self.robot_belief, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief = self.robot_belief
        mask = belief.copy()
        mask[mask != 127] = 0
        mask[mask == 127] = 255

        x_raw = Image.fromarray(self.robot_belief).convert('L')
        x_belief = Image.fromarray(belief).convert('L')
        mask = Image.fromarray(mask).convert('1')
        if not pad:
            x_belief = transforms.Resize((width_in, height_in))(x_belief)
            mask = transforms.Resize((width_in, height_in))(mask)
        x_belief = transforms.ToTensor()(x_belief).unsqueeze(0).to(self.predictor.device)
        x_belief = x_belief.mul_(2).add_(-1)
        x_raw = transforms.ToTensor()(x_raw).unsqueeze(0).to(self.predictor.device)
        x_raw = x_raw.mul_(2).add_(-1)
        mask = transforms.ToTensor()(mask).unsqueeze(0).to(self.predictor.device)
        return x_belief, mask, x_raw

    def update_predict_map(self):
        x_belief, mask, x_raw = self.pre_process_input()
        onehots = torch.tensor([[0.333, 0.333, 0.333], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]).unsqueeze(1).float().to(x_belief.device)
        predictions = []
        for i in range(self.predictor.nsample):
            _, x_inpaint = self.predictor.eval_step(x_belief, mask, onehots[i], self.robot_belief.shape)
            x_inpaint_processed = self.predictor.post_process(x_inpaint, x_raw, kernel_size=5)
            x_inpaint_processed = np.where(x_inpaint_processed > 0, 255, 1)
            predictions.append(x_inpaint_processed)
        self.pred_mean_belief = np.mean(predictions, axis=0)
        self.pred_max_belief = np.max(predictions, axis=0)

    def begin(self):
        self.robot_belief = self.update_robot_belief(self.start_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.update_predict_map()
        self.real_node_coords, _, self.real_node_utility, _, _ = self.graph_generator.generate_graph(
            self.start_position, self.ground_truth, self.robot_belief, self.frontiers)
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector, self.pred_prob, self.pred_signal = \
            self.graph_generator.update_predict_graph(self.start_position, self.pred_max_belief, self.pred_mean_belief, self.frontiers)

    def step(self, robot_position, next_position, travel_dist):
        dist = np.linalg.norm(robot_position - next_position)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        astar_dist_cur_to_target, _ = self.graph_generator.find_shortest_path(robot_position, self.target_position, 
                                                                           self.graph_generator.ground_truth_node_coords, self.graph_generator.ground_truth_graph)
        astar_dist_next_to_target, _ = self.graph_generator.find_shortest_path(next_position, self.target_position, 
                                                                            self.graph_generator.ground_truth_node_coords, self.graph_generator.ground_truth_graph)
        travel_dist += dist
        robot_position = next_position
        self.graph_generator.route_node.append(robot_position)
        next_node_index = self.find_index_from_coords(robot_position, index_in_real=True)
        self.graph_generator.nodes_list[next_node_index].set_visited()
        self.robot_belief = self.update_robot_belief(robot_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()
        self.update_predict_map()
        reward, done = self.calculate_reward(astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target)
        if self.plot:
            self.xPoints.append(robot_position[0])
            self.yPoints.append(robot_position[1])
        self.real_node_coords, _, self.real_node_utility, _, _ = self.graph_generator.update_graph(
            robot_position, self.robot_belief, self.old_robot_belief, frontiers, self.frontiers)
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector, self.pred_prob, self.pred_signal = \
            self.graph_generator.update_predict_graph(self.start_position, self.pred_max_belief, self.pred_mean_belief, frontiers)
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.frontiers = frontiers

        return reward, done, robot_position, travel_dist
    
    def import_ground_truth_pp(self, map_index):
        ground_truth = (skimage.io.imread(map_index, 1) * 255).astype(int)
        robot_location = np.nonzero(ground_truth == 209)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        target_location = np.nonzero(ground_truth == 68)
        target_location = np.array([np.array(target_location)[1, 127], np.array(target_location)[0, 127]])
        ground_truth = (ground_truth > 150)|((ground_truth<=80)&(ground_truth>=60))
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location, target_location

    def import_ground_truth(self, map_index):
        ground_truth = (skimage.io.imread(map_index, 1)).astype(int)
        if not self.test:
            rot = random.randint(0, 4)
            ground_truth = np.rot90(ground_truth, rot)
        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 10], np.array(robot_location)[0, 10]])
        target_location = np.nonzero(ground_truth == 232)
        target_location = np.array([np.array(target_location)[1, 10], np.array(target_location)[0, 10]])
        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location, target_location
    
    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief

    def calculate_reward(self, astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target):
        reward = 0
        done = False
        reward -= 0.5
        reward += (astar_dist_cur_to_target - astar_dist_next_to_target) / 64
        if dist_to_target == 0:
            reward += 20
            done = True
        return reward, done

    def evaluate_exploration_rate(self):
        rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def find_frontier(self):
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        mapping = (mapping == 127) * 1
        mapping = np.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)
        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        f = points[ind_to]
        f = f.astype(int)
        f = f * self.resolution
        return f

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        plt.figure(figsize=(10, 5))
        # plt.ion()
        # plt.cla()
        plt.subplot(1, 2, 1)
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        plt.imshow(self.robot_belief, cmap='gray')
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)  # plot edges will take long time
        # plt.scatter(self.real_node_coords[:, 0], self.real_node_coords[:, 1], c=self.real_node_utility, zorder=5)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.plot(self.xTarget, self.yTarget, 'o', markersize = 5)
        plt.plot(self.xPoints, self.yPoints, 'b', linewidth=2)
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'mo', markersize=8)
        # plt.plot(self.xPoints[0], self.yPoints[0], 'co', markersize=8)

        plt.subplot(1, 2, 2)
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        plt.imshow(self.pred_mean_belief, cmap='gray')
        alpha_mask = (self.robot_belief == 255) * 0.5
        plt.imshow(self.robot_belief, cmap='Blues', alpha=alpha_mask)
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=self.pred_prob, cmap='gray', s=5, zorder=2)
        for node, prob in zip(self.node_coords, self.pred_prob):
            prob = int(prob / 255 * self.predictor.nsample)
            plt.text(node[0], node[1], str(prob), fontsize=8, zorder=3)
        plt.plot(self.xPoints, self.yPoints, 'b', linewidth=2)
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'mo', markersize=8)
        # plt.plot(self.xPoints[0], self.yPoints[0], 'co', markersize=8)
        plt.plot(self.xTarget, self.yTarget, 'o', markersize = 5)

        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, travel_dist))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)