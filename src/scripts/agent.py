import time
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as colors
import parameter

from PIL import Image
import torchvision.transforms as transforms

from utils import *
# from parameter import *
from node_manager import NodeManager
from prediction_node_manager import PredictionNodeManager


class Agent:
    def __init__(self, policy_net, predictor, device='cpu', plot=False):
        self.device = device
        self.policy_net = policy_net
        self.predictor = predictor
        self.plot = plot

        # location and map
        self.location = None
        self.map_info = None

        # map related parameters
        self.cell_size = parameter.CELL_SIZE
        self.node_resolution = parameter.NODE_RESOLUTION
        self.updating_map_size = parameter.UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = NodeManager()

        # predicted map
        self.pred_node_manager = None
        self.pred_mean_map_info, self.pred_max_map_info = None, None

        # graph
        self.node_coords, self.utility, self.guidepost = None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        # rarefied graph
        self.key_node_coords, self.key_utility, self.key_guidepost = None, None, None
        self.key_current_index, self.key_adjacent_matrix, self.key_neighbor_indices = None, None, None

    def update_map(self, map_info):
        self.map_info = map_info

    def update_updating_map(self, location):
        # the updating map is the part of the global map that maybe affected by new measurements
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        else:
            node.data.set_visited()

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                       updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
                       updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.location = self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        t0 = time.time()
        self.node_manager.get_rarefied_graph(self.location, self.map_info)
        self.key_node_coords, self.key_utility, self.key_guidepost, self.key_adjacent_matrix, self.key_current_index, self.key_neighbor_indices = \
            self.update_key_node_observation()
        t1 = time.time()
        self.update_predict_map()
        t2 = time.time()
        print(f"rarefaction time: {t1-t0:.5f}, update predict map time: {t2-t1:.5f}")

    def pre_process_input(self):
        width_in, height_in, _ = self.predictor.config['image_shape']
        height_map, width_map = self.map_info.map.shape
        print('map size:', self.map_info.map.shape)

        pad = width_map <= width_in and height_map <= height_in
        if pad:
            pad_left = (width_in - width_map) // 2
            pad_top = (height_in - height_map) // 2
            pad_right = width_in - width_map - pad_left
            pad_bottom = height_in - height_map - pad_top
            belief = np.pad(self.map_info.map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief = self.map_info.map

        trans_belief = np.full_like(belief, 0)  # align with predictor input format
        trans_belief[belief == parameter.FREE] = 255
        trans_belief[belief == parameter.OCCUPIED] = 1
        trans_belief[belief == parameter.UNKNOWN] = 127

        trans_raw = np.full_like(self.map_info.map, 0)
        trans_raw[self.map_info.map == parameter.FREE] = 255
        trans_raw[self.map_info.map == parameter.OCCUPIED] = 1
        trans_raw[self.map_info.map == parameter.UNKNOWN] = 127

        mask = np.where(belief == parameter.UNKNOWN, 255.0, 0.0)

        x_raw = Image.fromarray(trans_raw).convert('L')
        x_belief = Image.fromarray(trans_belief).convert('L')
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
            x_inpaint = self.predictor.eval_step(x_belief, mask, onehots[i], self.map_info.map.shape)
            x_inpaint_processed = self.predictor.post_process(x_inpaint, x_raw, kernel_size=5)
            x_inpaint_processed = np.where(x_inpaint_processed > 0, parameter.FREE, parameter.OCCUPIED)
            predictions.append(x_inpaint_processed)
        self.pred_mean_map_info = MapInfo(np.mean(predictions, axis=0),
                                          self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size)
        self.pred_max_map_info = MapInfo(np.min(predictions, axis=0),
                                         self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size)  # change to min for union of free space
        
        self.pred_node_manager = PredictionNodeManager(self.node_manager, self.pred_max_map_info, self.map_info, self.key_node_coords, self.location, 
                                                        device=self.device, plot=self.plot)

    def update_key_node_observation(self):
        all_key_node_coords = []
        for key_node_coords in self.node_manager.key_node_dict.keys():
            all_key_node_coords.append(np.array(key_node_coords))
        all_key_node_coords = np.array(all_key_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []

        n_nodes = all_key_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_key_node_coords[:, 0] + all_key_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_key_node_coords):
            node = self.node_manager.key_node_dict[(coords[0], coords[1])]
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                neighbor = np.array([neighbor[0], neighbor[1]])
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        return all_key_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def get_observation(self):
        [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask],\
        [self.node_coords, self.utility, self.guidepost, self.explored_sign, self.adjacent_matrix, self.neighbor_indices]\
            = self.pred_node_manager.get_predicted_observation(self.location, self.pred_mean_map_info)
        return node_inputs, None, edge_mask, current_index, current_edge, edge_padding_mask

    def get_next_observation(self, next_node_index, observation):
        node_inputs, _, edge_mask, curren_index, _, _ = observation
        next_edge = torch.argwhere(edge_mask[0, next_node_index] == 0).flatten()
        # if curren_index.item() not in next_edge:
        #     print(f">>> current index {curren_index.item()} not in next edge {next_edge}, next node index {next_node_index}")
        next_in_edge = torch.argwhere(next_edge == next_node_index).item()
        curren_in_edge = torch.argwhere(next_edge == curren_index.item()).item()  # fixme: curr index not in next edge
        k_size = next_edge.size()[-1]
        next_edge = next_edge.unsqueeze(-1).unsqueeze(0)
        next_node_index = torch.tensor([next_node_index]).reshape(1, 1, 1).to(self.device)
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, next_in_edge] = 1
        edge_padding_mask[0, 0, curren_in_edge] = 1
        return node_inputs, None, edge_mask, next_node_index, next_edge, edge_padding_mask

    def select_next_waypoint(self, observation, greedy=True):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]
        # print("available next positions:", self.key_node_coords[current_edge[0].numpy()].reshape(-1, 2))

        return next_position, next_node_index

    def plot_env(self, step, robot_location):
        # quite slow, only use it to debug

        plt.switch_backend('TKAgg')
        robot_location = self.location
        plt.ion()
        plt.clf()

        plt.subplot(1, 2, 1)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
        robot = get_cell_position_from_coords(robot_location, self.map_info)
        plt.imshow(self.map_info.map + 1.1, cmap='gray_r', norm=colors.LogNorm())
        plt.axis('off')
        utility_vis = np.where(self.utility > 0, self.utility, 0).astype(np.uint8)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=utility_vis, s=2, zorder=2)
        # for node, utility in zip(nodes, self.utility):
        #     plt.text(node[0], node[1], str(int(utility)), fontsize=4, zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=5, zorder=5)
        # for coords in self.node_coords:
        #     node = self.node_manager.nodes_dict.find(coords.tolist()).data
        #     for neighbor_coords in node.neighbor_set:
        #         end = (np.array(neighbor_coords) - coords) / 2 + coords
        #         plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
        #                  (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)
                
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(self.pred_node_manager.pred_map_info.map, cmap='gray_r')
        alpha_mask = (self.map_info.map == parameter.FREE) * 0.5
        plt.imshow(self.map_info.map, cmap='Blues', alpha=alpha_mask)
        nodes = get_cell_position_from_coords(self.pred_node_manager.pred_node_coords, self.pred_node_manager.pred_map_info)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.pred_node_manager.pred_prob, cmap='gray_r', s=2, zorder=2)
        # for node, prob in zip(nodes, self.pred_node_manager.pred_prob):
        #     prob = int(prob / parameter.OCCUPIED * parameter.N_GEN_SAMPLE)
        #     plt.text(node[0], node[1], str(prob), fontsize=4, zorder=3)
        robot = get_cell_position_from_coords(robot_location, self.pred_node_manager.pred_map_info)
        plt.plot(robot[0], robot[1], 'mo', markersize=5, zorder=5)
        # for coords in self.node_coords:
        #     node = self.pred_node_manager.nodes_dict.find(coords.tolist()).data
        #     for neighbor_coords in node.neighbor_set:
        #         end = (np.array(neighbor_coords) - coords) / 2 + coords
        #         plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
        #                  (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', linewidth=1, zorder=1)

        plt.pause(1e-3)

        plt.savefig('{}/{}_samples.png'.format(f'gifs', step), dpi=150)
        # plt.close()
