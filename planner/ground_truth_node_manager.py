import torch
import matplotlib.pyplot as plt

from . import quads
from .utils import *
from .parameter import *


class GroundTruthNodeManager:
    def __init__(self, node_manager, ground_truth_map_info, device='cpu', plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.ground_truth_map_info = ground_truth_map_info
        self.pred_map_info = None
        self.ground_truth_node_coords = None
        self.ground_truth_node_utility = None
        self.explored_sign = None
        self.pred_prob = None
        self.device = device
        self.plot = plot
        self.path_to_nearest_frontier = None

        self.initialize_graph()

    def get_ground_truth_observation(self, robot_location, pred_map_info):
        self.update_graph()
        self.pred_map_info = pred_map_info

        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        for node in self.nodes_dict.__iter__():
            if node.data.explored == 0:
                all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        explored_sign = []
        pred_prob = []
        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            cell = get_cell_position_from_coords(coords, self.pred_map_info)
            prob = self.pred_map_info.map[cell[1], cell[0]]
            pred_prob.append(prob)
            utility.append(node.utility)
            explored_sign.append(node.explored)
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        explored_sign = np.array(explored_sign)
        pred_prob = np.array(pred_prob)

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = Dijkstra(self.nodes_dict, robot_location)
        guidepost = np.zeros_like(utility)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for end in utility_node_coords:
            if end[0] != robot_location[0] or end[1] != robot_location[1]:
                dist = dist_dict[(end[0], end[1])]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_utility_coords = end
        self.path_to_nearest_frontier, _ = get_Dijkstra_path_and_dist(dist_dict, prev_dict, nearest_utility_coords)
        for coords in self.path_to_nearest_frontier:
            coords_index = np.argwhere(node_coords_to_check == coords[0] + coords[1] * 1j)
            if coords_index:
                index = coords_index[0]
                guidepost[index] = 1

        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]
        
        # neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        neighbor_indices = []
        current_node_in_belief = self.node_manager.nodes_dict.find(robot_location.tolist()).data
        for neighbor in current_node_in_belief.neighbor_set:
            index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
            neighbor_indices.append(index)
        neighbor_indices = np.sort(np.array(neighbor_indices))

        self.ground_truth_node_coords = all_node_coords
        self.ground_truth_node_utility = utility
        self.explored_sign = explored_sign
        self.pred_prob = pred_prob

        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        node_predprob = pred_prob.reshape(-1, 1)
        node_guidepost = explored_sign.reshape(-1, 1)
        node_guidepost2 = guidepost.reshape(-1, 1)
        edge_mask = adjacent_matrix
        current_edge = neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                      axis=-1) / UPDATING_MAP_SIZE / 2
        #node_coords = node_coords / UPDATING_MAP_SIZE / 3
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_predprob = node_predprob / FREE
        node_inputs = np.concatenate((node_coords, node_utility, node_predprob, node_guidepost, node_guidepost2), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask],\
               [all_node_coords, utility, guidepost, explored_sign, adjacent_matrix, neighbor_indices]

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self):
        node_coords = self.get_ground_truth_node_coords(self.ground_truth_map_info)
        for coords in node_coords:
            self.add_node_to_dict(coords)

        for node in self.nodes_dict.__iter__():
            node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict)
        
    def update_graph(self):
        for node in self.node_manager.nodes_dict.__iter__():
            coords = node.data.coords
            ground_truth_node = self.nodes_dict.find(coords.tolist())
            ground_truth_node.data.utility = node.data.utility
            ground_truth_node.data.explored = 1
            ground_truth_node.data.visited = node.data.visited

    def get_ground_truth_node_coords(self, ground_truth_map_info):
        x_min = ground_truth_map_info.map_origin_x
        y_min = ground_truth_map_info.map_origin_y
        x_max = ground_truth_map_info.map_origin_x + (ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = ground_truth_map_info.map_origin_y + (ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, ground_truth_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < ground_truth_map_info.map.shape[0] and 0 <= cell[0] < ground_truth_map_info.map.shape[1]
            if ground_truth_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

        return nodes

    def plot_ground_truth_env(self, robot_location):
        plt.subplot(1, 4, 4)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.axis('off')
        robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.explored_sign, zorder=2)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)

    def plot_predicted_env(self, robot_location, belief_map):
        plt.subplot(1, 4, 3)
        plt.axis('off')
        plt.imshow(self.pred_map_info.map, cmap='gray')
        alpha_mask = (belief_map == FREE) * 0.5
        plt.imshow(belief_map, cmap='Blues', alpha=alpha_mask)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.pred_map_info)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.pred_prob, cmap='gray', s=3, zorder=2)
        for node, prob in zip(nodes, self.pred_prob):
            prob = int(prob / FREE * N_GEN_SAMPLE)
            plt.text(node[0], node[1], str(prob), fontsize=8, zorder=3)
        robot = get_cell_position_from_coords(robot_location, self.pred_map_info)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)


class Node:
    def __init__(self, coords):
        self.coords = coords
        self.utility = -(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        self.explored = 0
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def get_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, ground_truth_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
