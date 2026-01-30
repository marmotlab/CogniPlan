import torch
from copy import deepcopy

from utils import *
import parameter
import quads
import matplotlib.pyplot as plt


class PredictionNodeManager:
    def __init__(self, node_manager, pred_map_info, map_info, key_coords, robot_loc, device='cpu', plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.pred_map_info = pred_map_info
        self.pred_node_coords = None
        self.pred_node_utility = None
        self.explored_sign = None
        self.pred_prob = None
        self.device = device
        self.plot = plot
        self.path_to_nearest_frontier = None
        self.map_info = map_info
        self.initialize_graph(key_coords, robot_loc)

    def get_predicted_observation(self, robot_location, pred_map_info):
        self.update_graph()
        self.pred_map_info = pred_map_info

        all_node_coords = []
        for node in self.nodes_dict.__iter__():
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
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        guidepost = np.zeros_like(utility)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for end in utility_node_coords:
            if end[0] != robot_location[0] or end[1] != robot_location[1]:
                dist = dist_dict[(end[0], end[1])]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_utility_coords = end
        self.path_to_nearest_frontier, _ = self.get_Dijkstra_path_and_dist(dist_dict, prev_dict, nearest_utility_coords)
        for coords in self.path_to_nearest_frontier:
            coords_index = np.argwhere(node_coords_to_check == coords[0] + coords[1] * 1j)
            if coords_index:
                index = coords_index[0]
                guidepost[index] = 1

        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]
        
        # neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        neighbor_indices = []
        current_node_in_belief = self.nodes_dict.find((robot_location[0], robot_location[1])).data
        # current_node_in_belief = self.node_manager.key_node_dict[(robot_location[0], robot_location[1])]
        for neighbor in current_node_in_belief.neighbor_set:
            index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
            neighbor_indices.append(index)
        neighbor_indices = np.sort(np.array(neighbor_indices))

        self.pred_node_coords = all_node_coords
        self.pred_node_utility = utility
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
                                      axis=-1) / parameter.UPDATING_MAP_SIZE / 2
        #node_coords = node_coords / UPDATING_MAP_SIZE / 3
        node_utility = node_utility / (parameter.SENSOR_RANGE * 3.14 // parameter.FRONTIER_CELL_SIZE)
        node_predprob = 1 - node_predprob / parameter.OCCUPIED  # probability transformed here
        node_inputs = np.concatenate((node_coords, node_utility, node_predprob, node_guidepost, node_guidepost2), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, None, edge_mask, current_index, current_edge, edge_padding_mask],\
               [all_node_coords, utility, guidepost, explored_sign, adjacent_matrix, neighbor_indices]

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self, key_coords, robot_loc):
        node_coords = self.get_predicted_node_coords(robot_loc, key_coords, self.pred_map_info)
        for coords in node_coords:
            self.add_node_to_dict(coords)

        for node in self.nodes_dict.__iter__():
            node.data.get_neighbor_nodes(self.pred_map_info, self.nodes_dict)
        
    def update_graph(self):
        for coords, node in self.node_manager.key_node_dict.items():
            if self.nodes_dict.find(coords) is None:
                pred_node = self.add_node_to_dict(coords)
                pred_node.get_neighbor_nodes(self.pred_map_info, self.nodes_dict)
            pred_node = self.nodes_dict.find(coords)
            if pred_node is not None:
                pred_node.data.utility = node.utility
                pred_node.data.explored = 1
                pred_node.data.visited = node.visited
                for neighbor in node.neighbor_set:
                    pred_node.data.neighbor_set.add((neighbor[0], neighbor[1]))
                    if self.nodes_dict.find((neighbor[0], neighbor[1])) is None:
                        new_node = self.add_node_to_dict(neighbor)
                        new_node.get_neighbor_nodes(self.pred_map_info, self.nodes_dict)

    def get_predicted_node_coords(self, location, key_coords, pred_map_info):
        x_min = pred_map_info.map_origin_x
        y_min = pred_map_info.map_origin_y
        x_max = pred_map_info.map_origin_x + (pred_map_info.map.shape[1] - 1) * parameter.CELL_SIZE
        y_max = pred_map_info.map_origin_y + (pred_map_info.map.shape[0] - 1) * parameter.CELL_SIZE

        if x_min % parameter.NODE_RESOLUTION != 0:
            x_min = (x_min // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION
        if x_max % parameter.NODE_RESOLUTION != 0:
            x_max = x_max // parameter.NODE_RESOLUTION * parameter.NODE_RESOLUTION
        if y_min % parameter.NODE_RESOLUTION != 0:
            y_min = (y_min // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION
        if y_max % parameter.NODE_RESOLUTION != 0:
            y_max = y_max // parameter.NODE_RESOLUTION * parameter.NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, parameter.NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, parameter.NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        free_connected_map = get_free_and_connected_map(location, pred_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, pred_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        # indices = np.array(indices)
        # nodes = nodes[indices].reshape(-1, 2)

        # remove non-key nodes in known map
        key_indices = []
        key_coords_to_check = key_coords[:, 0] + key_coords[:, 1] * 1j
        for idx in indices:
            coords = nodes[idx]
            cell = get_cell_position_from_coords(coords, self.map_info)
            if self.map_info.map[cell[1], cell[0]] == parameter.FREE:
                if coords[0] + coords[1] * 1j in key_coords_to_check:
                    key_indices.append(idx)
            else:
                key_indices.append(idx)
        key_indices = np.array(key_indices)
        nodes = nodes[key_indices].reshape(-1, 2)
        
        # remove nodes that are far away AND in predicted areas
        if nodes.shape[0] > 0:
            max_predicted_distance = parameter.UPDATING_MAP_SIZE / 2
            deltas = nodes - np.array(location).reshape(1, 2)
            distances = np.linalg.norm(deltas, axis=1)
            known_cells = get_cell_position_from_coords(nodes, self.map_info).reshape(-1, 2)
            known_values = self.map_info.map[known_cells[:, 1], known_cells[:, 0]]
            predicted_mask = known_values == parameter.UNKNOWN
            keep_mask = np.ones(nodes.shape[0], dtype=bool)
            keep_mask[predicted_mask] = distances[predicted_mask] <= max_predicted_distance
            print("Removed {}/{} far away predicted nodes".format(np.sum(~keep_mask), nodes.shape[0]))
            nodes = nodes[keep_mask].reshape(-1, 2)

        return nodes
    
    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            # assert self.nodes_dict.find(u) is not None

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict
    
    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            print("destination is not in Dijkstra graph")
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)
    


class Node:
    def __init__(self, coords):
        self.coords = coords
        self.utility = -(parameter.SENSOR_RANGE * 3.14 // parameter.FRONTIER_CELL_SIZE)
        self.explored = 0
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def get_neighbor_nodes(self, pred_map_info, nodes_dict):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * parameter.NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * parameter.NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, pred_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
