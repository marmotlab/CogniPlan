import time
import heapq
import numpy as np
from . import quads
from .utils import *
from .parameter import *


class NodeManager:
    def __init__(self, plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.frontier = None

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, frontiers, updating_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, frontiers, updating_map_info)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.nodes_dict.find(neighbor_coords)
                neighbor_node.data.neighbor_set.remove(node.coords.tolist())
        self.nodes_dict.remove(node.coords.tolist())

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        if self.frontier is None:
            new_frontier = frontiers

        else:
            new_frontier = frontiers - self.frontier
            new_out_range= []
            for frontier in new_frontier:
                if np.linalg.norm(robot_location - np.array(frontier).reshape(2)) > SENSOR_RANGE + FRONTIER_CELL_SIZE:
                    new_out_range.append(frontier)
            for frontier in new_out_range:
                new_frontier.remove(frontier)

        self.frontier = frontiers

        all_node_list = []
        global_frontiers = get_frontier_in_map(map_info)
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info)
            else:
                node = node.data
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(new_frontier, global_frontiers, updating_map_info)
            all_node_list.append(node)

        for node in all_node_list:
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    SENSOR_RANGE + NODE_RESOLUTION):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)


class Node:
    def __init__(self, coords, frontiers, updating_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            self.utility = len(observable_frontiers)
            if self.utility <= MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        #cell = get_cell_position_from_coords(neighbor_coords, updating_map_info)
                        #if updating_map_info.map[cell[1], cell[0]] == 1:
                        #    self.neighbor_matrix[i, j] = 1
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, new_frontiers, global_frontiers, updating_map_info):
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if frontier not in global_frontiers:
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        if len(new_frontiers) > 0:
            new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0
