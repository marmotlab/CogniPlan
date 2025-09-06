import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from .utils import *
from .parameter import *
from .node_manager import NodeManager
from .ground_truth_node_manager import GroundTruthNodeManager


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
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION 
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = NodeManager(plot=self.plot)
        self.pred_node_manager = None

        # predicted map
        self.pred_mean_map_info, self.pred_max_map_info = None, None

        # graph
        self.node_coords, self.utility, self.guidepost, self.explored_sign = None, None, None, None
        self.adjacent_matrix, self.neighbor_indices = None, None

    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
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
        updating_map_origin_x = (location[
                                  0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                  1] - self.updating_map_size / 2)

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
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        self.update_predict_map()

    def pre_process_input(self):
        width_in, height_in, _ = self.predictor.config['image_shape']
        width_map, height_map = self.map_info.map.shape

        pad = width_map < width_in and height_map < height_in
        if pad:
            pad_left = (width_in - width_map) // 2
            pad_top = (height_in - height_map) // 2
            pad_right = width_in - width_map - pad_left
            pad_bottom = height_in - height_map - pad_top
            belief = np.pad(self.map_info.map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief = self.map_info.map
        mask = belief.copy()
        mask[mask != UNKNOWN] = 0
        mask[mask == UNKNOWN] = FREE

        x_raw = Image.fromarray(self.map_info.map).convert('L')
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
            _, x_inpaint = self.predictor.eval_step(x_belief, mask, onehots[i], self.map_info.map.shape)
            x_inpaint_processed = self.predictor.post_process(x_inpaint, x_raw, kernel_size=5)
            x_inpaint_processed = np.where(x_inpaint_processed > 0, FREE, OCCUPIED)
            predictions.append(x_inpaint_processed)
        self.pred_mean_map_info = MapInfo(np.mean(predictions, axis=0),
                                          self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size)
        self.pred_max_map_info = MapInfo(np.max(predictions, axis=0),
                                         self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size)
        self.pred_node_manager = GroundTruthNodeManager(self.node_manager, self.pred_max_map_info,
                                                        device=self.device, plot=self.plot)

    def get_observation(self):
        [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask],\
        [self.node_coords, self.utility, self.guidepost, self.explored_sign, self.adjacent_matrix, self.neighbor_indices]\
            = self.pred_node_manager.get_ground_truth_observation(self.location, self.pred_mean_map_info)
        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)
        if not greedy:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        else:
            action_index = torch.argmax(logp, dim=1).long()
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]

        return next_position, action_index

    def plot_env(self):
        plt.switch_backend('agg')

        plt.figure(figsize=(19, 5))
        plt.subplot(1, 4, 2)
        plt.axis('off')

        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2, zorder=4)
        robot = get_cell_position_from_coords(self.location, self.map_info)
        plt.imshow(self.pred_max_map_info.map, cmap='gray', vmin=0, vmax=255)
        alpha_mask = (self.map_info.map == FREE) * 0.5
        plt.imshow(self.map_info.map, cmap='Blues', alpha=alpha_mask)
        utility_vis = np.where(self.utility > 0, self.utility, 0).astype(np.uint8)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=utility_vis, zorder=2)
        for node, utility in zip(nodes, utility_vis):
            if utility > 0:
                plt.text(node[0], node[1], str(utility), fontsize=8, zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        guidepost_mask = np.array(self.guidepost, dtype=bool)
        if guidepost_mask.any():
            guidepost_nodes = nodes[guidepost_mask]
            plt.scatter(guidepost_nodes[:, 0], guidepost_nodes[:, 1], c='c', marker='+', zorder=4)
        for coords in self.node_coords:
            node = self.pred_node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                               (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size,
                         'tan', linewidth=1, zorder=1)
