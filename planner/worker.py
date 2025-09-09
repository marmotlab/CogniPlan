import copy
import imageio
import numpy as np
import torch
from .env import Env
from .model import PolicyNet
from .parameter import *

class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, predictor, global_step, device='cuda', greedy=False, save_image=True):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, predictor=predictor, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        indicator = copy.deepcopy(self.env.indicator)
        direction_vector = copy.deepcopy(self.env.direction_vector)
        pred_prob = copy.deepcopy(self.env.pred_prob)
        pred_signal = copy.deepcopy(self.env.pred_signal)
        # normalize observations
        node_coords = node_coords / 250
        node_utility = node_utility / 50
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape(n_nodes, 1)
        direction_nums = direction_vector.shape[0]
        direction_vector_inputs = direction_vector.reshape(direction_nums, 3)
        direction_vector_inputs[:, 2] /= 40
        indicator = indicator.reshape(n_nodes, 1)
        pred_prob = pred_prob.reshape(n_nodes, 1) / 255
        pred_signal = pred_signal.reshape(n_nodes, 1)
        node_inputs = np.concatenate((node_coords, node_utility_inputs, indicator, direction_vector_inputs, pred_prob, pred_signal), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)
        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)
        current_edge = edge_inputs[current_node_index]
        real_neighbor = pred_signal[current_edge]  # 1-real, 0-predicted

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)
        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(0)
        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
        # calculate a mask for the padded edges (denoted by 0)
        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        for idx, value in enumerate(real_neighbor):  # mask predicted neighbors
            if value[0] == 0:
                edge_padding_mask[0, 0, idx] = 1

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask)
        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
        next_node_index = edge_inputs[0, 0, action_index.item()]
        next_position = self.env.node_coords[next_node_index]
        return next_position, action_index

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_padding_mask.bool())
        self.episode_buffer[4] += copy.deepcopy(edge_padding_mask.bool())
        self.episode_buffer[5] += copy.deepcopy(edge_mask.bool())

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[8] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[9] += copy.deepcopy(node_inputs)
        self.episode_buffer[10] += copy.deepcopy(edge_inputs)
        self.episode_buffer[11] += copy.deepcopy(current_index)
        self.episode_buffer[12] += copy.deepcopy(node_padding_mask.bool())
        self.episode_buffer[13] += copy.deepcopy(edge_padding_mask.bool())
        self.episode_buffer[14] += copy.deepcopy(edge_mask.bool())

    def run_episode(self, curr_episode):
        done = False
        observations = self.get_observations()
        for i in range(128):
            self.save_observations(observations)
            next_position, action_index = self.select_node(observations)
            self.save_action(action_index)
            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position, self.travel_dist)
            self.save_reward_done(reward, done)
            observations = self.get_observations()
            self.save_next_observations(observations)
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)
            if done:
                break
        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def work(self, currEpisode):
        self.run_episode(currEpisode)

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


if __name__ == "__main__":
    import yaml
    import os
    from mapinpaint.networks import Generator
    from mapinpaint.evaluator import Evaluator
    net = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
    checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    net.load_state_dict(checkpoint['policy_model'])
    config_path = f'{generator_path}/config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    netG = Generator(config['netG'], USE_GPU)
    checkpoint_path = os.path.join(generator_path, [f for f in os.listdir(generator_path)
                                                    if f.startswith('gen') and f.endswith('.pt')][0])
    netG.load_state_dict(torch.load(checkpoint_path))
    evaluator = Evaluator(config, netG, USE_GPU, N_GEN_SAMPLE)

    worker = Worker(0, net, None, evaluator, 0, device='cpu', greedy=False, save_image=False)
    worker.run_episode(0)

