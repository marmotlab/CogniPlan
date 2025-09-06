import torch

from .env import Env
from .agent import Agent
from .utils import *
from .model import PolicyNet
from .ground_truth_node_manager import GroundTruthNodeManager

if not os.path.exists(gif_path):
    os.makedirs(gif_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, predictor, self.device, self.save_image)

        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(27):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        ground_truth_observation, _ = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location,
                                                                                                  self.robot.pred_mean_map_info)

        if self.save_image:
            self.robot.plot_env()
            self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            self.save_observation(observation, ground_truth_observation)

            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_set)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location)

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if not (self.robot.utility > 0).any() or self.env.explored_rate > 0.9999:
                done = True
                reward += 20
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            ground_truth_observation, _ = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location,
                                                                                                      self.robot.pred_mean_map_info)
            self.save_next_observations(observation, ground_truth_observation)

            if self.save_image:
                self.robot.plot_env()
                self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i+1)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['sr_room'] = done if 'room' in self.env.map_path else np.nan
        self.perf_metrics['sr_tunnel'] = done if 'tunnel' in self.env.map_path else np.nan
        self.perf_metrics['sr_outdoor'] = done if 'outdoor' in self.env.map_path else np.nan

        # save gif
        if self.save_image:
            make_gif(gif_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[15] += critic_node_inputs
        self.episode_buffer[16] += critic_node_padding_mask.bool()
        self.episode_buffer[17] += critic_edge_mask.bool()
        self.episode_buffer[18] += critic_current_index
        self.episode_buffer[19] += critic_current_edge
        self.episode_buffer[20] += critic_edge_padding_mask.bool()

        assert torch.all(current_edge == critic_current_edge), print(current_edge, critic_current_edge, current_index, critic_current_index)
        assert torch.all(node_inputs[0, current_index.item(), :2] == critic_node_inputs[0, critic_current_index.item(), :2]), print(node_inputs[0, current_index.item()], critic_node_inputs[0, critic_current_index.item()])
        assert torch.all(torch.gather(node_inputs, 1, current_edge.repeat(1, 1, 2)) == torch.gather(critic_node_inputs, 1, critic_current_edge.repeat(1, 1, 2)))

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[21] += critic_node_inputs
        self.episode_buffer[22] += critic_node_padding_mask.bool()
        self.episode_buffer[23] += critic_edge_mask.bool()
        self.episode_buffer[24] += critic_current_index
        self.episode_buffer[25] += critic_current_edge
        self.episode_buffer[26] += critic_edge_padding_mask.bool()

if __name__ == "__main__":
    import yaml
    import os
    from mapinpaint.networks import Generator
    from mapinpaint.evaluator import Evaluator

    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    checkpoint = torch.load('./' + model_path + '/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['policy_model'])
    generator_path = './' + generator_path

    config_path = f'{generator_path}/config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    netG = Generator(config['netG'], USE_GPU)
    checkpoint_path = os.path.join(generator_path, [f for f in os.listdir(generator_path)
                                                    if f.startswith('gen') and f.endswith('.pt')][0])
    netG.load_state_dict(torch.load(checkpoint_path))
    evaluator = Evaluator(config, netG, USE_GPU, N_GEN_SAMPLE)

    if USE_GPU:
        netG = netG.to('cuda')
    netG.eval()
    worker = Worker(0, model, evaluator, 77, save_image=True)
    worker.run_episode()
