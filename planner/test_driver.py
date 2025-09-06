import ray
import torch
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .model import PolicyNet
from .env import Env
from .agent import Agent
from .ground_truth_node_manager import GroundTruthNodeManager
from .utils import *
from .parameter import *
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator

# Other configuration settings in parameter.py
model_name = 'cogniplan_exp_pred7'
NUM_TEST = 150
NUM_META_AGENT = 10  # number of parallel tests, NUM_TEST % NUM_META_AGENT should be 0
SAFE_MODE = True
SAVE_GIFS = False

if SAVE_GIFS:
    os.makedirs(gif_path, exist_ok=True)


def run_test():
    device = torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    print(f"Testing on {device}, model: {model_name}, num of tests: {NUM_TEST}, num of samples: {N_GEN_SAMPLE}")
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', weights_only=True, map_location=device)

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    travel_dist = []
    explored_rate = []
    success_rate = []
    sr_room = []
    sr_tunnel = []
    sr_outdoor = []
    td_room = []
    td_tunnel = []
    td_outdoor = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(travel_dist) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                travel_dist.append(metrics['travel_dist'])
                explored_rate.append(metrics['explored_rate'])
                success_rate.append(metrics['success_rate'])
                if 'room' in info['map_path']:
                    sr_room.append(metrics['success_rate'])
                    td_room.append(metrics['travel_dist'])
                elif 'tunnel' in info['map_path']:
                    sr_tunnel.append(metrics['success_rate'])
                    td_tunnel.append(metrics['travel_dist'])
                elif 'outdoor' in info['map_path']:
                    sr_outdoor.append(metrics['success_rate'])
                    td_outdoor.append(metrics['travel_dist'])

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        print('=====================================')
        print('| Test:', FOLDER_NAME)
        print('| Total test: {} with {} predictions'.format(NUM_TEST, N_GEN_SAMPLE))
        print('| Average success rate:', np.array(success_rate).mean())
        print('| Average travel distance:', np.array(travel_dist).mean())
        print('| Average explored rate:', np.array(explored_rate).mean())
        print('| Room success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(np.mean(sr_room), np.mean(td_room), np.std(td_room)))
        print('| Tunnel success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(np.mean(sr_tunnel), np.mean(td_tunnel), np.std(td_tunnel)))
        print('| Outdoor success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(np.mean(sr_outdoor), np.mean(td_outdoor), np.std(td_outdoor)))


    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image, test=True)

        self.robot = Agent(policy_net, predictor, self.device, self.save_image)

        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)

        self.perf_metrics = dict()
        self.location_history = []
        self.loop_detected = False
        self.path_to_nearest_frontier = None

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location, self.robot.pred_mean_map_info)

        if self.save_image:
            self.robot.plot_env()
            self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            self.location_history.append(self.env.robot_location)
            if SAFE_MODE and len(self.location_history) >= 4:
                if np.array_equal(self.location_history[-1], self.location_history[-3]) and \
                   np.array_equal(self.location_history[-2], self.location_history[-4]):
                    print("Loop detected, go to nearest frontier")
                    self.loop_detected = True
                    self.path_to_nearest_frontier = self.robot.pred_node_manager.path_to_nearest_frontier.copy()
            if self.loop_detected:
                next_location = np.array(self.path_to_nearest_frontier.pop(0))
                if len(self.path_to_nearest_frontier) == 0:
                    self.loop_detected = False
                node_exist = self.robot.node_manager.nodes_dict.find((next_location[0], next_location[1]))
                if node_exist is None:
                    next_location, action_index = self.robot.select_next_waypoint(observation, greedy=True)
                    self.loop_detected = False
            else:
                next_location, action_index = self.robot.select_next_waypoint(observation, greedy=True)

            self.env.step(next_location)
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if not (self.robot.utility > 0).any() or self.env.explored_rate > 0.9999:
                done = True
            observation = self.robot.get_observation()
            self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location, self.robot.pred_mean_map_info)

            if self.save_image:
                self.robot.plot_env()
                self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i+1)

            if done:
                break

        if not done:
            print(f"Exploration not completed in env: {self.env.map_path}")

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gif_path, self.global_step, self.env.frame_files, self.env.explored_rate)


@ray.remote(num_cpus=1)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.worker = None
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)
        self.predictor = self.load_predictor()

    def load_predictor(self):
        config_path = f'{generator_path}/config.yaml'
        checkpoint_path = os.path.join(generator_path, [f for f in os.listdir(generator_path)
                                                        if f.startswith('gen') and f.endswith('.pt')][0])
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        generator = Generator(config['netG'], USE_GPU)
        generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.predictor = Evaluator(config, generator, USE_GPU, N_GEN_SAMPLE)
        print("Map predictor loaded from {}".format(checkpoint_path))
        return self.predictor

    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        self.worker = TestWorker(self.meta_agent_id, self.network, self.predictor, episode_number, save_image=SAVE_GIFS)
        self.worker.run_episode()
        perf_metrics = self.worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
            "map_path": self.worker.env.map_path,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()
