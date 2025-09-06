import os
import yaml
import torch
import ray

from .model import PolicyNet
from .worker import Worker
from .parameter import *
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
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

    def get_weights(self):
        return self.network.state_dict()

    def set_policy_net_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        # save_img = True
        worker = Worker(self.meta_agent_id, self.network, self.predictor, episode_number, device=self.device,
                        save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_policy_net_weights(weights_set[0])

        job_results, metrics = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, metrics, info


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
