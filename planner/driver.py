import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast  # torch.cuda.amp
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import wandb

from .model import PolicyNet, QNet
from .runner import RLRunner
from .parameter import *

ray.init()
print(f"Training CogniPlan exploration on {ray.cluster_resources()}")

writer = SummaryWriter(log_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gif_path):
    os.makedirs(gif_path)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project='CogniPlan Exploration', name=FOLDER_NAME, entity='your_entity', config=vars(parameter),
                   resume='allow', id=None, notes=None)

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs are available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # mixed precision training
    scaler = GradScaler()

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate', 'sr_room', 'sr_tunnel', 'sr_outdoor']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(27):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            buffer_size = len(experience_buffer[0])
            for i in range(len(experience_buffer)):
                assert len(experience_buffer[i]) == buffer_size

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).to(device)
                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)

                    critic_node_inputs = torch.stack(rollouts[15]).to(device)
                    critic_node_padding_mask = torch.stack(rollouts[16]).to(device)
                    critic_edge_mask = torch.stack(rollouts[17]).to(device)
                    critic_current_index = torch.stack(rollouts[18]).to(device)
                    critic_current_edge = torch.stack(rollouts[19]).to(device)
                    critic_edge_padding_mask = torch.stack(rollouts[20]).to(device)
                    critic_next_node_inputs = torch.stack(rollouts[21]).to(device)
                    critic_next_node_padding_mask = torch.stack(rollouts[22]).to(device)
                    critic_next_edge_mask = torch.stack(rollouts[23]).to(device)
                    critic_next_current_index = torch.stack(rollouts[24]).to(device)
                    critic_next_current_edge = torch.stack(rollouts[25]).to(device)
                    critic_next_edge_padding_mask = torch.stack(rollouts[26]).to(device)

                    observation = [node_inputs, node_padding_mask, edge_mask, current_index,
                                   current_edge, edge_padding_mask]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask,
                                        next_current_index, next_current_edge, next_edge_padding_mask]

                    critic_observation = [critic_node_inputs, critic_node_padding_mask, critic_edge_mask,
                                          critic_current_index,
                                          critic_current_edge, critic_edge_padding_mask]
                    critic_next_observation = [critic_next_node_inputs, critic_next_node_padding_mask,
                                               critic_next_edge_mask,
                                               critic_next_current_index, critic_next_current_edge,
                                               critic_next_edge_padding_mask]

                    # SAC
                    with torch.no_grad():
                        with autocast(device.type):
                            q_values1 = dp_q_net1(*critic_observation)
                            q_values2 = dp_q_net2(*critic_observation)
                            q_values = torch.min(q_values1, q_values2)

                    with autocast(device.type):
                        logp = dp_policy(*observation)
                        policy_loss = torch.sum(
                            (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                            dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    scaler.scale(policy_loss).backward()
                    scaler.unscale_(global_policy_optimizer)
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    scaler.step(global_policy_optimizer)
                    scaler.update()

                    with torch.no_grad():
                        with autocast(device.type):
                            next_logp = dp_policy(*next_observation)
                            next_q_values1 = dp_target_q_net1(*critic_next_observation)
                            next_q_values2 = dp_target_q_net2(*critic_next_observation)
                            next_q_values = torch.min(next_q_values1, next_q_values2)
                            value_prime = torch.sum(
                                next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                                dim=1).unsqueeze(1)
                            target_q = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    with autocast(device.type):
                        q_values1 = dp_q_net1(*critic_observation)
                        q1 = torch.gather(q_values1, 1, action)
                        q1_loss = mse_loss(q1, target_q.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    scaler.scale(q1_loss).backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    scaler.step(global_q_net1_optimizer)
                    scaler.update()

                    with autocast(device.type):
                        q_values2 = dp_q_net2(*critic_observation)
                        q2 = torch.gather(q_values2, 1, action)
                        q2_loss = mse_loss(q2, target_q.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    scaler.scale(q2_loss).backward()
                    scaler.unscale_(global_q_net2_optimizer)
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    scaler.step(global_q_net2_optimizer)
                    scaler.update()

                    with autocast(device.type):
                        entropy = (logp * logp.exp()).sum(dim=-1)
                        alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    scaler.scale(alpha_loss).backward()
                    scaler.step(log_alpha_optimizer)
                    scaler.update()

                    target_q_update_counter += 1

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
        if USE_WANDB:
            wandb.finish(quiet=True)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    (reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss,
     travel_dist, success_rate, explored_rate, sr_room, sr_tunnel, sr_outdoor) = tensorboard_data

    metrics = {
        "Losses/Value": value,
        "Losses/Policy Loss": policy_loss,
        "Losses/Alpha Loss": alpha_loss,
        "Losses/Q Value Loss": q_value_loss,
        "Losses/Entropy": entropy,
        "Losses/Policy Grad Norm": policy_grad_norm,
        "Losses/Q Value Grad Norm": q_value_grad_norm,
        "Losses/Log Alpha": log_alpha,
        "Perf/Reward": reward,
        "Perf/Travel Distance": travel_dist,
        "Perf/Explored Rate": explored_rate,
        "Perf/Success Rate": success_rate,
        "Perf/Success Rate Room": sr_room,
        "Perf/Success Rate Tunnel": sr_tunnel,
        "Perf/Success Rate Outdoor": sr_outdoor,
    }
    for k, v in metrics.items():
        writer.add_scalar(k, v, curr_episode)
    if USE_WANDB:
        wandb.log(metrics, step=curr_episode)


if __name__ == "__main__":
    main()
