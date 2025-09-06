import os
import random
import time
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import wandb

from .trainer import Trainer
from .evaluator import Evaluator
from .dataset import Dataset
from .tools import get_config
from .logger import get_logger

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='mapinpaint/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, default=42, help='manual seed')
parser.add_argument('--wandb', action='store_true', help='use wandb for logging')


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    if cuda:
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('./checkpoints', config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(checkpoint_path)
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call
    if args.wandb:
        wandb.init(project='CogniPlan Inpainting', config=config, name=config['expname'], entity='your_entity',
                   resume='allow', id=None, notes=None)

    logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info("Configuration: {}".format(config))

    try:
        logger.info("Training on: {}".format(config['train_data_path']))
        train_dataset = Dataset(data_path=config['train_data_path'],
                                image_shape=config['image_shape'],
                                data_aug=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])

        eval_dataset = Dataset(data_path=config['eval_data_path'],
                               image_shape=config['image_shape'],
                               data_aug=False)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)

        # Define the trainer and evaluator
        trainer = Trainer(config)
        evaluator = Evaluator(config, trainer.netG, cuda)

        if cuda:
            trainer = nn.parallel.DataParallel(trainer)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        # Get the resume iteration to restart training
        start_iteration = trainer_module.resume(checkpoint_path, config['resume']) if config['resume'] else 1

        iterable_train_loader = iter(train_loader)

        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):
            try:
                ground_truth, x, mask, map_onehot, _ = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth, x, mask, map_onehot, _ = next(iterable_train_loader)

            # Prepare the inputs
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()
                map_onehot = map_onehot.cuda()

            if iteration <= config['warmup_iter']:
                # only calculate the reconstruction loss
                trainer_module.optimizer_g.zero_grad()
                compute_g_loss = True
                losses, inpainted_result = trainer(x, mask, ground_truth, map_onehot, compute_g_loss)
                losses['g'] = losses['ae'] * config['ae_loss_alpha'] + \
                              losses['l1'] * config['l1_loss_alpha'] + \
                              losses['f1'] * config['f1_loss_alpha']
                losses['g'].backward()
                trainer_module.optimizer_g.step()

            else:
                compute_g_loss = iteration % config['n_critic'] == 0
                losses, inpainted_result = trainer(x, mask, ground_truth, map_onehot, compute_g_loss)
                # Scalars from different devices are gathered into vectors
                for k in losses.keys():
                    if not losses[k].dim() == 0:
                        losses[k] = torch.mean(losses[k])

                # Update D
                trainer_module.optimizer_d.zero_grad()
                losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
                losses['d'].backward()

                # Update G
                if compute_g_loss:
                    trainer_module.optimizer_g.zero_grad()
                    losses['g'] = losses['ae'] * config['ae_loss_alpha'] + \
                                  losses['l1'] * config['l1_loss_alpha'] + \
                                  losses['f1'] * config['f1_loss_alpha'] + \
                                  losses['wgan_g'] * config['gan_loss_alpha']
                    losses['g'].backward()
                    trainer_module.optimizer_g.step()
                trainer_module.optimizer_d.step()  # put at last to prevent inplace op

            # evaluation and log
            eval_metrics = {'mae': [], 'iou': [], 'f1': []}
            if config['eval_iter'] and (iteration % config['eval_iter'] == 0):
                iterable_eval_loader = iter(eval_loader)
                for n in range(len(eval_loader)):
                    gt_e, x_e, mask_e, map_onehot_e, _ = next(iterable_eval_loader)
                    if cuda:
                        x_e = x_e.cuda()
                        mask_e = mask_e.cuda()
                        gt_e = gt_e.cuda()
                        map_onehot_e = map_onehot_e.cuda()
                    metrics, _ = evaluator.eval_step(x_e, mask_e, map_onehot_e, eval_dataset.image_raw_shape,
                                                     gt_e, calc_metrics=True)
                    for k, vl in eval_metrics.items():
                        vl.append(metrics[k])
                message = 'Eval: [%d] ' % iteration
                wangb_eval_log = {}
                for k, vl in eval_metrics.items():
                    v = np.mean(vl)
                    message += '%s: %.6f ' % (k, v)
                    k = 'eval/' + k
                    wangb_eval_log[k] = v
                    writer.add_scalar(k, v, iteration)
                if args.wandb:
                    wandb.log(wangb_eval_log, step=iteration)
                logger.info(message)

            # Log and visualization
            log_losses = ['l1', 'ae', 'f1', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                elapsed = time.time() - time_count
                speed = config['print_iter'] / elapsed
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                wandb_log = {}
                for k in log_losses:
                    v = losses.get(k, 0.)
                    message += '%s: %.6f ' % (k, v)
                    k = 'loss/' + k
                    wandb_log[k] = v
                    writer.add_scalar(k, v, iteration)
                if args.wandb:
                    wandb_log['speed'] = speed
                    wandb.log(wandb_log, step=iteration)
                message += speed_msg
                logger.info(message)

            if iteration % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                              ground_truth[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, ground_truth], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                  nrow=3 * 4,
                                  normalize=True)

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)

    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    main()
