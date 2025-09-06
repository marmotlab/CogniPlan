import os
import torch
import torch.nn as nn
from torch import autograd

from .networks import Generator, Discriminator
from .tools import get_model_list, spatial_discounting_mask
from .logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']

        self.netG = Generator(self.config['netG'], self.use_cuda)
        self.netD = Discriminator(self.config['netD'], self.use_cuda)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_d = torch.optim.Adam(self.netD.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        if self.use_cuda:
            self.netG.to('cuda')
            self.netD.to('cuda')

    def forward(self, x, masks, ground_truth, map_onehot, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x_out = self.netG(x, masks, map_onehot)
        x_inpaint = x_out * masks + x * (1. - masks)

        # D part
        # wgan d loss
        real_pred, fake_pred = self.dis_forward(self.netD, ground_truth, x_inpaint.detach())
        losses['wgan_d'] = torch.mean(fake_pred - real_pred)

        # gradients penalty loss
        gp_loss = self.calc_gradient_penalty(self.netD, ground_truth, x_inpaint.detach())
        losses['wgan_gp'] = gp_loss

        # G part
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config, masks)
            x_inpaint_01 = (x_inpaint + 1) / 2
            ground_truth_01 = (ground_truth + 1) / 2
            intersection = (x_inpaint_01 * ground_truth_01).sum(dim=(1,2,3))
            f1_score = (2 * intersection + 1e-6) / (x_inpaint_01.sum(dim=(1,2,3)) + ground_truth_01.sum(dim=(1,2,3)) + 1e-6)

            losses['l1'] = l1_loss(x_inpaint * sd_mask, ground_truth * sd_mask)
            losses['ae'] = l1_loss(x_out * (1. - masks), ground_truth * (1. - masks))
            losses['f1'] = 1 - f1_score.mean()

            # wgan g loss
            _, fake_pred_for_g = self.dis_forward(self.netD, ground_truth, x_inpaint)
            losses['wgan_g'] = - torch.mean(fake_pred_for_g)

        return losses, x_inpaint

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x_out = self.netG(x, masks)
        x_inpaint = x_out * masks + x * (1. - masks)
        return x_inpaint

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save(self.netD.state_dict(), dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        if not test:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            self.netD.load_state_dict(torch.load(last_model_name))
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration
