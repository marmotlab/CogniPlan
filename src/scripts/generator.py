import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

import cv2
import numpy as np


class Generator(nn.Module):
    def __init__(self, config, use_cuda):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.generator = ImageGenerator(self.input_dim, self.cnum, self.use_cuda)

    def forward(self, x, mask, onehot):
        x_out = self.generator(x, mask, onehot)
        return x_out


class ImageGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True):
        super(ImageGenerator, self).__init__()
        self.use_cuda = use_cuda

        self.conv1 = gen_conv(input_dim + 4, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum * 2, 3, 2, 1)
        self.conv3 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        self.conv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.conv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.conv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.conv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.conv17 = gen_conv(cnum // 2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask, onehot):
        onehot_expanded = onehot.view(onehot.size(0), onehot.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        if self.use_cuda:
            mask = mask.cuda()
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, onehot_expanded, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x = torch.clamp(x, -1., 1.)
        return x

    
def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class Evaluator:
    def __init__(self, config, netG, device, nsample=1):
        self.config = config
        self.nsample = nsample
        self.netG = netG
        self.device = device
        self.netG.to(self.device)

    @torch.no_grad()
    def eval_step(self, x, mask, onehot, img_raw_size):
        self.netG.eval()
        x_out = self.netG(x, mask, onehot)
        inpainted_result = x_out * mask + x * (1. - mask)

        width, height = x.size(2), x.size(3)
        crop = img_raw_size[0] <= width and img_raw_size[1] <= height
        if crop:
            pass
            i_left = (width - img_raw_size[0]) // 2
            i_top = (height - img_raw_size[1]) // 2
            i_right = i_left + img_raw_size[0]
            i_bottom = i_top + img_raw_size[1]
            inpainted_result = inpainted_result[:, :, i_left:i_right, i_top:i_bottom]
        else:
            # reshape
            inpainted_result = F.interpolate(inpainted_result, size=(img_raw_size[0], img_raw_size[1]), mode='bilinear', align_corners=False)

        return inpainted_result

    @staticmethod
    def post_process(inpaint, x, kernel_size=5, return_tensor=False):
        unique_values, counts = torch.unique(x, return_counts=True)
        k = min(3, counts.size(0))
        topk_indices = torch.topk(counts, k=k).indices
        topk_values = unique_values[topk_indices]
        obs_v, free_v = topk_values.min(), topk_values.max()

        inpaint = torch.where(inpaint > -0.3, free_v, obs_v)  # binarization
        binary_img = inpaint.cpu().numpy()[0, 0]
        obs_v, free_v = obs_v.item(), free_v.item()

        mask = np.zeros_like(binary_img, dtype=np.uint8)
        mask[binary_img == free_v] = 255
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close op to fill small holes
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)  # open op to remove small noise
        morph_clean_img = np.where(opening == 255, free_v, obs_v).astype(binary_img.dtype)
        x_array = x.cpu().numpy()[0, 0]

        morph_clean_img = np.where((x_array == obs_v) | (x_array == free_v), x_array, morph_clean_img)
        if return_tensor:
            morph_clean_img = torch.from_numpy(morph_clean_img).unsqueeze(0).unsqueeze(0).float().to(inpaint.device)
        return morph_clean_img
    
