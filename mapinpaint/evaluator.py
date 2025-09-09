import cv2
import numpy as np

import torch
import torch.nn.functional as F


class Evaluator:
    def __init__(self, config, netG, cuda, nsample=1):
        self.config = config
        self.use_cuda = cuda
        self.nsample = nsample
        self.netG = netG
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.netG.to(self.device)

    @torch.no_grad()
    def eval_step(self, x, mask, onehot, img_raw_size, ground_truth=None, calc_metrics=False):
        self.netG.eval()
        x_out = self.netG(x, mask, onehot)
        inpainted_result = x_out * mask + x * (1. - mask)

        width, height = x.size(2), x.size(3)
        crop = img_raw_size[0] < width and img_raw_size[1] < height
        if crop:
            i_left = (width - img_raw_size[0]) // 2
            i_top = (height - img_raw_size[1]) // 2
            i_right = i_left + img_raw_size[0]
            i_bottom = i_top + img_raw_size[1]
            inpainted_result = inpainted_result[:, :, i_left:i_right, i_top:i_bottom]
        else:
            # reshape
            inpainted_result = F.interpolate(inpainted_result, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)

        if calc_metrics:
            if crop:
                x = x[:, :, i_left:i_right, i_top:i_bottom]
                ground_truth = ground_truth[:, :, i_left:i_right, i_top:i_bottom]
            else:
                x = F.interpolate(x, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)
                ground_truth = F.interpolate(ground_truth, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)
            mae, iou, f1 = calc_similarity(inpainted_result, ground_truth)
            metrics = {'mae': mae, 'iou': iou, 'f1': f1}
        else:
            metrics = {'mae': None, 'iou': None, 'f1': None}

        return metrics, inpainted_result

    @staticmethod
    def post_process(inpaint, x, kernel_size=5, return_tensor=False):
        unique_values, counts = torch.unique(x, return_counts=True)
        k = min(3, counts.size(0))
        topk_indices = torch.topk(counts, k=k).indices
        topk_values = unique_values[topk_indices]
        obs_v, free_v = topk_values.min(), topk_values.max()

        inpaint = torch.where(inpaint > -0.3, free_v, obs_v)  # binarization
        binary_img = inpaint.cpu().numpy()[0, 0]
        obs_v = obs_v.item()
        free_v = free_v.item()

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


def calc_similarity(img1, img2):
    mae = F.l1_loss(img1, img2).item()
    img1_flat = (img1 > 0).view(-1)
    img2_flat = (img2 > 0).view(-1)
    intersection = (img1_flat & img2_flat).sum().float()
    union = (img1_flat | img2_flat).sum().float()
    TP = intersection
    FP = (img1_flat & ~img2_flat).sum().float()
    FN = (~img1_flat & img2_flat).sum().float()
    iou = (intersection / union).item()
    f1 = (2 * TP / (2 * TP + FP + FN)).item()
    return mae, iou, f1

