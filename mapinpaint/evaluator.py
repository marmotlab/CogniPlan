import os
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from .dataset import Dataset
from .networks import Generator
from .tools import get_config


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


def main():
    run_path = './checkpoints/wgan_inpainting'
    config_path = f'{run_path}/config.yaml'
    checkpoint_path = os.path.join(run_path, [f for f in os.listdir(run_path) if f.startswith('gen') and f.endswith('.pt')][0])
    save_img = True
    save_csv = True
    nsample = 4  # set 1, >=2 to set the number of samples
    if save_img:
        os.makedirs(f"{run_path}/images", exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help="testing configuration")
    parser.add_argument('--seed', type=int, default=0, help='manual seed')

    args = parser.parse_args()
    config = get_config(args.config)
    print("Arguments: {}".format(args))

    # CUDA configuration
    cuda = config['cuda']
    if cuda:
        cudnn.benchmark = True
    print("Configuration: {}".format(config))

    # Define the trainer
    netG = Generator(config['netG'], cuda)
    netG.load_state_dict(torch.load(checkpoint_path))
    print("Resume from {}".format(checkpoint_path))

    evaluator = Evaluator(config, netG, cuda, nsample)

    # Dataset
    # config['eval_data_path'] = '../dataset/kth_test_maps/50052748'
    eval_dataset = Dataset(data_path=config['eval_data_path'],
                           image_shape=config['image_shape'],
                           data_aug=False)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    iterable_eval_loader = iter(eval_loader)

    results = []
    similarities = []

    for n in range(len(eval_loader)):
        ground_truth, x, mask, _, (ground_truth_raw, x_raw, mask_raw) = next(iterable_eval_loader)
        if cuda:
            x = x.cuda()
            mask = mask.cuda()
            ground_truth = ground_truth.cuda()
            x_raw = x_raw.cuda()
            ground_truth_raw = ground_truth_raw.cuda()
        explored_rate = ((x > 0.99).sum() / (ground_truth > 0.99).sum()).item()

        all_inpaints = []
        all_inpaints_processed = []
        onehots = torch.tensor([[0.333,0.333,0.333],[1,0,0],[0,1,0],[0,0,1],
                                [0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6]]).unsqueeze(1).float().to(x.device)

        for i in range(nsample):
            metrics, inpainted_result = evaluator.eval_step(x, mask, onehots[i], eval_dataset.image_raw_shape,
                                                            ground_truth, calc_metrics=True)

            inpaint_processed = evaluator.post_process(inpainted_result, x_raw, return_tensor=True)
            all_inpaints.append(inpainted_result)
            all_inpaints_processed.append(inpaint_processed)

            log_metrics = {"explored_rate": explored_rate,
                           "mae": metrics['mae'],
                           "iou": metrics['iou'],
                           "f1": metrics['f1']}

            results.append(log_metrics)

        for i in range(nsample):
            for j in range(i + 1, nsample):
                f1 = calc_similarity(all_inpaints[i], all_inpaints[j])[2]
                similarities.append(f1)

        if save_img:
            avg_inpainted_result = torch.stack(all_inpaints, dim=0).mean(dim=0)
            avg_inpainted_processed = torch.stack(all_inpaints_processed, dim=0).mean(dim=0)
            viz_images = torch.stack([x_raw, avg_inpainted_result, avg_inpainted_processed, ground_truth_raw], dim=1)
            viz_images = viz_images.view(-1, *list(x_raw.size())[1:])
            vutils.save_image(viz_images,
                              f"{run_path}/images/{n:03d}.png",
                              normalize=True)
            if nsample > 1:
                # all_inpaints = torch.cat(all_inpaints, dim=0)
                all_inpaints_processed = torch.cat(all_inpaints_processed, dim=0)
                vutils.save_image(all_inpaints_processed,
                                  f"{run_path}/images/{n:03d}_sample.png",
                                  normalize=True)

    if save_csv:
        df = pd.DataFrame(results)
        df = df.sort_values(by="explored_rate", ascending=True)
        df.to_csv(f"{run_path}/traj_metrics.csv", index=False)
    print(f"Mean mae: {np.mean([y['mae'] for y in results]):.4f}")
    print(f"Mean iou: {np.mean([y['iou'] for y in results]):.4f}")
    print(f"Mean f1: {np.mean([y['f1'] for y in results]):.4f}")
    print(f"Mean similarity between prediction: {np.mean(similarities):.4f}")


if __name__ == '__main__':
    main()
