import os
from copy import deepcopy
from itertools import permutations

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm

from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

__all__ = ['MnistRunner']

BATCH_SIZE = 64
N = 2  # Number of digits

def psnr(est, gt):
    """Returns the P signal to noise ratio between the estimate and gt"""
    return float(-10 * torch.log10(((est - gt) ** 2).mean()).detach().cpu())

def gehalf(input_tensor):
    """Returns a sigmoid proxy for x > 0.5"""
    return 1 / (1 + torch.exp(-5 * (input_tensor - 0.5)))

def get_images_split(first_digits, second_digits):
    """Returns two images, one from [0,4] and the other from [5,9]"""
    rand_idx_1 = np.random.randint(0, first_digits.shape[0] - 1, BATCH_SIZE)
    rand_idx_2 = np.random.randint(0, second_digits.shape[0] - 1, BATCH_SIZE)

    image1 = first_digits[rand_idx_1, :, :].float().view(BATCH_SIZE, 1, 28, 28) / 255.
    image2 = second_digits[rand_idx_2, :, :].float().view(BATCH_SIZE, 1, 28, 28) / 255.

    return image1, image2

def get_images_no_split(dataset):
    image1_batch = torch.zeros(BATCH_SIZE, 28, 28)
    image2_batch = torch.zeros(BATCH_SIZE, 28, 28)
    for idx in range(BATCH_SIZE):
        idx1 = np.random.randint(0, len(dataset))
        image1 = dataset.data[idx1]
        image1_label = dataset[idx1][1]
        image2_label = image1_label

        # Continously sample image2 until not same label
        while image1_label == image2_label:
            idx2 = np.random.randint(0, len(dataset))
            image2 = dataset.data[idx2]
            image2_label = dataset[idx2][1]

        image1_batch[idx] = image1
        image2_batch[idx] = image2

    image1_batch = image1_batch.float().view(BATCH_SIZE, 1, 28, 28) / 255.
    image2_batch = image2_batch.float().view(BATCH_SIZE, 1, 28, 28) / 255.

    return image1_batch, image2_batch

def get_single_image(dataset):
    """Returns two images, one from [0,4] and the other from [5,9]"""
    rand_idx = np.random.randint(0, dataset.data.shape[0] - 1, BATCH_SIZE)
    image = dataset.data[rand_idx].float().view(BATCH_SIZE, 1, 28, 28) / 255.

    return image


class MnistRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        # Load MNIST test set
        dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, transform=transforms.ToTensor(), download=True)
        batch_size = 128  # Use 128 so x0 and x1 are size 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, (x, y) in enumerate(tqdm.tqdm(dataloader, desc='processing batches')):
            if x.shape[0] < 2:
                continue
            if x.shape[0] % 2 != 0:
                x = x[:-1]
                y = y[:-1]

            x = x.to(self.config.device)
            # Split the batch into even and odd elements
            x0 = x[0::2]
            x1 = x[1::2]
            y0 = y[0::2]
            y1 = y[1::2]

            indices0 = np.arange(i * batch_size, i * batch_size + x.shape[0], 2)
            indices1 = np.arange(i * batch_size + 1, i * batch_size + x.shape[0], 2)

            self.write_images(x0.cpu(), 'gt_x0', indices0)
            self.write_images(x1.cpu(), 'gt_x1', indices1)

            mixed = (x0 + x1)
            self.write_images(mixed.cpu()/2., 'mix', indices0, indices1)

            # Parameters to optimize (initialize with uniform noise)
            xs = [
                nn.Parameter(torch.Tensor(x0.shape).uniform_().to(self.config.device)),
                nn.Parameter(torch.Tensor(x1.shape).uniform_().to(self.config.device))
            ]

            step_lr = 0.00003
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100

            for sigma_idx, sigma in enumerate(tqdm.tqdm(sigmas, desc='annealed Langevin dynamics')):
                lambda_recon = 1./(sigma**2)
                labels = torch.ones(1, device=self.config.device) * sigma_idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                for step in range(n_steps_each):
                    noises = [torch.randn_like(xs[j]) * np.sqrt(step_size * 2) for j in range(2)]
                    
                    grads = [scorenet(xs[j], labels).detach() for j in range(2)]

                    recon_loss = (torch.norm(torch.flatten(xs[0] + xs[1] - mixed)) ** 2)
                    recon_grads = torch.autograd.grad(recon_loss, xs)

                    for j in range(2):
                        xs[j].data = xs[j].data + (step_size * grads[j]) + (-step_size * lambda_recon * recon_grads[j].detach()) + noises[j]

            for j in range(2):
                xs[j].data = torch.clamp(xs[j].data, 0, 1)

            # PSNR Measure and Permutation Correction
            x_final = [torch.zeros_like(x0.cpu()) for _ in range(2)]
            for idx in range(x0.shape[0]):
                best_psnr = -10000
                best_permutation = None
                
                # Check ground truth x0 vs xs[0] and x1 vs xs[1] or vice-versa
                # gt_images in this case are x0[idx] and x1[idx]
                gt = [x0[idx], x1[idx]]
                
                for p in permutations(range(2)):
                    curr_psnr = sum([psnr(xs[p[j]][idx], gt[j]) for j in range(2)])
                    if curr_psnr > best_psnr:
                        best_psnr = curr_psnr
                        best_permutation = p
                
                for j in range(2):
                    x_final[j][idx] = xs[best_permutation[j]][idx].detach().cpu()

            # Save results
            self.write_images(x_final[0], 'x0', indices0)
            self.write_images(x_final[1], 'x1', indices1)
            self.write_images((xs[0] + xs[1]).detach().cpu()/2., 'mixed_approx', indices0, indices1)

    def write_images(self, x, prefix, indices, indices2=None):
        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)
        # x is (B, C, H, W)
        x = x.numpy().transpose(0, 2, 3, 1)
        for k in range(x.shape[0]):
            assert x[k].min() >= 0 and x[k].max() <= 1, f"{x[k].min()}, {x[k].max()}"
            img = np.round(255 * x[k]).clip(0, 255).astype(np.uint8)
            # If it's single channel (MNIST), cv2.imwrite expect H,W or H,W,3
            if img.shape[2] == 1:
                img = img[:, :, 0]
            
            if indices2 is not None:
                filename = f"{prefix}_{indices[k]:05d}_{indices2[k]:05d}.png"
            else:
                filename = f"{prefix}_{indices[k]:05d}.png"
            
            # Add ncsn_mnist_ prefix to match previous naming convention if desired, 
            # but user didn't explicitly ask for it, just "mix_0_1.png".
            # Wait, "Mix should be named mix_0_1.png". 
            # I'll use the user's requested naming format.
            
            # The user said: "Mix should be named mix_0_1.png for the first mix element. 
            # Please make sure the names are padded, so that they are all the same length."
            # So mix_00000_00001.png
            
            path = os.path.join(self.args.image_folder, filename)
            cv2.imwrite(path, img)

