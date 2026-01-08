import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm

from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ['CifarRunner']

class CifarRunner():
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

        # Grab some samples from the test set
        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar'), train=False, transform=transforms.ToTensor(), download=True)
        batch_size = 100
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, (x, y) in enumerate(tqdm.tqdm(dataloader, desc='processing batches')):
            x = x.to(self.config.device)
            # Split the batch into even and odd elements
            x0 = x[0::2]
            x1 = x[1::2]
            
            indices0 = np.arange(i * batch_size, i * batch_size + x.shape[0], 2)
            indices1 = np.arange(i * batch_size + 1, i * batch_size + x.shape[0], 2)

            self.write_images(x0.cpu(), 'xgt', indices0)
            self.write_images(x1.cpu(), 'ygt', indices1)

            mixed = (x0 + x1)
            self.write_images(mixed.cpu()/2., 'mix', indices0, indices1)

            x0_param = nn.Parameter(torch.Tensor(x0.shape).uniform_()).to(self.config.device)
            x1_param = nn.Parameter(torch.Tensor(x1.shape).uniform_()).to(self.config.device)

            x0, x1 = x0_param, x1_param

            step_lr=0.00003

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100

            for idx, sigma in enumerate(tqdm.tqdm(sigmas, desc='annealed Langevin dynamics')):
                lambda_recon = 1./sigma**2
                labels = torch.ones(1, device=x0.device) * idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                
                print('sigma = {}'.format(sigma))
                for step in range(n_steps_each):
                    recon = ((x0 + x1 - mixed)**2).view(-1,3*32*32).sum(1).mean()

                    noise_x = torch.randn_like(x0) * np.sqrt(step_size * 2)
                    noise_y = torch.randn_like(x1) * np.sqrt(step_size * 2)

                    grad_x0 = scorenet(x0, labels).detach()
                    grad_x1 = scorenet(x1, labels).detach()

                    norm0 = np.linalg.norm(grad_x0.view(-1,3*32*32).cpu().numpy(),axis=1).mean()
                    norm1 = np.linalg.norm(grad_x1.view(-1,3*32*32).cpu().numpy(),axis=1).mean()

                    x0 += step_size * (grad_x0 - lambda_recon * (x0 + x1 - mixed)) + noise_x
                    x1 += step_size * (grad_x1 - lambda_recon * (x0 + x1 - mixed)) + noise_y

                print(' recon: {}, |norm1|: {}, |norm2|: {}'.format(recon,norm0,norm1))

                # Write x0 and x1
                self.write_images(x0.detach().cpu(), 'x', indices0)
                self.write_images(x1.detach().cpu(), 'y', indices1)

    def write_images(self, x, prefix, indices, indices2=None):
        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)
        x = x.numpy().transpose(0, 2, 3, 1)
        for k in range(x.shape[0]):
            img = np.round(255 * x[k]).clip(0, 255).astype(np.uint8)[:, :, ::-1]
            if indices2 is not None:
                filename = f"{prefix}_{indices[k]:05d}_{indices2[k]:05d}.png"
            else:
                filename = f"{prefix}_{indices[k]:05d}.png"
            cv2.imwrite(os.path.join(self.args.image_folder, filename), img)
