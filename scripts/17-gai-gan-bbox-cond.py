from __future__ import annotations

from collections import OrderedDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import FashionMNIST
from torch.nn import (BatchNorm2d, Flatten, Linear, Module, ReLU, Sigmoid, Sequential, Conv2d, MaxPool2d)
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
from torch.nn.functional import interpolate
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import torch
import torch.autograd as autograd


sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

def init(worker_id: int) -> None:
    seed = worker_id + 42
    torch.manual_seed(seed)
    np.random.seed = seed
    random.seed(seed)

def grad_penalty(c_model: Module, real: Tensor, fake: Tensor, mask: Tensor) -> Tensor:
    alpha = torch.rand((real.shape[0], 1, 1, 1), device=real.device, requires_grad=True)
    t = alpha * real - (1 - alpha) * fake
    mixed = c_model(t, mask)
    grads = autograd.grad(mixed, t, torch.ones_like(mixed), True, True, True)[0]
    return torch.mean((grads.view(grads.shape[0], -1).norm(2, dim=1) - 1) ** 2)

T = lambda x: to_tensor(x).float()
dataset = FashionMNIST("/tmp/fmnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=18, worker_init_fn=init, drop_last=True)

h_dim, z_dim = 256, 2
gen_latent_code = lambda B: torch.randn((B, z_dim), dtype=torch.float32, device="cuda")
condition = lambda x, c: torch.cat((x, c), dim=1)

class Generator(Sequential):
    def __init__(self) -> None:
        super().__init__(OrderedDict(
            expand=Sequential(
                Conv2d(z_dim,  64, 3, 1, 1, bias=False), BatchNorm2d( 64), ReLU(True),
                Conv2d(   64, 128, 3, 1, 1, bias=False), BatchNorm2d(128), ReLU(True),
                Conv2d(  128, 256, 5, 1, 2, bias=False), BatchNorm2d(256), ReLU(True),
            ),
            squeeeze=Sequential(
                Conv2d(256, 128, 5, 1, 2, bias=False), BatchNorm2d(128), ReLU(True),
                Conv2d(128,  64, 3, 1, 1, bias=False), BatchNorm2d( 64), ReLU(True),
                Conv2d( 64,   1, 3, 1, 1), Sigmoid(),
            ),
        ))

    def forward(self, z: Tensor, m: Tensor) -> Tensor:
        return super().forward(z[:, :, None, None] * m)

class Critic(Sequential):
    def __init__(self) -> None:
        super().__init__(OrderedDict(
            features=Sequential(
                Conv2d(1 + 1,  6, 5), ReLU(True), MaxPool2d(2),
                Conv2d(    6, 16, 5), ReLU(True), MaxPool2d(2),
            ),
            flatten=Flatten(),
            score=Sequential(
                Linear(256, 128), ReLU(True),
                Linear(128,  64), ReLU(True),
                Linear(64,    1)
            ),
        ))

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        return super().forward(condition(x, m))

g_model = Generator().cuda()
c_model = Critic().cuda()
g_optim = AdamW(g_model.parameters(), lr=1e-3)
c_optim = AdamW(c_model.parameters(), lr=1e-3)

names = ["c_loss_fake", "c_loss_real", "g_loss_fake"]
history = []
with tqdm(range(100), desc="Epoch") as pbar:
    for epoch in pbar:
        g_model, c_model = g_model.train(), c_model.train()
        losses = [0 for _ in range(len(names))]
        for real, _ in loader:
            real = real.cuda()
            mask = (real > 0.05).to(dtype=real.dtype)
            mask = interpolate(interpolate(mask, size=(9, 9), mode="bicubic"), size=(28, 28), mode="bicubic")
            mask = (real > 0.05).to(dtype=real.dtype)

            fake = g_model(gen_latent_code(real.shape[0]), mask)
            g_loss_fake = torch.relu(1.0 - c_model(fake, mask)).mean(dim=0)
            g_loss = g_loss_fake
            g_loss.backward(); g_optim.step(); g_optim.zero_grad(set_to_none=True)

            for _ in range(5):
                with torch.no_grad(): fake = g_model(gen_latent_code(real.shape[0]), mask)
                c_loss_fake = torch.relu(1.0 + c_model(fake, mask)).mean(dim=0)
                c_loss_real = torch.relu(1.0 - c_model(real, mask)).mean(dim=0)
                c_loss_gp = grad_penalty(c_model, real.detach(), fake.detach(), mask.detach())
                c_loss = c_loss_fake + c_loss_real + 10 * c_loss_gp
                c_loss.backward(); c_optim.step(); c_optim.zero_grad(set_to_none=True)
            
            for idx, name in enumerate(names): losses[idx] += eval(name).item() / len(loader)
        pbar.set_postfix(**dict(zip(names, [f"{loss:.2e}" for loss in losses])))
        history.append(losses)

plt.figure(figsize=(5 * 4, 2 * 4))

with torch.inference_mode():
    z1, z2 = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    grid = np.hstack((z1.flatten()[:, None], z2.flatten()[:, None]))
    grid = torch.from_numpy(grid).float().cuda()
    for i in range(5):
        m = mask[i:i + 1].repeat(16 * 16, 1, 1, 1)
        x_ = make_grid(g_model(grid, m), nrow=16).permute(1, 2, 0)
        m = m[0].repeat(3, 1, 1).permute(1, 2, 0)

        plt.subplot(2, 5, i + 1)
        plt.imshow(x_.cpu())
        plt.axis("off")
        plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(m.cpu())
        plt.axis("off")

plt.tight_layout()
plt.savefig("./figures/core_gai_gan_bbox_cond.svg")