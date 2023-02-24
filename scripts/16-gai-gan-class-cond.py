from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torch.nn import (Linear, Module, ReLU, Sigmoid, Sequential)
from torch.nn.functional import one_hot
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
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

def grad_penalty(c_model: Module, real: Tensor, fake: Tensor, cond: Tensor) -> Tensor:
    alpha = torch.rand((real.shape[0], 1), device=real.device, requires_grad=True)
    t = alpha * real - (1 - alpha) * fake
    mixed = c_model(condition(t, cond))
    grads = autograd.grad(mixed, t, torch.ones_like(mixed), True, True, True)[0]
    return torch.mean((grads.view(grads.shape[0], -1).norm(2, dim=1) - 1) ** 2)

T = lambda x: to_tensor(x).float().flatten()
dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=18, worker_init_fn=init, drop_last=True)

h_dim, z_dim = 256, 2
g_model = Sequential(
    Linear(z_dim + 10, h_dim), ReLU(),
    Linear(h_dim, h_dim), ReLU(),
    Linear(h_dim,  28 * 28), Sigmoid(),
).cuda()
c_model = Sequential(
    Linear(28 * 28 + 10, h_dim), ReLU(),
    Linear(h_dim, h_dim), ReLU(),
    Linear(h_dim, 1),
).cuda()
g_optim = AdamW(g_model.parameters(), lr=1e-3)
c_optim = AdamW(c_model.parameters(), lr=1e-3)

gen_latent_code = lambda B: torch.randn((B, z_dim), dtype=torch.float32, device="cuda")
condition = lambda x, c: torch.cat((x, c), dim=-1)

names = ["c_loss_fake", "c_loss_real", "g_loss_fake"]
history = []
with tqdm(range(50), desc="Epoch") as pbar:
    for epoch in pbar:
        g_model, c_model = g_model.train(), c_model.train()
        losses = [0 for _ in range(len(names))]
        for real, label in loader:
            real, label = real.cuda(), label.cuda()
            cond = one_hot(label, num_classes=10).requires_grad_(False)

            fake = g_model(condition(gen_latent_code(real.shape[0]), cond))
            g_loss_fake = torch.relu(1.0 - c_model(condition(fake, cond))).mean(dim=0)
            g_loss = g_loss_fake
            g_loss.backward(); g_optim.step(); g_optim.zero_grad(set_to_none=True)

            for _ in range(5):
                with torch.no_grad(): fake = g_model(condition(gen_latent_code(real.shape[0]), cond))
                c_loss_fake = torch.relu(1.0 + c_model(condition(fake, cond))).mean(dim=0)
                c_loss_real = torch.relu(1.0 - c_model(condition(real, cond))).mean(dim=0)
                c_loss_gp = grad_penalty(c_model, real.detach(), fake.detach(), cond.detach())
                c_loss = c_loss_fake + c_loss_real + 10 * c_loss_gp
                c_loss.backward(); c_optim.step(); c_optim.zero_grad(set_to_none=True)
            
            for idx, name in enumerate(names): losses[idx] += eval(name).item() / len(loader)
        pbar.set_postfix(**dict(zip(names, [f"{loss:.2e}" for loss in losses])))
        history.append(losses)

g_model = g_model.cpu()

plt.figure(figsize=(5 * 4, 2 * 4))

with torch.inference_mode():
    z1, z2 = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    grid = np.hstack((z1.flatten()[:, None], z2.flatten()[:, None]))
    grid = torch.from_numpy(grid).float()
    for label in range(10):
        cond = one_hot(torch.tensor([label] * (16 * 16), dtype=torch.long), num_classes=10)
        x_ = make_grid(g_model(condition(grid, cond)).reshape(-1, 1, 28, 28), nrow=16).permute(1, 2, 0)

        plt.subplot(2, 5, label + 1)
        plt.imshow(x_)
        plt.axis("off")

plt.tight_layout()
plt.savefig("./figures/core_gai_gan_class_cond.svg")