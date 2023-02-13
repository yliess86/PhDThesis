from collections import OrderedDict
from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torch.nn import (Linear, ReLU, Sigmoid, Sequential)
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import torch


sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

def init(worker_id: int) -> None:
    seed = worker_id + 42
    torch.manual_seed(seed)
    np.random.seed = seed
    random.seed(seed)

T = lambda x: to_tensor(x).float().flatten()
dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=12, worker_init_fn=init, drop_last=True)

class GaussianDistribution:
    def __init__(self, params: Tensor) -> None:
        self.mu, self.rho = params.chunk(2, dim=-1)
        self.std = torch.exp(0.5 * self.rho)
        self.var = torch.exp(self.rho)

    def sample(self) -> Tensor: return self.mu + self.std * torch.randn_like(self.mu)
    def kl(self) -> Tensor: return 0.5 * (self.mu.pow(2) + self.var - 1.0 - self.rho).sum(dim=1).mean(dim=0)

h_dim, z_dim = 256, 2
g_model = Sequential(Linear(z_dim, h_dim), ReLU(), Linear(h_dim,  28 * 28), Sigmoid()).cuda()
c_model = Sequential(Linear(28 * 28, h_dim), ReLU(), Linear(h_dim, 1)).cuda()
g_optim, c_optim = AdamW(g_model.parameters(), lr=1e-3), AdamW(c_model.parameters(), lr=1e-3)

history = []
with tqdm(range(50), desc="Epoch") as pbar:
    for epoch in pbar:
        g_model, c_model = g_model.train(), c_model.train()
        total_g_loss, total_c_loss = 0, 0
        for real, _ in loader:
            z = torch.rand((real.shape[0], z_dim)).float().cuda()
            real, fake = real.cuda(), g_model(z)
            g_loss = c_model(fake).mean(dim=0)
            c_loss = c_model(real).mean(dim=0) - c_model(fake.detach()).mean(dim=0)
            g_loss.backward(); g_optim.step(); g_optim.zero_grad(set_to_none=True)
            c_loss.backward(); c_optim.step(); c_optim.zero_grad(set_to_none=True)
            total_g_loss += g_loss.item() / len(loader)
            total_c_loss += c_loss.item() / len(loader)
        pbar.set_postfix(g_loss=f"{total_g_loss:.2e}", c_loss=f"{total_c_loss:.2e}")
        history.append((total_g_loss, total_c_loss))

g_model = g_model.cpu()
loader = DataLoader(dataset, batch_size=8 * 8, shuffle=False, num_workers=12, worker_init_fn=init, drop_last=True)

plt.figure(figsize=(8, 4))
plt.plot([h[0] for h in history], label="g_loss", c="dodgerblue")
plt.plot([h[1] for h in history], label="c_loss", c="darkorange")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.tight_layout()
# plt.savefig("./figures/core_gai_gan_history.svg")

with torch.inference_mode():
    z1, z2 = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    grid = np.hstack((z1.flatten()[:, None], z2.flatten()[:, None]))
    grid = torch.from_numpy(grid).float()
    x_ = make_grid(g_model(grid).reshape(-1, 1, 28, 28), nrow=16).permute(1, 2, 0)

plt.figure(figsize=(4, 4))
plt.imshow(x_)
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.tight_layout()
# plt.savefig("./figures/core_gai_gan_latent_sampling.svg")

plt.show()