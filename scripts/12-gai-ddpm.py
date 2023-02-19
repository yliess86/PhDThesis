from __future__ import annotations

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torch.nn import (BatchNorm2d, Conv2d, GELU, Module, Sequential)
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
from torch import Tensor
from tqdm import tqdm
from typing import Callable
from functools import partial
from torch.nn.functional import mse_loss

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

BetaSchedule = Callable[[int], Tensor]
def linear_beta_schedule(T: int, beta_0: float = 1e-4, beta_T: float = 0.02) -> Tensor:
    return (beta_T - beta_0) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta_0

class DDPM(Module):
    def __init__(self, T: int, schedule: BetaSchedule) -> None:
        super().__init__()
        self.T = T
        
        betas = schedule(T)
        alphas = 1.0 - betas
        alphas_cp = torch.cumprod(alphas, axis=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cp", alphas_cp)
        self.register_buffer("sqrt_betas", torch.sqrt(betas))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cp", torch.sqrt(alphas_cp))
        self.register_buffer("sqrt_one_minus_alphas_cp", torch.sqrt(1.0 - alphas_cp))
        self.register_buffer("beta_over_sqrt_one_minus_alphas_cp", betas / torch.sqrt(1.0 - alphas_cp))
    
    def _extract(self, x: Tensor, t: Tensor) -> Tensor:
        return x.gather(-1, t).reshape(-1, 1, 1, 1)
    
    def forward_diffusion(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        extract = partial(self._extract, t=t)
        return extract(self.sqrt_alphas_cp) * x_0 + extract(self.sqrt_one_minus_alphas_cp) * noise

    def reverse_diffusion(self, eps: Tensor, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        extract = partial(self._extract, t=t)
        eps = eps * extract(self.beta_over_sqrt_one_minus_alphas_cp)
        return extract(self.sqrt_recip_alphas) * (x_t - eps) + extract(self.sqrt_betas) * noise


h_dim = 256
CBG = lambda ic, oc: Sequential(Conv2d(ic, oc, 7, 1, 3), BatchNorm2d(oc), GELU())
class NoiseModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential(
            CBG(1, h_dim // 4), GELU(),
            CBG(h_dim // 4, h_dim // 2), GELU(),
            CBG(h_dim // 2, h_dim), GELU(),
            CBG(h_dim, h_dim // 2), GELU(),
            CBG(h_dim // 2, h_dim // 4), GELU(),
            Conv2d(h_dim // 4, 1, 3, 1, 1),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.model(x_t)

T = lambda x: 2 * to_tensor(x).float() - 1
dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=18, worker_init_fn=init, drop_last=True)

ddpm = DDPM(1_000, lambda T: linear_beta_schedule(T)).cuda()
model = NoiseModel().cuda()
optim = AdamW(model.parameters(), lr=1e-4)

history = []
with tqdm(range(100), desc="Epoch") as pbar:
    for epoch in pbar:
        model = model.train()
        total_loss = 0.0
        for x_0, _ in loader:
            x_0 = x_0.cuda()
            
            B = x_0.shape[0]
            with torch.no_grad():
                t = torch.randint(0, ddpm.T + 1, size=(B, ), dtype=torch.long, device=x_0.device)
                noise = torch.randn_like(x_0)
                x_t = ddpm.forward_diffusion(x_0, t, noise)
            loss = mse_loss(model(x_t, t[:, None, None, None] / ddpm.T), noise)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            total_loss += loss.item() / len(loader)
        pbar.set_postfix(loss=f"{total_loss:.2e}")
        history.append(total_loss)

plt.figure(figsize=(8, 4))
plt.plot(history, label="loss", c="dodgerblue")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("mse loss")
plt.savefig("./figures/core_gai_ddpm_history.svg")

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, worker_init_fn=init, drop_last=True)
with torch.inference_mode():
    x_t = next(iter(loader))[0].cuda()
    imgs = torch.empty((ddpm.T, 1, 28, 28), dtype=torch.float32)
    imgs[0] = x_t
    for t_idx in tqdm(range(1, ddpm.T)):
        t = torch.tensor([t_idx] * x_t.shape[0], dtype=torch.long, device="cuda")
        x_t = ddpm.forward_diffusion(x_t, t, torch.randn_like(x_t))
        imgs[t_idx] = x_t.cpu()
    img = (0.5 + 0.5 * imgs.clip(-1, 1)).reshape(-1, 1, 28, 28)
    img = make_grid(img, nrow=2 * int(np.sqrt(ddpm.T))).permute(1, 2, 0)

plt.figure(figsize=(2 * 4, 4))
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.savefig("./figures/core_gai_ddpm_forward.svg")

with torch.inference_mode():
    x_t = torch.randn((1, 1, 28, 28), dtype=torch.float32, device="cuda")
    imgs = torch.empty((ddpm.T, 1, 28, 28), dtype=torch.float32)
    imgs[0] = x_t
    for t_idx in tqdm(range(ddpm.T, 0, -1)):
        t = torch.tensor([t_idx] * x_t.shape[0], dtype=torch.long, device="cuda")
        noise = torch.randn_like(x_t) if t_idx > 1 else 0
        eps = model(x_t, t[:, None, None, None] / ddpm.T)
        x_t = ddpm.reverse_diffusion(eps, x_t, t, noise)
        imgs[-t_idx] = x_t.cpu()
    img = (0.5 + 0.5 * imgs.clip(-1, 1)).reshape(-1, 1, 28, 28)
    img = make_grid(img, nrow=2 * int(np.sqrt(ddpm.T))).permute(1, 2, 0)

plt.figure(figsize=(2 * 4, 4))
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.savefig("./figures/core_gai_ddpm_latent_sampling.svg")

with torch.inference_mode():
    x_t = torch.randn((40, 1, 28, 28), dtype=torch.float32, device="cuda")
    for t_idx in tqdm(range(ddpm.T, 0, -1)):
        t = torch.tensor([t_idx] * x_t.shape[0], dtype=torch.long, device="cuda")
        noise = torch.randn_like(x_t) if t_idx > 1 else 0
        eps = model(x_t, t[:, None, None, None] / ddpm.T)
        x_t = ddpm.reverse_diffusion(eps, x_t, t, noise)
    img = (0.5 + 0.5 * x_t.clip(-1, 1)).cpu().reshape(-1, 1, 28, 28)
    img = make_grid(img, nrow=2 * int(np.sqrt(32))).permute(1, 2, 0)

plt.figure(figsize=(2 * 4, 4))
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.savefig("./figures/core_gai_ddpm_latent_samples.svg")