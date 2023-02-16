from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torch.nn import (Linear, Module, ReLU, Sigmoid, Sequential)
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

def grad_penalty(c_model: Module, real: Tensor, fake: Tensor) -> Tensor:
    alpha = torch.rand((real.shape[0], 1), device=real.device, requires_grad=True)
    t = alpha * real - (1 - alpha) * fake
    mixed = c_model(t)
    grads = autograd.grad(mixed, t, torch.ones_like(mixed), True, True, True)[0]
    return torch.mean((grads.view(grads.shape[0], -1).norm(2, dim=1) - 1) ** 2)

T = lambda x: to_tensor(x).float().flatten()
dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=18, worker_init_fn=init, drop_last=True)

h_dim, z_dim = 256, 2
g_model = Sequential(
    Linear(z_dim, h_dim), ReLU(),
    Linear(h_dim, h_dim), ReLU(),
    Linear(h_dim,  28 * 28), Sigmoid(),
).cuda()
c_model = Sequential(
    Linear(28 * 28, h_dim), ReLU(),
    Linear(h_dim, h_dim), ReLU(),
    Linear(h_dim, 1),
).cuda()
g_optim, c_optim = AdamW(g_model.parameters(), lr=1e-3), AdamW(c_model.parameters(), lr=1e-3)

gen_latent_code = lambda B: torch.randn((B, z_dim), dtype=torch.float32, device="cuda")

names = ["c_loss_fake", "c_loss_real", "g_loss_fake"]
history = []
with tqdm(range(100), desc="Epoch") as pbar:
    for epoch in pbar:
        g_model, c_model = g_model.train(), c_model.train()
        losses = [0 for _ in range(len(names))]
        for real, _ in loader:
            real = real.cuda()
            
            fake = g_model(gen_latent_code(real.shape[0]))
            g_loss_fake = torch.relu(1.0 - c_model(fake)).mean(dim=0)
            g_loss = g_loss_fake
            g_loss.backward(); g_optim.step(); g_optim.zero_grad(set_to_none=True)

            for _ in range(5):
                with torch.no_grad(): fake = g_model(gen_latent_code(real.shape[0]))
                c_loss_fake = torch.relu(1.0 + c_model(fake)).mean(dim=0)
                c_loss_real = torch.relu(1.0 - c_model(real)).mean(dim=0)
                c_loss_gp = grad_penalty(c_model, real.detach(), fake.detach())
                c_loss = c_loss_fake + c_loss_real + 10 * c_loss_gp
                c_loss.backward(); c_optim.step(); c_optim.zero_grad(set_to_none=True)
            
            for idx, name in enumerate(names): losses[idx] += eval(name).item() / len(loader)
        pbar.set_postfix(**dict(zip(names, [f"{loss:.2e}" for loss in losses])))
        history.append(losses)

fnames = ["$ReLU(1.0 + D(G(z)))$", "$ReLU(1.0 - D(x))$", "$ReLU(1.0 - D(G(z)))$"]
plt.figure(figsize=(8, 4))
for idx, name in enumerate(names):
    c = "dodgerblue" if "c_" in name else "darkorange"
    linestyle = "--" if idx == 1 else "solid"
    plt.plot([h[idx] for h in history], label=name, c=c, linestyle=linestyle)
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("./figures/core_gai_gan_history.svg")

g_model = g_model.cpu()
with torch.inference_mode():
    z1, z2 = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    grid = np.hstack((z1.flatten()[:, None], z2.flatten()[:, None]))
    grid = torch.from_numpy(grid).float()
    x_ = make_grid(g_model(grid).reshape(-1, 1, 28, 28), nrow=16).permute(1, 2, 0)

plt.figure(figsize=(2 * 4, 4))

plt.subplot(1, 2, 1)
plt.scatter(grid[:, 0], grid[:, 1], marker="+", label="sample", c="black")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.imshow(x_)
plt.axis("off")
plt.tight_layout()

plt.savefig("./figures/core_gai_gan_latent_sampling.svg")