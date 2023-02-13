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

h_dim, z_dim = 256, 2
model = Sequential(OrderedDict(
    encoder=Sequential(Linear(28 * 28, h_dim), ReLU(), Linear(h_dim,   z_dim)),
    decoder=Sequential(Linear(  z_dim, h_dim), ReLU(), Linear(h_dim, 28 * 28), Sigmoid()),
)).cuda()
optim = AdamW(model.parameters(), lr=1e-3)

history = []
with tqdm(range(50), desc="Epoch") as pbar:
    for _ in pbar:
        model = model.train()
        total_loss = 0
        for x, _ in loader:
            x = x.cuda()
            x_ = model(x)
            loss = binary_cross_entropy(x_, x)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            total_loss += loss.item() / len(loader)
        pbar.set_postfix(loss=f"{total_loss:.2e}")
        history.append(total_loss)

model = model.cpu()
loader = DataLoader(dataset, batch_size=8 * 8, shuffle=False, num_workers=12, worker_init_fn=init, drop_last=True)
x  : Tensor = next(iter(loader))[0]
x_ : Tensor = model(x)
x  = make_grid(x. reshape(-1, 1, 28, 28)).permute(1, 2, 0)
x_ = make_grid(x_.reshape(-1, 1, 28, 28)).permute(1, 2, 0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history, label="train", c="dodgerblue")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("binary cross entropy")

plt.subplot(1, 4, 3)
plt.imshow(x)
plt.axis("off")
plt.xlabel("$x$")

plt.subplot(1, 4, 4)
plt.imshow(x_)
plt.axis("off")
plt.xlabel("$D(E(x))$")

plt.tight_layout()
plt.savefig("./figures/core_gai_autoencoder_history.svg")

dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=8 * 8, shuffle=False, num_workers=12, worker_init_fn=init, drop_last=False)

with torch.inference_mode():
    pairs = [(z, l) for xs, ls in tqdm(loader, desc="Inference") for (z, l) in zip(model(xs).numpy(), ls.numpy())]
latents = np.array([z for z, _ in pairs])
labels = np.array([l for _, l in pairs])

plt.figure(figsize=(4, 4))
for l in set(labels):
    idxs = labels == l
    plt.scatter(latents[idxs, 0], latents[idxs, 1], label=l)
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("./figures/core_gai_autoencoder_latent.svg")

m1, M1 = 0.7 * np.min(latents[:, 0]), 0.7 * np.max(latents[:, 0])
m2, M2 = 0.7 * np.min(latents[:, 1]), 0.7 * np.max(latents[:, 1])
with torch.inference_mode():
    z1, z2 = np.meshgrid(np.linspace(m1, M1, 16), np.linspace(m2, M2, 16))
    grid = np.hstack((z1.flatten()[:, None], z2.flatten()[:, None]))
    grid = torch.from_numpy(grid).float()
    x_ = make_grid(model.decoder(grid).reshape(-1, 1, 28, 28), nrow=16).permute(1, 2, 0)

plt.figure(figsize=(2 * 4, 4))

plt.subplot(1, 2, 1)
for l in set(labels):
    idxs = labels == l
    plt.scatter(latents[idxs, 0], latents[idxs, 1], label=l)
plt.scatter(grid[:, 0], grid[:, 1], marker="+", label="sample", c="black")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.imshow(x_)
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")

plt.tight_layout()
plt.savefig("./figures/core_gai_autoencoder_latent_sampling.svg")