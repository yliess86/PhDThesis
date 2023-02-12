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
import umap


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

h_dim, z_dim = 128, 32
model = Sequential(OrderedDict(
    encoder=Sequential(Linear(28 * 28, h_dim), ReLU(), Linear(h_dim, 2 * z_dim), ReLU()),
    decoder=Sequential(Linear(  z_dim, h_dim), ReLU(), Linear(h_dim,   28 * 28), Sigmoid()),
))
optim = AdamW(model.parameters(), lr=1e-3)

history = []
with tqdm(range(10), desc="Epoch") as pbar:
    for _ in pbar:
        model = model.train()
        total_loss, total_reco, total_kl = 0, 0, 0
        for x, _ in loader:
            p  = GaussianDistribution(model.encoder(x))
            x_ = model.decoder(p.sample())
            loss_reco = binary_cross_entropy(x_, x)
            loss_kl = p.kl()
            loss = loss_reco + 1e-4 * loss_kl
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            total_loss += loss.item() / len(loader)
            total_reco += loss_reco.item() / len(loader)
            total_kl += loss_kl.item() / len(loader)
        pbar.set_postfix(loss=f"{total_loss:.2e}", reco=f"{total_reco:.2e}", kl=f"{total_kl:.2e}")
        history.append((total_loss, total_reco, total_kl))

loader = DataLoader(dataset, batch_size=8 * 8, shuffle=False, num_workers=12, worker_init_fn=init, drop_last=True)
x  : Tensor = next(iter(loader))[0]
x_ : Tensor = model.decoder(GaussianDistribution(model.encoder(x)).sample())
x  = make_grid(x. reshape(-1, 1, 28, 28)).permute(1, 2, 0)
x_ = make_grid(x_.reshape(-1, 1, 28, 28)).permute(1, 2, 0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([h[0] for h in history], label="loss", c="dodgerblue")
plt.plot([h[1] for h in history], label="reco", c="darkorange")
plt.plot([h[2] for h in history], label="kld",  c="crimson")
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
plt.show()
# plt.savefig("./figures/core_gai_vae_history.svg")

dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
loader = DataLoader(dataset, batch_size=8 * 8, shuffle=False, num_workers=12, worker_init_fn=init, drop_last=False)

with torch.inference_mode():
    pairs = [
        (x, l)
        for xs, ls in tqdm(loader, desc="Inference")
        for (x, l) in zip(GaussianDistribution(model.encoder(xs)).mu.numpy(), ls.numpy())
    ]
inputs = np.array([x for x, _ in pairs])
labels = np.array([l for _, l in pairs])

reduction = umap.UMAP(n_components=2)
latents = reduction.fit_transform(inputs)

plt.figure(figsize=(4, 4))
for l in set(labels):
    idxs = labels == l
    plt.scatter(inputs[idxs, 0], inputs[idxs, 1], label=l)
plt.xlabel("$umap_1$")
plt.ylabel("$umap_2$")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
# plt.savefig("./figures/core_gai_vae_latent.svg")