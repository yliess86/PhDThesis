from torch.optim import AdamW
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.datasets as skld


np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

n = 250
dim = 32

x, l = skld.make_moons(n, noise=0.2)
x[:, 0] = 0.5 * (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = 0.5 * (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
x = torch.from_numpy(x).float().cuda()
l = torch.from_numpy(l).long().cuda()

f = nn.Sequential(nn.Linear(2, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 2)).cuda()
optim = AdamW(f.parameters(), lr=1e-3)

for _ in tqdm(range(1_000)):
    idxs = np.random.choice(range(len(x)), n, replace=False)
    x, l = x[idxs], l[idxs]

    y_ = f(x)
    L = F.cross_entropy(y_, l, reduction="mean")
    
    L.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

with torch.inference_mode():
   xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 1_000), np.linspace(-1.5, 1.5, 1_000))
   grid = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
   grid = torch.from_numpy(grid).float().cuda()
   zz = f(grid).argmax(dim=1).reshape(xx.shape)

x, l, zz = x.cpu(), l.cpu(), zz.cpu()

plt.figure(figsize=(8, 4))
plt.contourf(xx, yy, zz, cmap="Paired", alpha=0.2)
for k, c in zip([0, 1], ["dodgerblue", "darkorange"]):
    idxs = l == k
    plt.scatter(x[idxs, 0], x[idxs, 1], marker="+", color=c, label=k)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout()
plt.savefig("./figures/core_nn_classification.svg")