from torch.optim import AdamW
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

n = 50
dim = 16

X = 3 * torch.linspace(-np.pi, np.pi, 1_000, dtype=torch.float).reshape(-1, 1).cuda()
Y = torch.sin(X)

idxs = np.random.choice(range(len(X)), n, replace=False)
x = X.clone()[idxs]
y = Y.clone()[idxs]

f = nn.Sequential(nn.Linear(1, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1)).cuda()
optim = AdamW(f.parameters(), lr=1e-3)

for _ in tqdm(range(1_000)):
    idxs = np.random.choice(range(len(x)), n, replace=False)
    x, y = x[idxs], y[idxs]

    y_ = f(x)
    L = (1 / n) * ((y_ - y) ** 2).sum()
    
    L.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

with torch.inference_mode():
    y_ = f(x)
    Y_ = f(X)

X, Y, Y_ = X.cpu(), Y.cpu(), Y_.cpu()
x, y, y_ = x.cpu(), y.cpu(), y_.cpu()

plt.figure(figsize=(8, 4))
plt.plot(X, Y, label="$D$")
plt.plot(X, Y_, linestyle="--", label="$f_\\theta$")
plt.scatter(x, y, label="$\{(x_i, y_i)\} \in D$", marker="+")
plt.scatter(x, y_, label="$\hat{y}_i = f_\\theta(x_i)$", marker="+")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("./figures/core_nn_regression.svg")