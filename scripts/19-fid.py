from __future__ import annotations

from PIL import Image
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms.functional as F

FID = FrechetInceptionDistance()
@torch.no_grad()
def compute_fid(real: Tensor, fake: Tensor) -> float:
    FID.reset()
    FID.update((real * 255.0).unsqueeze(0).repeat(4, 1, 1, 1).byte(), real=True)
    FID.update((fake * 255.0).unsqueeze(0).repeat(4, 1, 1, 1).byte(), real=False)
    return FID.compute().item()

sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

img = Image.open("./figures/AMERICANED_UCLACOMM_095.jpg").convert("RGB").resize((384 * 2, 256 * 2))
img = np.array(img)[:256 * 2, :256 * 2]

ts = torch.linspace(0, 1, 4, dtype=torch.float32)
plt.figure(figsize=(2 * 4, 4))

# =============================================================
real = (torch.from_numpy(img) / 255.0).float().permute(2, 0, 1)
noise = torch.rand_like(real)
fakes = [noise * t + real * (1.0 - t) for t in ts]
fids = [compute_fid(real, fake) for fake in fakes]

plt.subplot(2, 2, 1)
plt.imshow((make_grid(torch.stack(fakes)).permute(1, 2, 0) * 255).byte().numpy())
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(ts, fids)
plt.scatter(ts, fids, marker="+")
plt.ylabel("FID")
plt.xlabel("noise factor $t$")

# =============================================================
real = (torch.from_numpy(img) / 255.0).float().permute(2, 0, 1)
noise = real.clone()
for _ in range(100): noise = F.gaussian_blur(noise, (9, 9))
fakes = [noise * t + real * (1.0 - t) for t in ts]
fids = [compute_fid(real, fake) for fake in fakes]

plt.subplot(2, 2, 2)
plt.imshow((make_grid(torch.stack(fakes)).permute(1, 2, 0) * 255).byte().numpy())
plt.axis("off")

plt.subplot(2, 2, 4)
plt.plot(ts, fids)
plt.scatter(ts, fids, marker="+")
plt.ylabel("FID")
plt.xlabel("blur factor $t$")

# =============================================================
plt.tight_layout()
plt.savefig("./figures/met_fid.svg")