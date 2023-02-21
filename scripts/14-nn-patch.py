from torchvision.utils import make_grid
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


PATCH = 32

np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

x = Image.open("./figures/AMERICANED_UCLACOMM_095.jpg").convert("RGB").resize((384, 256))
x = torch.from_numpy(np.array(x)[:256, :256, :] / 255.0).float().permute(2, 0, 1)
x = x.unsqueeze(0)
print(x.shape)

p = x.unfold(1, 3, 3).unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
p = p.reshape(-1, (256 // PATCH) * (256 // PATCH), 3, PATCH, PATCH)
print(p.shape)

transform = lambda x: (255 * x.permute(1, 2, 0)).byte().numpy()

plt.figure(figsize=(2 * 4, 4))

plt.subplot(1, 2, 1)
plt.imshow(transform(x[0]))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(transform(make_grid(p[0], nrow=256 // PATCH, padding=PATCH // 4, pad_value=1))[PATCH // 4:-PATCH // 4, PATCH // 4:-PATCH // 4])
plt.axis("off")

plt.tight_layout()
plt.savefig("./figures/core_nn_patches.svg")