from torch import Tensor
from torch.nn import Conv2d
from torchvision.models import (vgg16, VGG16_Weights)
from torchvision.utils import make_grid
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

x = Image.open("./figures/AMERICANED_UCLACOMM_095.jpg").convert("RGB").resize((384, 256))
x = torch.from_numpy(np.array(x)[:256, :256, :] / 255.0).float().permute(2, 0, 1)
x = x.unsqueeze(0)

model = vgg16(weights=VGG16_Weights.DEFAULT)
extract = lambda x: make_grid(x.reshape(-1, 1, x.shape[-2], x.shape[-1]), nrow=int(np.sqrt(x.shape[1]))).permute(1, 2, 0)

plt.figure(figsize=(2 * 4, 4))

plt.subplot(1, 2, 1)
plt.imshow(x[0].permute(1, 2, 0))
plt.axis("off")

i = 0
for module in model.features:
    x: Tensor = module(x)
    if isinstance(module, Conv2d):
        m, M = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
        feature_map = extract((x - m) / (M - m))
        
        if i == 0: plt.subplot(2, 4, 3)
        if i == 1: plt.subplot(2, 4, 7)
        if i == 2: plt.subplot(2, 4, 4)
        if i == 3: plt.subplot(2, 4, 8)
        plt.imshow(feature_map)
        plt.axis("off")
        i += 1
        if i >= 4: break
        
plt.tight_layout()
plt.savefig("./figures/core_nn_vgg_activations.svg")