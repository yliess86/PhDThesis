from __future__ import annotations

from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import (Subset, DataLoader)
from torchvision.datasets.mnist import MNIST
from torch.nn import (LayerNorm, Linear, Module, ReLU, Sequential)
from torch import Tensor
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import torch


ATTENTION: Tensor = None


sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

def init(worker_id: int) -> None:
    seed = worker_id + 42
    torch.manual_seed(seed)
    np.random.seed = seed
    random.seed(seed)


class MultHeadAttention(Module):
    def __init__(self, emb_dim: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.emb_dim, self.n_heads, self.head_dim = emb_dim, n_heads, head_dim
        self.qkv = Linear(emb_dim, 3 * n_heads * head_dim, bias=False)
        self.out = Linear(n_heads * head_dim, emb_dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        qkv = map(lambda x: x.reshape(B, S, self.n_heads, self.head_dim), qkv)
        qkv = map(lambda x: x.permute(0, 2, 1, 3), qkv)
        q, k, v = qkv
        a = torch.softmax(self.scale * (q @ k.transpose(-2, -1)), dim=-1); global ATTENTION; ATTENTION = a
        z = (a @ v).reshape(B, S, self.n_heads * self.head_dim)
        return self.out(z)
    

class FeadForward(Sequential):
    def __init__(self, emb_dim: int, h_dim: int) -> None:
        super().__init__(
            Linear(emb_dim, h_dim, bias=False), ReLU(inplace=True),
            Linear(h_dim, emb_dim, bias=False),
        )


class TransformerBlock(Module):
    def __init__(self, emb_dim: int, n_heads: int, head_dim: int, h_dim: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm(emb_dim, eps=1e-5)
        self.mha = MultHeadAttention(emb_dim, n_heads, head_dim)
        self.norm2 = LayerNorm(emb_dim, eps=1e-5)
        self.ff = FeadForward(emb_dim, h_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mha(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x
    

class PositionalEncoder(Module):
    def __init__(self, emb_dim: int, max_len: int = 10_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        denom = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32) * (-np.log(10_000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]


emb_dim, n_heads, patch_size, patch_stride = 128, 8, 4, 4
class VisionTransformer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = Sequential(Linear(patch_size * patch_size, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
        self.pe = PositionalEncoder(emb_dim)
        self.block = TransformerBlock(emb_dim, n_heads, emb_dim // n_heads, emb_dim)
        self.head = Linear(emb_dim, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        x = x.reshape(x.size(0), -1, patch_size * patch_size)
        x = self.emb(x.reshape(-1, patch_size * patch_size)).reshape(x.size(0), -1, emb_dim)
        x = self.pe(x)
        x = self.block(x)
        x = x.mean(dim=1)
        return self.head(x)


T = lambda x: to_tensor(x).float()
dataset = MNIST("/tmp/mnist", train=True,  transform=T, download=True)
testset = MNIST("/tmp/mnist", train=False, transform=T, download=True)

train_idxs = np.random.choice(range(len(dataset)), size=int(np.floor(0.8 * len(dataset))), replace=False)
valid_idxs = np.array([idx for idx in range(len(dataset)) if idx not in train_idxs])
trainset, validset = Subset(dataset, train_idxs), Subset(dataset, valid_idxs)

loader = lambda d, shuffle: DataLoader(d, batch_size=128, shuffle=shuffle, num_workers=12, worker_init_fn=init)
trainloader, validloader, testloader = loader(trainset, True),  loader(validset, False), loader(testset, False)

vit = VisionTransformer().cuda()
optim = AdamW(vit.parameters(), lr=1e-2)

history = {"Train": [], "Valid": [], "Test": []}
for epoch in tqdm(range(10)):
    for split in ["Train", "Valid"]:
        dataloader = {"Train": trainloader, "Valid": validloader}[split]
        vit = vit.train() if split == "Train" else vit.eval()
        with torch.inference_mode(mode=split == "Valid"):
            total_loss, acc = 0, 0
            for x, l in dataloader:
                x, l = x.cuda(), l.cuda()
                
                y_ = vit(x)
                loss = cross_entropy(y_, l)
                n_correct = (y_.argmax(dim=-1) == l).sum()

                if split == "Train":
                    loss.backward()
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                total_loss += loss.item() / len(dataloader)
                acc += n_correct.item() / len(dataloader.dataset)
            history[split].append((total_loss, acc))

vit = vit.eval()
with torch.inference_mode():
    total_loss, acc = 0, 0
    for x, l in testloader:
        x, l = x.cuda(), l.cuda()
        y_ = vit(x)
        loss = cross_entropy(y_, l)
        n_correct = (y_.argmax(dim=-1) == l).sum()
        total_loss += loss.item() / len(testloader)
        acc += n_correct.item() / len(testloader.dataset)
    history["Test"].append((total_loss, acc))

print(f"{history['Test'][-1][0]:.2e} | {history['Test'][-1][1] * 100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([loss for loss, _ in history["Train"]], label="train", c="dodgerblue")
plt.plot([loss for loss, _ in history["Valid"]], label="valid", c="darkorange")
plt.plot([loss for loss, _ in history["Test"] * len(history["Train"])], label="test", linestyle="--", c="limegreen")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("cross entropy")

plt.subplot(1, 2, 2)
plt.plot([acc * 100 for _, acc in history["Train"]], label="train", c="dodgerblue")
plt.plot([acc * 100 for _, acc in history["Valid"]], label="valid", c="darkorange")
plt.plot([acc * 100 for _, acc in history["Test"] * len(history["Train"])], label="test", linestyle="--", c="limegreen")
plt.legend(loc="lower right")
plt.xlabel("epochs")
plt.ylabel("accuracy (%)")

plt.tight_layout()
plt.savefig("./figures/core_nn_vit_history.svg")


ATTENTION = ATTENTION.cpu()
plt.figure(figsize=(ATTENTION.shape[1] * 4, 4))

for i in range(ATTENTION.shape[1]):
    plt.subplot(1, ATTENTION.shape[1], i + 1)
    plt.title(f"Head ${i + 1}$")
    plt.imshow(ATTENTION[0][i], origin="lower", vmin=0, cmap="viridis")
    plt.xticks(range(0, ATTENTION.shape[-1], 5))
    plt.yticks(range(0, ATTENTION.shape[-2], 5))

plt.tight_layout()
plt.savefig("./figures/core_nn_vit_mha.svg")