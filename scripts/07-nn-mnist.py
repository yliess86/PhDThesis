from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import (Subset, DataLoader)
from torchvision.datasets.mnist import MNIST
from torch.nn import (Linear, ReLU, Sequential)
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

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
testset = MNIST("/tmp/mnist", train=False, transform=T, download=True)

train_idxs = np.random.choice(range(len(dataset)), size=int(np.floor(0.8 * len(dataset))), replace=False)
valid_idxs = np.array([idx for idx in range(len(dataset)) if idx not in train_idxs])
trainset, validset = Subset(dataset, train_idxs), Subset(dataset, valid_idxs)

loader = lambda d, shuffle: DataLoader(d, batch_size=1_024, shuffle=shuffle, num_workers=12, worker_init_fn=init)
trainloader, validloader, testloader = loader(trainset, True),  loader(validset, False), loader(testset, False)

h_dim = 128
model = Sequential(Linear(28 * 28, h_dim), ReLU(), Linear(h_dim, h_dim), ReLU(), Linear(h_dim, 10))
model[-1].bias.data.fill_(0.0)
model[-1].weight.data.fill_(0.01)
optim = AdamW(model.parameters(), lr=1e-2)

history = {"Train": [], "Valid": [], "Test": []}
for epoch in tqdm(range(10)):
    for split in ["Train", "Valid"]:
        dataloader = {"Train": trainloader, "Valid": validloader}[split]
        model = model.train() if split == "Train" else model.eval()
        with torch.inference_mode(mode=split == "Valid"):
            total_loss, acc = 0, 0
            for x, l in dataloader:
                y_ = model(x)
                loss = cross_entropy(y_, l)
                n_correct = (y_.argmax(dim=-1) == l).sum()

                if split == "Train":
                    loss.backward()
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                total_loss += loss.item() / len(dataloader)
                acc += n_correct.item() / len(dataloader.dataset)
            history[split].append((total_loss, acc))

model = model.eval()
with torch.inference_mode():
    total_loss, acc = 0, 0
    for x, l in testloader:
        y_ = model(x)
        loss = cross_entropy(y_, l)
        n_correct = (y_.argmax(dim=-1) == l).sum()
        total_loss += loss.item() / len(testloader)
        acc += n_correct.item() / len(testloader.dataset)
    history["Test"].append((total_loss, acc))

plt.figure(figsize=(12, 4))
for i in range(27):
    plt.subplot(3, 9, i + 1)
    plt.imshow(testset[i][0].reshape(28, 28).numpy(), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.savefig("./figures/core_nn_mnist.svg")

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

plt.savefig("./figures/core_nn_mnist_history.svg")