import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

SQRT2PI = np.sqrt(2 * np.pi)

a = 0.08
m1, s1, a1 = -0.5, 0.25, 0.01
m2, s2, a2 = 0.8, 0.15, 0.002
m3, s3, a3 = 0.6, 0.05, 0.0005

square = lambda x: x.pow(2)
gauss = lambda x, m, s: (1 / (s * SQRT2PI)) * (-0.5 * ((x - m) ** 2) / (s ** 2)).exp()
L = lambda x: square(a * x) - (a1 * gauss(x, m1, s1) + a2 * gauss(x, m2, s2) + a3 * gauss(x, m3, s3))

x = torch.linspace(-2, 2, 1_000)
y = L(x)


b1, b2 = 0.95, 0.999


def sgd(x: torch.Tensor, lr: float) -> torch.Tensor:
    L(x).backward()
    x = x.detach() - lr * x.grad
    x.grad = None
    x.requires_grad = True
    return x


moment_v = torch.tensor(0.0)
def moment(x: torch.Tensor, lr: float) -> torch.Tensor:
    global moment_v
    L(x).backward()
    moment_v = b1 * moment_v + x.grad
    x = x.detach() - lr * moment_v
    x.grad = None
    x.requires_grad = True
    return x


adagrad_r = torch.tensor(0.0)
def adagrad(x: torch.Tensor, lr: float) -> torch.Tensor:
    global adagrad_r
    L(x).backward()
    adagrad_r = adagrad_r + x.grad.pow(2)
    x = x.detach() - lr / (1e-8 + adagrad_r.sqrt()) * x.grad
    x.grad = None
    x.requires_grad = True
    return x


rmsprop_r = torch.tensor(0.0)
def rmsprop(x: torch.Tensor, lr: float) -> torch.Tensor:
    global rmsprop_r
    L(x).backward()
    rmsprop_r = b2 * rmsprop_r + (1.0 - b2) * x.grad.pow(2)
    x = x.detach() - lr / (1e-8 + rmsprop_r.sqrt()) * x.grad
    x.grad = None
    x.requires_grad = True
    return x


adam_m, adam_v = torch.tensor(0.0), torch.tensor(0.0)
def adam(x: torch.Tensor, lr: float) -> torch.Tensor:
    global adam_m, adam_v
    L(x).backward()
    adam_m = b1 * adam_m + (1.0 - b1) * x.grad
    adam_v = b2 * adam_v + (1.0 - b2) * x.grad.pow(2)
    x = x.detach() - lr * adam_m / (1e-8 + adam_v.sqrt()) * x.grad
    x.grad = None
    x.requires_grad = True
    return x


plt.figure(figsize=(12, 6))
plt.plot(x, y)

lr = 0.5
for i, (method, c) in enumerate(zip([sgd, moment, adagrad, rmsprop, adam], ["dodgerblue", "darkorange", "crimson", "limegreen", "darkmagenta"])):
   plt.subplot(2, 3, i + 1)
   plt.plot(x, y, label="$y = x^2$", alpha=0.5)
  
   xs = [torch.tensor(1.0, requires_grad=True)]
   for step in range(50): xs.append(method(xs[step], lr))
   ys = [L(x_).item() for x_ in xs]
   xs = [x_.item() for x_ in xs]
   alpha = [0.2 * (1 - i / len(xs)) + (i / len(xs)) for i in range(len(xs))]
   plt.plot(xs, ys, c=c, alpha=0.2)
   plt.scatter(xs[::-1], ys[::-1], label=f"{method.__name__} $\epsilon = {lr:.1f}$", marker="+", c=c, alpha=alpha[::-1])
   plt.xlim(-1.2, 1.2)
   plt.ylim(y.min().item() * (1 + 0.1), y.max().item() * (1 + 0.01))
  
   plt.legend(loc="upper left")
   if i % 3 != 0: plt.yticks([], [])
   plt.xlabel("$x$")
   if i % 3 == 0: plt.ylabel("$y$")

plt.savefig("./figures/core_nn_sgd_moments.svg")