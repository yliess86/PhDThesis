import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

x = np.linspace(-4, 4, 1_000)
y = np.linspace(-3.8, 3.8, 1_000)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, 1 / (1 + np.exp(-x)), label="$y = \sigma(x)$", c="dodgerblue")
plt.plot(x, [0] * len(x), c="black", linestyle="--", alpha=0.2)
plt.plot(x, [1] * len(x), c="black", linestyle="--", alpha=0.2)
plt.plot([0] * len(y), y, c="black", linestyle="--", alpha=0.2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.ylim(-4, 4)
plt.legend(loc="lower right")

plt.subplot(1, 3, 2)
plt.plot(x, np.tanh(x), label="$y = tanh(x)$", c="dodgerblue")
plt.plot(x, [-1] * len(x), c="black", linestyle="--", alpha=0.2)
plt.plot(x, [1] * len(x), c="black", linestyle="--", alpha=0.2)
plt.plot([0] * len(y), y, c="black", linestyle="--", alpha=0.2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.ylim(-4, 4)
plt.legend(loc="lower right")

plt.subplot(1, 3, 3)
plt.plot(x, np.maximum(x, 0), label="$y = ReLU(x)$", c="dodgerblue")
plt.plot(x, [0] * len(x), c="black", linestyle="--", alpha=0.2)
plt.plot([0] * len(y), y, c="black", linestyle="--", alpha=0.2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.ylim(-4, 4)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("./figures/core_nn_activations.svg")