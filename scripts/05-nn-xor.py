import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter([0, 1, 0], [0, 0, 1], 60, label="0", marker="_", c="dodgerblue")
plt.scatter([1], [1], 60, label="1", marker="+", c="darkorange")
x = np.linspace(-0.2, 1.2, 1_000)
plt.plot(x, 1.5 - x, label="hyperplan", c="black")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.legend(loc="center")

plt.subplot(1, 3, 2)
plt.scatter([0], [0], 60, label="0", marker="_", c="dodgerblue")
plt.scatter([1, 1, 0], [1, 0, 1], 60, label="1", marker="+", c="darkorange")
x = np.linspace(-0.2, 1.2, 1_000)
plt.plot(x, 0.5 - x, label="hyperplan", c="black")
plt.yticks([], [])
plt.xlabel("$x$")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.legend(loc="center")

plt.subplot(1, 3, 3)
plt.scatter([0, 1], [0, 1], 60, label="0", marker="_", c="dodgerblue")
plt.scatter([1, 0], [0, 1], 60, label="1", marker="+", c="darkorange")
x = np.linspace(-0.2, 1.2, 1_000)
plt.plot(x, 1.5 - x, label="hyperplan?", c="black", linestyle="--")
x = np.linspace(-0.2, 1.2, 1_000)
plt.plot(x, 0.5 - x, c="black", linestyle="--")
plt.yticks([], [])
plt.xlabel("$x$")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.legend(loc="center")

plt.tight_layout()
plt.savefig("./figures/core_nn_xor.svg")