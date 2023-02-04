import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

L = lambda x: x ** 2
dL = lambda x: 2 * x

x = np.linspace(-2, 2, 1_000)
y = L(x)

plt.figure(figsize=(10, 4))


for i, (lr, c) in enumerate(zip([2, 1, 0.1], ["dodgerblue", "darkorange", "crimson"])):
    plt.subplot(1, 3, i + 1)
    plt.plot(x, y, label="$y = x^2$", alpha=0.5)
    
    xs = [-1.0]
    for step in range(10):
        xs.append(xs[step] - lr * dL(xs[step]))
    ys = L(np.array(xs))
    plt.plot(xs, ys, c=c)
    plt.scatter(xs, ys, label=f"$\epsilon = {lr:.1f}$", marker="+", c=c)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.2, 1.2)
    
    plt.legend(loc="lower right")
    if i > 0: plt.yticks([], [])
    plt.xlabel("$x$")
    if i == 0: plt.ylabel("$y$")

plt.tight_layout()
plt.savefig("./figures/core_nn_sgd.svg")