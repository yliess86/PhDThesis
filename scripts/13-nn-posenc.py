import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


np.random.seed = 42
sns.set_style("white")
sns.set_context(context="paper", font_scale=1)

def pos_enc(seq_len: int, dim: int, n: int = 10_000) -> np.ndarray:
    P = np.zeros((seq_len, dim))
    for k in range(seq_len):
        for i in np.arange(dim // 2):
            denom = np.power(n, 2 * i / dim)
            P[k, 2 * i] = np.sin(k / denom)
            P[k, 2 * i + 1] = np.cos(k / denom)
    return P


plt.figure(figsize=(4 * 4, 4))
img = plt.imshow(pos_enc(100, 512), cmap="viridis")
plt.xlabel("$d_x$")
plt.ylabel("sequence length")
plt.colorbar(img, fraction=0.0094, pad=0.02)
plt.tight_layout()
plt.savefig("./figures/core_nn_posenc.svg")