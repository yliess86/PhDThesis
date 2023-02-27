from __future__ import annotations

from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("white")
sns.set_context(context="paper", font_scale=1)


def dog(x: np.ndarray, size: tuple[int, int], sigma: float, k: float, gamma: float) -> np.ndarray:
    if x.shape[-1] == 3: x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(x, size, sigma) - gamma * cv2.GaussianBlur(x, size, sigma * k)


def xdog(x: np.ndarray, sigma: float, k: float, gamma: float, epsilon: float, phi: float) -> np.ndarray:
    x = dog(x, (0, 0), sigma, k, gamma) / 255
    return np.where(x < epsilon, 255, 255 * (1 + np.tanh(phi * x))).astype(np.uint8)


def sketch(x: np.ndarray, sigma: float, k: float, gamma: float, epsilon: float, phi: float, area_min: int) -> np.ndarray:
    x = xdog(x, sigma, k, gamma, epsilon, phi)
    x = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = [contour for contour in contours if cv2.contourArea(contour) > area_min]
    return 255 - cv2.drawContours(np.zeros_like(x), contours, -1, (255, 255, 255), -1)


img = Image.open("./figures/AMERICANED_UCLACOMM_095.jpg").convert("RGB").resize((384 * 2, 256 * 2))
img = np.array(img)[:256 * 2, :256 * 2]

plt.figure(figsize=(4 * 4, 2 * 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")

configs = [
    (0.3,  4.5, 0.95, -1.0, 10e9, 2),
    (0.6,  4.5, 0.95, -1.0, 10e9, 2),
    (0.4, 10.0, 0.95, -1.0, 10e9, 2),
    (0.4,  6.0, 0.85, -1.0, 10e9, 10),
]
for idx, conf in enumerate(configs):
    plt.subplot(2, 4, 2 * (idx // 2 + 1) + (idx + 1))
    plt.imshow(sketch(img, *conf), cmap="gray")
    plt.xlabel(f"$\sigma={conf[0]}$, $k={conf[1]}$, $\gamma={conf[2]}$, $\epsilon={conf[3]:.0f}$, $\phi={conf[4]:.0e}$")
    plt.xticks([], [])
    plt.yticks([], [])

plt.tight_layout()
plt.savefig("./figures/met_xdog.svg")