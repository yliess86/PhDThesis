from __future__ import annotations

import numpy as np


def circshift(x: np.ndarray, shift: np.ndarray) -> np.ndarray:
    for i in range(shift.size):
        x = np.roll(x, shift[i], axis=i)
    return x


def prepare_psf(psf: np.ndarray, out_size: tuple) -> np.ndarray:
    psf_size = np.int32(psf.shape)
    out_size = psf_size if out_size is None else np.int32(out_size)
    new_psf = np.zeros(out_size, dtype=np.float32)
    new_psf[:psf_size[0], :psf_size[1]] = psf[:, :]
    return circshift(new_psf, -psf_size // 2)


def psf2otf(psf: np.ndarray, out_size: tuple) -> np.ndarray:
    return np.complex64(np.fft.fftn(prepare_psf(psf, out_size)))


def l0_smoothing(x: np.ndarray, kappa: float = 2.0, lamb: float = 1e-1) -> np.ndarray:
    S = x / 255.0
    H, W, C = x.shape
    
    size_2d = [H, W]
    fx = np.int32([[1, -1]])
    fy = np.int32([[1], [-1]])
    otffx = psf2otf(fx, size_2d)
    otffy = psf2otf(fy, size_2d)

    FI = np.complex64(np.zeros((H, W, C)))
    FI[..., 0] = np.fft.fft2(S[..., 0])
    FI[..., 1] = np.fft.fft2(S[..., 1])
    FI[..., 2] = np.fft.fft2(S[..., 2])

    MTF = np.abs(otffx) ** 2 + np.abs(otffy) ** 2
    MTF = np.tile(MTF[..., None], (1, 1, C))

    h    = np.float32  (np.zeros((H, W, C)))
    v    = np.float32  (np.zeros((H, W, C)))
    dxhp = np.float32  (np.zeros((H, W, C)))
    dyvp = np.float32  (np.zeros((H, W, C)))
    FS   = np.complex64(np.zeros((H, W, C)))

    beta_max, beta = 1e5, 2 * lamb
    while beta < beta_max:
        h[:, 0:W-1, :] = np.diff(S, 1, 1)
        h[:, W-1:W, :] = S[:, 0:1, :] - S[:, W-1:W, :]

        v[0:H-1, :, :] = np.diff(S, 1, 0)
        v[H-1:H, :, :] = S[0:1, :, :] - S[H-1:H, :, :]

        t = np.sum(h ** 2 + v ** 2, axis=-1) < lamb / beta
        t = np.tile(t[:, :, None], (1, 1, 3))

        h[t], v[t] = 0, 0

        dxhp[:, 0:1, :] = h[:, W-1:W, :] - h[:, 0:1, :]
        dxhp[:, 1:W, :] = -(np.diff(h, 1, 1))
        dyvp[0:1, :, :] = v[H-1:H, :, :] - v[0:1, :, :]
        dyvp[1:H, :, :] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        FS[..., 0] = np.fft.fft2(normin[..., 0])
        FS[..., 1] = np.fft.fft2(normin[..., 1])
        FS[..., 2] = np.fft.fft2(normin[..., 2])

        denorm = 1 + beta * MTF
        FS[...] = (FI + beta * FS) / denorm

        S[..., 0] = np.float32((np.fft.ifft2(FS[..., 0])).real)
        S[..., 1] = np.float32((np.fft.ifft2(FS[..., 1])).real)
        S[..., 2] = np.float32((np.fft.ifft2(FS[..., 2])).real)

        beta *= kappa

    return (np.clip(S, 0, 1) * 255.0).astype(np.uint8)

from PIL import Image
import shutil
import requests

urls = [
    "https://i.etsystatic.com/39530894/r/il/a067d2/4562558346/il_fullxfull.4562558346_g7qx.jpg",
    "https://m.media-amazon.com/images/I/81m6eNdRWwL.jpg",
    "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/7db99600-32a3-4d55-bf3b-fb620b858ce4/width=2048/00004.jpeg",
    "https://i.pinimg.com/736x/43/f8/4e/43f84e27080dd6fb83f8a76e89149d8a--anime-kimono-manga-anime.jpg"
]

srcs = []
imga = []
imgb = []
for url in urls:
    response = requests.get(url, stream=True)
    with open('/tmp/img.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    img = Image.open('/tmp/img.jpg').convert("RGB")
    w, h = img.size
    img = img.resize((int(w * 1024 / h), 1024), Image.Resampling.LANCZOS)
    img = np.array(img)[:768, :1024]
    srcs.append(img)
    imgb.append(l0_smoothing(img, lamb=0.1))
    imga.append(l0_smoothing(img, lamb=0.4))

Image.fromarray(np.vstack([
    np.hstack(srcs),
    np.hstack(imgb),
    np.hstack(imga),
])).save("figures/stablepaint_l0_smoothing.png")
