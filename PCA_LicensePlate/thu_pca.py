import cv2
import numpy as np
import matplotlib.pyplot as plt
from pca import *


def run_pca(image, k):
    # Split từng channel
    channels = cv2.split(image)
    new_channels = []
    for ch in channels:
        ch = ch.astype(np.float32)

        X_pca, eigenvec, mean = perform_pca(ch, k)
        recon = reconstruct_image(X_pca, eigenvec, mean)

        recon = np.clip(recon, 0, 255).astype(np.uint8)
        new_channels.append(recon)
    # Ghép các channel lại
    new_image = cv2.merge(new_channels)
    return new_image

def visualize(image, new_image, k):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Original Image with shape {image.shape}")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title(f"PCA Image (k={k}) with shape {new_image.shape}")
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

image = cv2.imread(r"C:\0001.jpg")
k = 50
new_image = run_pca(image, k)
visualize(image, new_image, k)