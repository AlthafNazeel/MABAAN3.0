"""
Visualization utilities for MABAAN.

Plotting functions for sample batches, predictions, and complexity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

from .metrics import compute_boundary_f1


def plot_sample_batch(batch):
    """Visualize a single batch: image, edge, mask, boundary, complexity."""
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax[0].imshow(batch['input'][0, 0].numpy(), cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(batch['input'][0, 3].numpy(), cmap='hot')
    ax[1].set_title('Edge Map')
    ax[2].imshow(batch['mask'][0, 0].numpy(), cmap='gray')
    ax[2].set_title('Mask')
    ax[3].imshow(batch['boundary'][0, 0].numpy(), cmap='hot')
    ax[3].set_title('Boundary')
    ax[4].bar(['Complexity'], [batch['complexity'][0].item()])
    ax[4].set_title('Shape Complexity')
    ax[4].set_ylim(0, 1)
    for a in ax[:4]:
        a.axis('off')
    plt.tight_layout()
    plt.show()


def plot_predictions(preds, tgts, complexities, title="Predictions", n_samples=4):
    """Plot prediction comparisons: ground truth, prediction, binary, error map."""
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    indices = np.random.choice(len(preds), n_samples, replace=False)
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(tgts[idx][0], cmap='gray')
        axes[row, 0].set_title('Ground Truth')
        axes[row, 1].imshow(preds[idx][0], cmap='gray')
        axes[row, 1].set_title('Prediction')
        axes[row, 2].imshow((preds[idx][0] > 0.5).astype(float), cmap='gray')
        axes[row, 2].set_title('Binary Pred')
        diff = np.abs(tgts[idx][0] - (preds[idx][0] > 0.5).astype(float))
        axes[row, 3].imshow(diff, cmap='hot')
        axes[row, 3].set_title(f'Error (c={complexities[idx]:.2f})')
    for a in axes.flat:
        a.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_complexity_analysis(preds, tgts, complexities, metrics_df):
    """Plot scatter and box plots of complexity vs metrics."""
    c_arr = np.array(complexities)
    bf1_arr = np.array([compute_boundary_f1(p[0], t[0]) for p, t in zip(preds, tgts)])
    dice_arr = metrics_df['dice'].values
    hd95_arr = metrics_df['hausdorff_95'].replace([np.inf], np.nan).values

    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(c_arr, dice_arr, alpha=0.3, s=10)
    axes[0].set_xlabel('Shape Complexity')
    axes[0].set_ylabel('Dice')
    axes[0].set_title('Complexity vs Dice')

    axes[1].scatter(c_arr, bf1_arr, alpha=0.3, s=10, c='orange')
    axes[1].set_xlabel('Shape Complexity')
    axes[1].set_ylabel('Boundary F1')
    axes[1].set_title('Complexity vs Boundary F1')

    valid_hd = ~np.isnan(hd95_arr)
    if valid_hd.sum() > 0:
        axes[2].scatter(c_arr[valid_hd], hd95_arr[valid_hd], alpha=0.3, s=10, c='red')
    axes[2].set_xlabel('Shape Complexity')
    axes[2].set_ylabel('HD95')
    axes[2].set_title('Complexity vs HD95')

    plt.suptitle('MABAAN: Shape Complexity vs Metrics', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Box plots by morphology bin
    bins_label = np.where(c_arr < 0.2, 'Circular', np.where(c_arr <= 0.5, 'Moderate', 'Irregular'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bin_order = ['Circular', 'Moderate', 'Irregular']

    for ax, metric, name in [(axes[0], dice_arr, 'Dice'), (axes[1], bf1_arr, 'Boundary F1')]:
        data = [metric[bins_label == b] for b in bin_order]
        ax.boxplot(data, labels=bin_order)
        ax.set_title(f'{name} by Morphology')
        ax.set_ylabel(name)

    if valid_hd.sum() > 0:
        data = [hd95_arr[valid_hd & (bins_label == b)] for b in bin_order]
        data = [d[~np.isnan(d)] for d in data]
        axes[2].boxplot(data, labels=bin_order)
        axes[2].set_title('HD95 by Morphology')
        axes[2].set_ylabel('HD95')

    plt.tight_layout()
    plt.show()
