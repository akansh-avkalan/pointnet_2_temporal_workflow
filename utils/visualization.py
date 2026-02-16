# utils/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_point_cloud_comparison(original, reconstructed, title="", save_path=None):
    """Plot original and reconstructed point clouds side by side."""
    fig = plt.figure(figsize=(14, 6))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], 
                c='blue', s=1, alpha=0.6)
    ax1.set_title("Original")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Reconstructed
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                c='red', s=1, alpha=0.6)
    ax2.set_title("Reconstructed")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multiple_comparisons(originals, reconstructeds, ids, save_path, grid_size=(2, 5)):
    """Plot multiple point cloud comparisons in a grid."""
    rows, cols = grid_size
    n_samples = min(len(originals), rows * cols)
    
    fig = plt.figure(figsize=(cols * 6, rows * 6))
    
    for i in range(n_samples):
        # Original
        ax1 = fig.add_subplot(rows, cols * 2, i * 2 + 1, projection='3d')
        ax1.scatter(originals[i][:, 0], originals[i][:, 1], originals[i][:, 2],
                   c='blue', s=0.5, alpha=0.6)
        ax1.set_title(f"Original {ids[i]}")
        ax1.axis('off')
        
        # Reconstructed
        ax2 = fig.add_subplot(rows, cols * 2, i * 2 + 2, projection='3d')
        ax2.scatter(reconstructeds[i][:, 0], reconstructeds[i][:, 1], reconstructeds[i][:, 2],
                   c='red', s=0.5, alpha=0.6)
        ax2.set_title(f"Recon {ids[i]}")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(log_file, save_path):
    """Parse training log and plot loss curves."""
    epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch [' in line and 'Train Loss:' in line:
                try:
                    parts = line.split('|')
                    epoch_part = parts[0].split('[')[1].split('/')[0]
                    epoch = int(epoch_part)
                    epochs.append(epoch)
                    
                    train_part = parts[1].split(':')[1].strip()
                    train_losses.append(float(train_part))
                    
                    if 'Val Loss:' in line:
                        val_part = parts[2].split(':')[1].strip()
                        val_losses.append(float(val_part))
                        val_epochs.append(epoch)
                except (IndexError, ValueError) as e:
                    continue
    
    if len(epochs) == 0:
        print("Warning: No training data found in log file")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2, alpha=0.7)
    
    if len(val_losses) > 0 and len(val_epochs) > 0:
        ax.plot(val_epochs, val_losses, label='Validation Loss', 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space_pca(latent_vectors, labels, save_path):
    """Plot PCA of latent space."""
    n_samples = len(latent_vectors)
    
    # PCA requires at least 2 samples
    if n_samples < 2:
        print(f"Warning: Not enough samples ({n_samples}) for PCA. Skipping.")
        return
    
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                        c=labels, cmap='viridis', s=20, alpha=0.6)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('Latent Space - PCA Projection', fontsize=14)
    
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space_tsne(latent_vectors, labels, save_path, perplexity=30):
    """Plot t-SNE of latent space."""
    n_samples = len(latent_vectors)
    
    # Adjust perplexity if needed (must be less than n_samples)
    if n_samples <= perplexity:
        perplexity = max(2, n_samples - 1)
    
    # t-SNE requires at least 3 samples
    if n_samples < 3:
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE. Skipping.")
        return
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                        c=labels, cmap='viridis', s=20, alpha=0.6)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'Latent Space - t-SNE Projection (perplexity={perplexity})', fontsize=14)
    
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_chamfer_distribution(chamfer_distances, save_path):
    """Plot distribution of Chamfer distances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(chamfer_distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(chamfer_distances), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(chamfer_distances):.6f}')
    ax1.axvline(np.median(chamfer_distances), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(chamfer_distances):.6f}')
    ax1.set_xlabel('Chamfer Distance', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Reconstruction Errors', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(chamfer_distances, vert=True)
    ax2.set_ylabel('Chamfer Distance', fontsize=12)
    ax2.set_title('Box Plot of Reconstruction Errors', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_point_cloud_ply(points, filepath):
    """Save point cloud as .ply file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")