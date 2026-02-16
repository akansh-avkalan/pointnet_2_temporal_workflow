# utils/preprocessing.py

import numpy as np
import open3d as o3d


def load_ply_numpy(path):
    """Load a .ply file and return points as numpy array."""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float32)
    return pts


def normalize_pointcloud(points):
    """Normalize point cloud to unit sphere centered at origin."""
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / (scale + 1e-9)
    
    return points_normalized, centroid.astype(np.float32), np.float32(scale)


def farthest_point_sample_numpy(points, n_samples):
    """Farthest Point Sampling for downsampling point cloud."""
    N = points.shape[0]
    
    if n_samples >= N:
        return np.arange(N, dtype=np.int64)
    
    centroids = np.zeros((n_samples,), dtype=np.int64)
    distances = np.ones((N,), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    
    return centroids