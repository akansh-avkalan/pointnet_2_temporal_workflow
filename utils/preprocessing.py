# utils/preprocessing.py

# Helper Function for activities/preprocess_data.py

import numpy as np

def load_ply_numpy(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float32)
    return pts

def normalize_pointcloud(points):
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-9)
    return points, centroid, scale

def farthest_point_sample_numpy(points, n_samples):
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

