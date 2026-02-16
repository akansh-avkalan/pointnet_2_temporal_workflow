# utils/metrics.py

import torch


def chamfer_distance(pc1, pc2):
    """
    Chamfer Distance between two point clouds.
    
    Args:
        pc1: [B, N, 3]
        pc2: [B, M, 3]
    
    Returns:
        Scalar chamfer distance
    """
    dist = torch.cdist(pc1, pc2)
    
    min_dist_pc1, _ = torch.min(dist, dim=2)
    min_dist_pc2, _ = torch.min(dist, dim=1)
    
    loss = min_dist_pc1.mean() + min_dist_pc2.mean()
    return loss