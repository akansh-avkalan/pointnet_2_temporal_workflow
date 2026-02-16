# activities/preprocess_activity.py

from temporalio import activity
from configs.dataset_config import DatasetConfig


@activity.defn
async def preprocess_activity(config: DatasetConfig) -> str:
    """
    Preprocess raw .ply point clouds.
    Converts each file into a normalized + FPS sampled .npz file.
    """
    
    import os
    import glob
    import numpy as np
    from tqdm import tqdm
    from utils.preprocessing import (
        load_ply_numpy,
        normalize_pointcloud,
        farthest_point_sample_numpy,
    )
    
    input_dir = config.raw_data_path
    output_dir = config.processed_data_path
    npoints = config.npoints
    normalize = config.normalize
    
    os.makedirs(output_dir, exist_ok=True)
    
    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    
    if len(ply_files) == 0:
        raise ValueError(f"No .ply files found in {input_dir}")
    
    processed_files = set(
        os.path.splitext(os.path.basename(f))[0] 
        for f in glob.glob(os.path.join(output_dir, "*.npz"))
    )
    
    files_to_process = [
        f for f in ply_files 
        if os.path.splitext(os.path.basename(f))[0] not in processed_files
    ]
    
    if len(files_to_process) == 0:
        return output_dir
    
    for ply_path in tqdm(files_to_process):
        fname = os.path.splitext(os.path.basename(ply_path))[0]
        out_path = os.path.join(output_dir, f"{fname}.npz")
        
        if os.path.exists(out_path):
            continue
        
        points = load_ply_numpy(ply_path)
        
        if normalize:
            points, centroid, scale = normalize_pointcloud(points)
        else:
            centroid = np.zeros(3, dtype=np.float32)
            scale = np.float32(1.0)
        
        sampled_indices = farthest_point_sample_numpy(points, npoints)
        processed_points = points[sampled_indices]
        
        np.savez_compressed(
            out_path,
            points=processed_points.astype(np.float32),
            centroid=centroid,
            scale=scale,
        )
        
        activity.heartbeat()
    
    return output_dir