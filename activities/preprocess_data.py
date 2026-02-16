# activities/preprocess_data.py: 

# Input - Path of dataset dir.
# Input files are in .ply file format, dir contains 1045 files each of them containing the point cloud representation with 10K points. 
# Output - Path of pre processed dataset. -> .npz file for quick/fast retrival. 
# Processing - Each file -> pre processing(Downsample to 2048 + FPS + Normalization).
# Additional feature: Resume and saving logic : If some files are processed and the terminal crash, then you restart the preprocessing then processing should start from where it was left off. 

# activities/preprocess_data.py

from temporalio import activity
from shared import DatasetConfig


@activity.defn
async def preprocess_data(config: DatasetConfig) -> str:
    """
    Temporal activity for preprocessing raw .ply point clouds.
    Converts each file into a normalized + FPS sampled .npz file.

    Returns:
        str: Path to processed dataset directory
    """

    # Heavy imports MUST be inside activity
    import os
    import glob
    import numpy as np
    import open3d as o3d
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

    print(f"Found {len(ply_files)} .ply files.")
    print("Starting preprocessing...")

    for ply_path in tqdm(ply_files):

        fname = os.path.splitext(os.path.basename(ply_path))[0]
        out_path = os.path.join(output_dir, f"{fname}.npz")

        # Resume-safe: Skip already processed files
        if os.path.exists(out_path):
            continue

        # Load
        points = load_ply_numpy(ply_path, o3d)

        # Normalize
        if normalize:
            points, centroid, scale = normalize_pointcloud(points)
        else:
            centroid = np.zeros(3, dtype=np.float32)
            scale = np.float32(1.0)

        # FPS
        ids = farthest_point_sample_numpy(points, npoints)
        processed_points = points[ids]

        # Save
        np.savez(
            out_path,
            points=processed_points.astype(np.float32),
            centroid=centroid,
            scale=scale,
        )

    print("Preprocessing completed.")

    return output_dir
