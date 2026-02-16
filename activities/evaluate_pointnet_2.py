# activities/evaluate_activity.py

from temporalio import activity
from configs.eval_config import EvalConfig
from configs.dataset_config import DatasetConfig
from configs.train_config import TrainConfig


@activity.defn
async def evaluate_activity(
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    model_path: str
) -> str:
    """
    Evaluate trained PointNet++ autoencoder.
    """
    
    import os
    import glob
    import random
    import numpy as np
    import torch
    from model.pointnet_2 import PointNetPPAutoEncoder
    from utils.metrics import chamfer_distance
    from utils.visualization import (
        plot_point_cloud_comparison,
        plot_multiple_comparisons,
        plot_loss_curves,
        plot_latent_space_pca,
        plot_latent_space_tsne,
        plot_chamfer_distribution,
        save_point_cloud_ply
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directories
    eval_dir = eval_config.output_dir
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "reconstructions"), exist_ok=True)
    
    # Load model
    model = PointNetPPAutoEncoder(
        latent_dim=train_config.latent_dim,
        num_points=train_config.num_points
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    activity.heartbeat("Model loaded")
    
    # Load dataset
    all_files = glob.glob(os.path.join(dataset_config.processed_data_path, '*.npz'))
    all_files.sort()
    
    random.seed(eval_config.random_seed)
    random.shuffle(all_files)
    
    # Split dataset: train/val/test
    n_total = len(all_files)
    n_test = int(eval_config.test_split * n_total)
    n_train = int((n_total - n_test) * train_config.train_split)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_total - n_test]
    test_files = all_files[n_total - n_test:]
    
    # Determine which splits to evaluate
    eval_splits = {}
    if eval_config.eval_validation:
        eval_splits['validation'] = val_files
    if eval_config.eval_test:
        eval_splits['test'] = test_files
    
    all_results = {}
    
    # Evaluate each split
    for split_name, split_files in eval_splits.items():
        
        chamfer_distances = []
        latent_vectors = []
        originals = []
        reconstructeds = []
        file_ids = []
        
        # Run inference
        for i, filepath in enumerate(split_files):
            data = np.load(filepath)
            points = data['points']
            
            points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                recon, latent = model(points_tensor)
            
            recon_np = recon.squeeze(0).cpu().numpy()
            latent_np = latent.squeeze(0).cpu().numpy()
            
            # Compute Chamfer distance
            cd = chamfer_distance(recon, points_tensor).item()
            chamfer_distances.append(cd)
            
            # Store for visualization
            if i < eval_config.num_samples_visualize:
                originals.append(points)
                reconstructeds.append(recon_np)
                file_ids.append(os.path.basename(filepath).replace('.npz', ''))
            
            latent_vectors.append(latent_np)
            
            if (i + 1) % 50 == 0:
                activity.heartbeat(f"{split_name}: {i+1}/{len(split_files)}")
        
        activity.heartbeat(f"{split_name} inference complete")
        
        # Convert to numpy arrays
        chamfer_distances = np.array(chamfer_distances)
        latent_vectors = np.array(latent_vectors)
        
        # Compute statistics
        stats = {
            'mean_cd': float(np.mean(chamfer_distances)),
            'median_cd': float(np.median(chamfer_distances)),
            'std_cd': float(np.std(chamfer_distances)),
            'min_cd': float(np.min(chamfer_distances)),
            'max_cd': float(np.max(chamfer_distances)),
            'p95_cd': float(np.percentile(chamfer_distances, 95)),
            'n_samples': len(chamfer_distances)
        }
        
        all_results[split_name] = stats
        
        # Save statistics
        stats_file = os.path.join(eval_dir, f'{split_name}_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Evaluation Results - {split_name.upper()} Set\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of samples: {stats['n_samples']}\n\n")
            f.write("Chamfer Distance Statistics:\n")
            f.write(f"  Mean:   {stats['mean_cd']:.6f}\n")
            f.write(f"  Median: {stats['median_cd']:.6f}\n")
            f.write(f"  Std:    {stats['std_cd']:.6f}\n")
            f.write(f"  Min:    {stats['min_cd']:.6f}\n")
            f.write(f"  Max:    {stats['max_cd']:.6f}\n")
            f.write(f"  95th percentile: {stats['p95_cd']:.6f}\n")
        
        # Visualization 1: Side-by-side comparisons
        if len(originals) > 0:
            try:
                viz_path = os.path.join(eval_dir, "visualizations", 
                                       f'{split_name}_comparisons.png')
                plot_multiple_comparisons(originals, reconstructeds, file_ids, viz_path)
            except Exception as e:
                print(f"Warning: Could not generate comparison plots: {e}")
        
        # Visualization 2: Chamfer distance distribution
        try:
            dist_path = os.path.join(eval_dir, "visualizations",
                                    f'{split_name}_chamfer_distribution.png')
            plot_chamfer_distribution(chamfer_distances, dist_path)
        except Exception as e:
            print(f"Warning: Could not generate chamfer distribution: {e}")
        
        # Visualization 3: Latent space analysis
        if eval_config.compute_latent_analysis and len(latent_vectors) >= 2:
            try:
                # PCA (needs at least 2 samples)
                pca_path = os.path.join(eval_dir, "visualizations",
                                       f'{split_name}_latent_pca.png')
                plot_latent_space_pca(latent_vectors, 
                                     np.arange(len(latent_vectors)), 
                                     pca_path)
            except Exception as e:
                print(f"Warning: Could not generate PCA plot: {e}")
            
            # t-SNE (needs at least 3 samples)
            if len(latent_vectors) >= 3:
                try:
                    tsne_path = os.path.join(eval_dir, "visualizations",
                                            f'{split_name}_latent_tsne.png')
                    plot_latent_space_tsne(latent_vectors,
                                          np.arange(len(latent_vectors)),
                                          tsne_path,
                                          perplexity=eval_config.tsne_perplexity)
                except Exception as e:
                    print(f"Warning: Could not generate t-SNE plot: {e}")
        
        if len(latent_vectors) < 2:
            print(f"Warning: {split_name} has only {len(latent_vectors)} samples. Skipping latent space visualization.")
        
        # Export reconstructed point clouds as .ply
        if eval_config.save_ply_files:
            for i in range(min(eval_config.num_samples_visualize, len(reconstructeds))):
                ply_path = os.path.join(eval_dir, "reconstructions",
                                       f'{split_name}_{file_ids[i]}_recon.ply')
                save_point_cloud_ply(reconstructeds[i], ply_path)
                
                # Also save original for comparison
                ply_orig_path = os.path.join(eval_dir, "reconstructions",
                                            f'{split_name}_{file_ids[i]}_orig.ply')
                save_point_cloud_ply(originals[i], ply_orig_path)
        
        activity.heartbeat(f"{split_name} visualizations complete")
    
    # Generate training loss curves
    if eval_config.generate_loss_curves:
        log_file = os.path.join(train_config.checkpoint_dir, "train_log.txt")
        if os.path.exists(log_file):
            try:
                loss_curve_path = os.path.join(eval_dir, "visualizations",
                                              "training_loss_curves.png")
                plot_loss_curves(log_file, loss_curve_path)
            except Exception as e:
                print(f"Warning: Could not generate loss curves: {e}")
    
    # Save summary report
    summary_path = os.path.join(eval_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_config.processed_data_path}\n\n")
        
        for split_name, stats in all_results.items():
            f.write(f"\n{split_name.upper()} SET RESULTS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Samples: {stats['n_samples']}\n")
            f.write(f"Mean Chamfer Distance: {stats['mean_cd']:.6f}\n")
            f.write(f"Median Chamfer Distance: {stats['median_cd']:.6f}\n")
            f.write(f"Std Dev: {stats['std_cd']:.6f}\n")
            f.write(f"Min: {stats['min_cd']:.6f}\n")
            f.write(f"Max: {stats['max_cd']:.6f}\n")
            f.write(f"95th Percentile: {stats['p95_cd']:.6f}\n")
    
    return eval_dir