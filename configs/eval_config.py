# configs/eval_config.py

from dataclasses import dataclass


@dataclass
class EvalConfig:
    output_dir: str = "outputs/evaluation"
    
    # Metrics
    compute_chamfer: bool = True
    compute_latent_analysis: bool = True
    
    # Visualizations
    num_samples_visualize: int = 10
    save_ply_files: bool = True
    generate_loss_curves: bool = True
    generate_latent_viz: bool = True
    
    # Dataset splits
    eval_validation: bool = True
    eval_test: bool = True
    test_split: float = 0.1  # 10% for test, rest for train+val
    
    # t-SNE/PCA parameters
    tsne_perplexity: int = 30
    random_seed: int = 42