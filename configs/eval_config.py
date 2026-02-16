from dataclasses import dataclass

@dataclass
class EvalConfig:
    model_path: str = "outputs/checkpoints/latest.pt"
    output_dir: str = "outputs/evaluation"
