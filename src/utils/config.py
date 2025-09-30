"""
Configuration management utilities
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration class"""
    # Model settings
    base_model: str = "openai/whisper-base"
    languages: list = field(default_factory=lambda: ["uz", "ru", "mixed"])
    adapter_dim: int = 256
    freeze_base_model: bool = True

    # Training hyperparameters
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    num_epochs: int = 10
    max_steps: Optional[int] = None

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: str = "fp16"
    gradient_clip_val: float = 1.0

    # Logging and evaluation
    logging_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    eval_steps: int = 500

    # Early stopping
    early_stopping_patience: Optional[int] = 5

    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "uzbek_whisper"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "uzbek-whisper"

    # Data loader
    dataloader_num_workers: int = 4

    # Curriculum learning
    curriculum_learning: Optional[Dict] = None

@dataclass
class ServingConfig:
    """Serving/API configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # Model settings
    model_path: str = "openai/whisper-tiny"  # Use pre-trained model for demo
    processor_path: Optional[str] = None
    device: str = "auto"

    # API limits
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_batch_size: int = 10
    request_timeout: int = 30

    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_max_connections: int = 100
    cache_ttl: int = 3600

    # Performance
    enable_streaming: bool = True
    batch_inference: bool = True

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_config_from_env() -> ServingConfig:
    """Get serving config from environment variables"""
    return ServingConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        model_path=os.getenv("MODEL_PATH", "./models/latest"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        max_file_size=int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024))),
    )