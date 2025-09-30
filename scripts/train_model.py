#!/usr/bin/env python3
"""
Training script for multilingual Uzbek Whisper model
Supports distributed training, curriculum learning, and advanced optimization
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any

import torch
import torch.distributed as dist
from transformers import WhisperProcessor
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.architectures.whisper_multilingual import create_multilingual_whisper
from src.models.training.distributed_trainer import (
    DistributedMultilingualTrainer,
    MultilingualDataset
)
from src.utils.config import TrainingConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train multilingual Uzbek Whisper model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="uzbek_whisper_training",
        help="Experiment name for logging"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate training configuration"""
    required_keys = [
        'model',
        'training',
        'data'
    ]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False

    # Validate model config
    model_config = config['model']
    if 'base_model' not in model_config:
        logger.error("Missing 'base_model' in model config")
        return False

    # Validate training config
    training_config = config['training']
    required_training_keys = [
        'batch_size',
        'learning_rate',
        'num_epochs'
    ]

    for key in required_training_keys:
        if key not in training_config:
            logger.error(f"Missing required training config key: {key}")
            return False

    # Validate data config
    data_config = config['data']
    required_data_keys = [
        'train_manifest',
        'eval_manifest'
    ]

    for key in required_data_keys:
        if key not in data_config:
            logger.error(f"Missing required data config key: {key}")
            return False

        # Check if file exists
        if not os.path.exists(data_config[key]):
            logger.error(f"Data file does not exist: {data_config[key]}")
            return False

    return True

def setup_distributed():
    """Setup distributed training"""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group(backend='nccl')

        logger.info(f"Initialized distributed training on local rank {local_rank}")
        return True

    return False

def create_datasets(config: Dict[str, Any], processor: WhisperProcessor):
    """Create training and evaluation datasets"""
    data_config = config['data']

    # Training dataset
    train_dataset = MultilingualDataset(
        manifest_path=data_config['train_manifest'],
        processor=processor,
        max_audio_length=data_config.get('max_audio_length', 30.0),
        curriculum_filter=None  # Will be handled by curriculum scheduler
    )

    # Evaluation dataset
    eval_dataset = MultilingualDataset(
        manifest_path=data_config['eval_manifest'],
        processor=processor,
        max_audio_length=data_config.get('max_audio_length', 30.0),
        curriculum_filter=None
    )

    logger.info(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    return train_dataset, eval_dataset

def main():
    """Main training function"""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level)

    logger.info("Starting Uzbek Whisper training...")
    logger.info(f"Arguments: {args}")

    # Load and validate configuration
    config = load_config(args.config)
    if not validate_config(config):
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Setup distributed training if available
    is_distributed = setup_distributed()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize wandb
    if config.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=config['wandb'].get('project', 'uzbek-whisper'),
            name=args.experiment_name,
            config=config,
            tags=['whisper', 'uzbek', 'multilingual', 'code-switching']
        )

    try:
        # Create model
        logger.info("Creating multilingual Whisper model...")
        model_config = config['model']

        model = create_multilingual_whisper(
            base_model_name=model_config['base_model'],
            languages=model_config.get('languages', ['uz', 'ru', 'mixed']),
            adapter_dim=model_config.get('adapter_dim', 256),
            freeze_base=model_config.get('freeze_base_model', True)
        )

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Create processor
        processor = WhisperProcessor.from_pretrained(model_config['base_model'])

        # Create datasets
        train_dataset, eval_dataset = create_datasets(config, processor)

        # Create training configuration object
        training_config = TrainingConfig()

        # Update training config with values from YAML
        for key, value in config['training'].items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)

        # Set output directory
        training_config.output_dir = str(output_dir)
        training_config.experiment_name = args.experiment_name

        # Create trainer
        logger.info("Creating distributed trainer...")
        trainer = DistributedMultilingualTrainer(
            config=training_config,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processor=processor,
            resume_from_checkpoint=args.resume_from_checkpoint
        )

        # Start training
        logger.info("Starting training...")
        trainer.train(training_config.num_epochs)

        logger.info("Training completed successfully!")

        # Save final model
        final_model_path = output_dir / 'final_model'
        final_model_path.mkdir(exist_ok=True)

        # Get unwrapped model for saving
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_model_path)
        processor.save_pretrained(final_model_path)

        logger.info(f"Final model saved to {final_model_path}")

        # Log training summary
        training_summary = {
            'experiment_name': args.experiment_name,
            'config_path': args.config,
            'output_dir': str(output_dir),
            'final_model_path': str(final_model_path),
            'training_completed': True,
            'best_wer': trainer.best_wer,
            'total_steps': trainer.global_step,
            'final_epoch': trainer.current_epoch
        }

        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)

        if wandb.run:
            wandb.log(training_summary)
            wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        sys.exit(1)

    finally:
        # Cleanup distributed training
        if is_distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()