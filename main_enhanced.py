#!/usr/bin/env python3
"""
Enhanced Two-Step Action Recognition Main Script
Optimized for maximum accuracy with advanced training techniques
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from datetime import datetime
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from config_enhanced import config
from trainer_enhanced import create_enhanced_trainer
from models_enhanced import create_enhanced_model, calculate_model_complexity
from utils_enhanced import create_visualizations, save_experiment_results

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Two-Step A2D Action Recognition Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=config.num_epochs,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.learning_rate,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=config.weight_decay,
                       help='Weight decay')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default=config.backbone,
                       choices=['efficientnet_b3', 'efficientnet_b4', 'resnet50', 'resnet101'],
                       help='Backbone network')
    parser.add_argument('--shared-backbone', action='store_true', default=config.shared_backbone,
                       help='Use shared backbone for both tasks')
    parser.add_argument('--num-frames', type=int, default=config.num_frames,
                       help='Number of frames per video')
    parser.add_argument('--frame-size', type=int, nargs=2, default=config.frame_size,
                       help='Frame size (height width)')
    
    # Data parameters
    parser.add_argument('--temporal-stride', type=int, default=config.temporal_stride,
                       help='Temporal stride for frame sampling')
    parser.add_argument('--val-split', type=float, default=config.val_split,
                       help='Validation split ratio')
    
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default=config.optimizer,
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default=config.scheduler,
                       choices=['step', 'cosine', 'cosine_warmup', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--action-loss', type=str, default=config.action_loss_type,
                       choices=['cross_entropy', 'label_smoothing', 'focal'],
                       help='Action loss function')
    parser.add_argument('--actor-loss', type=str, default=config.actor_loss_type,
                       choices=['cross_entropy', 'label_smoothing', 'focal'],
                       help='Actor loss function')
    
    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=config.dropout_rate,
                       help='Dropout rate')
    parser.add_argument('--mixup-alpha', type=float, default=config.mixup_alpha,
                       help='Mixup alpha parameter')
    parser.add_argument('--cutmix-prob', type=float, default=config.cutmix_prob,
                       help='CutMix probability')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=config.device,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=config.num_workers,
                       help='Number of data loader workers')
    parser.add_argument('--mixed-precision', action='store_true', default=config.mixed_precision,
                       help='Use mixed precision training')
    
    # Logging parameters
    parser.add_argument('--experiment-name', type=str, default=config.experiment_name,
                       help='Experiment name')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--use-tensorboard', action='store_true', default=config.use_tensorboard,
                       help='Use TensorBoard logging')
    
    # Evaluation parameters
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation on trained model')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model for evaluation')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run test on trained model')
    
    # Visualization parameters
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save training plots and visualizations')
    parser.add_argument('--plot-interval', type=int, default=5,
                       help='Interval for saving plots during training')
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update config with command line arguments"""
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.backbone = args.backbone
    config.shared_backbone = args.shared_backbone
    config.num_frames = args.num_frames
    config.frame_size = tuple(args.frame_size)
    config.temporal_stride = args.temporal_stride
    config.val_split = args.val_split
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    config.action_loss_type = args.action_loss
    config.actor_loss_type = args.actor_loss
    config.dropout_rate = args.dropout
    config.mixup_alpha = args.mixup_alpha
    config.cutmix_prob = args.cutmix_prob
    config.device = args.device
    config.num_workers = args.num_workers
    config.mixed_precision = args.mixed_precision
    config.experiment_name = args.experiment_name
    config.use_wandb = args.use_wandb
    config.use_tensorboard = args.use_tensorboard


def create_experiment_dir(experiment_name: str) -> str:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    
    return experiment_dir


def save_experiment_config(experiment_dir: str):
    """Save experiment configuration"""
    config_dict = {
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'backbone': config.backbone,
        'shared_backbone': config.shared_backbone,
        'num_frames': config.num_frames,
        'frame_size': config.frame_size,
        'temporal_stride': config.temporal_stride,
        'val_split': config.val_split,
        'optimizer': config.optimizer,
        'scheduler': config.scheduler,
        'action_loss_type': config.action_loss_type,
        'actor_loss_type': config.actor_loss_type,
        'dropout_rate': config.dropout_rate,
        'mixup_alpha': config.mixup_alpha,
        'cutmix_prob': config.cutmix_prob,
        'device': config.device,
        'num_workers': config.num_workers,
        'mixed_precision': config.mixed_precision,
        'num_action_classes': len(config.action_classes),
        'num_actor_classes': len(config.actor_classes),
        'action_classes': config.action_classes,
        'actor_classes': config.actor_classes,
        'action_to_actors': config.action_to_actors,
        'actor_to_actions': config.actor_to_actions,
        'valid_labels': config.valid_labels,
        'hidden_dim': config.hidden_dim,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers
    }
    
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"📄 Configuration saved to: {config_path}")


def print_experiment_info(experiment_dir: str):
    """Print experiment information"""
    print("\n" + "="*80)
    print("🎯 ENHANCED TWO-STEP ACTION RECOGNITION EXPERIMENT")
    print("="*80)
    print(f"📁 Experiment Directory: {experiment_dir}")
    print(f"📱 Device: {config.device}")
    print(f"🏗️ Backbone: {config.backbone}")
    print(f"🔗 Shared Backbone: {config.shared_backbone}")
    print(f"🎬 Number of Frames: {config.num_frames}")
    print(f"📐 Frame Size: {config.frame_size}")
    print(f"📦 Batch Size: {config.batch_size}")
    print(f"📈 Learning Rate: {config.learning_rate}")
    print(f"⚙️ Optimizer: {config.optimizer}")
    print(f"📊 Scheduler: {config.scheduler}")
    print(f"🎯 Action Loss: {config.action_loss_type}")
    print(f"🎭 Actor Loss: {config.actor_loss_type}")
    print(f"📊 Action Classes: {len(config.action_classes)}")
    print(f"🎭 Actor Classes: {len(config.actor_classes)}")
    print(f"✅ Valid Labels: {len(config.valid_labels)}")
    print(f"🔧 Mixed Precision: {config.mixed_precision}")
    print(f"🧠 Hidden Dimension: {config.hidden_dim}")
    print(f"👥 Number of Heads: {config.num_heads}")
    print(f"📚 Number of Layers: {config.num_layers}")
    print("="*80)
    
    # Print action-actor mappings
    print("\n🎯 Action-Actor Mappings:")
    for action, actors in config.action_to_actors.items():
        print(f"  {action}: {', '.join(actors)}")
    print("="*80)


def validate_dataset():
    """Validate dataset structure"""
    print("Validating dataset...")
    
    # Check if dataset files exist
    required_files = [
        config.csv_file,
        config.videos_dir,
        config.annotations_dir
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Required file/directory not found: {file_path}")
            return False
    
    # Check if videos directory has files
    video_files = [f for f in os.listdir(config.videos_dir) if f.endswith('.mp4')]
    if len(video_files) == 0:
        print(f"❌ ERROR: No video files found in {config.videos_dir}")
        return False
    
    print(f"Dataset validation passed! Found {len(video_files)} video files")
    return True


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(42)
    
    # Update config with arguments
    update_config_from_args(args)
    
    # Validate dataset
    if not validate_dataset():
        print("❌ Dataset validation failed. Exiting...")
        return
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(args.experiment_name)
    
    # Save configuration
    save_experiment_config(experiment_dir)
    
    # Print experiment info
    print_experiment_info(experiment_dir)
    
    # Check if evaluation only
    if args.eval_only:
        if not args.model_path:
            print("❌ ERROR: Model path required for evaluation only mode")
            return
        
        print(f"🧪 Running evaluation on model: {args.model_path}")
        # TODO: Implement evaluation only mode
        return
    
    # Check if test only
    if args.test_only:
        if not args.model_path:
            print("❌ ERROR: Model path required for test only mode")
            return
        
        print(f"🧪 Running test on model: {args.model_path}")
        print("🏗️ Creating trainer...")
        trainer = create_enhanced_trainer(experiment_dir)
        print(f"📂 Loading checkpoint from: {args.model_path}")
        success, start_epoch = trainer.load_checkpoint(args.model_path)
        if not success:
            print("❌ Failed to load checkpoint.")
            return
        test_metrics = trainer.test()
        print(f"🧪 Final test accuracy: {test_metrics['combined_acc']:.2f}%")
        print(f"📊 Final test F1 score: {test_metrics['action_f1']:.4f}")
        
        # Create visualizations
        if args.save_plots:
            print("📊 Creating visualizations...")
            create_visualizations(trainer, experiment_dir)
        
        # Save final results
        save_experiment_results(trainer, experiment_dir, test_metrics)
        
        print(f"📁 Results saved to: {experiment_dir}")
        
        return
    
    # Create trainer
    print("🏗️ Creating enhanced trainer...")
    trainer = create_enhanced_trainer(experiment_dir)
    
    # Load checkpoint if specified
    if args.resume:
        print(f"📂 Loading checkpoint from: {args.resume}")
        success, start_epoch = trainer.load_checkpoint(args.resume)
        if not success:
            print("❌ Failed to load checkpoint. Starting fresh training.")
            start_epoch = 0
    
    # Calculate model complexity
    dummy_input = torch.randn(1, config.num_frames, 3, *config.frame_size)
    complexity = calculate_model_complexity(trainer.model, dummy_input.shape)
    print(f"🔢 Model Parameters: {complexity['total_parameters']:,}")
    print(f"⚡ Model FLOPs: {complexity['flops']:,}")
    
    # Start enhanced training
    print("🚀 Starting enhanced training...")
    try:
        trainer.train()
        
        print("✅ Enhanced training completed successfully!")
        print(f"🏆 Best validation accuracy: {trainer.best_val_accuracy:.2f}%")
        
        # Final test on test set
        print("🧪 Running final test on test set...")
        test_metrics = trainer.test()
        print(f"🧪 Final test accuracy: {test_metrics['combined_acc']:.2f}%")
        print(f"📊 Final test F1 score: {test_metrics['action_f1']:.4f}")
        
        # Create visualizations
        if args.save_plots:
            print("📊 Creating visualizations...")
            create_visualizations(trainer, experiment_dir)
        
        # Save final results
        save_experiment_results(trainer, experiment_dir, test_metrics)
        
        print(f"📁 Results saved to: {experiment_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_final=True)
        print("💾 Emergency checkpoint saved")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        trainer.save_checkpoint(trainer.current_epoch, is_final=True)
        print("💾 Emergency checkpoint saved before exit")
        sys.exit(1)

    # Save final results
    results = {
        'experiment_dir': experiment_dir,
        'best_val_accuracy': trainer.best_val_accuracy,
        'best_action_accuracy': trainer.best_action_accuracy,
        'best_actor_accuracy': trainer.best_actor_accuracy,
        'train_history': trainer.train_history,
        'val_history': trainer.val_history,
        'test_metrics': test_metrics if 'test_metrics' in locals() else {}
    }
    
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {results_path}")
    print(f"🏆 Best Combined Accuracy: {trainer.best_val_accuracy:.2f}%")
    print(f"🎯 Best Action Accuracy: {trainer.best_action_accuracy:.2f}%")
    print(f"🎭 Best Actor Accuracy: {trainer.best_actor_accuracy:.2f}%")


if __name__ == "__main__":
    main()
