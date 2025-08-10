import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import torch
from datetime import datetime

from config_enhanced import config


def create_visualizations(trainer, experiment_dir: str):
    """Create comprehensive visualizations for enhanced training"""
    print("📊 Creating enhanced visualizations...")
    
    # Create training curves
    create_training_curves(trainer, experiment_dir)
    
    # Create confusion matrices
    create_confusion_matrices(trainer, experiment_dir)
    
    print("✅ All visualizations created successfully!")


def create_training_curves(trainer, experiment_dir: str):
    """Create enhanced training curves"""
    epochs = range(1, len(trainer.train_history['loss']) + 1)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Training Curves', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(epochs, trainer.train_history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, trainer.val_history['loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Action accuracy curves
    axes[0, 1].plot(epochs, trainer.train_history['action_acc'], 'b-', label='Train Action Acc', linewidth=2)
    axes[0, 1].plot(epochs, trainer.val_history['action_acc'], 'r-', label='Val Action Acc', linewidth=2)
    axes[0, 1].set_title('Action Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actor accuracy curves
    axes[1, 0].plot(epochs, trainer.train_history['actor_acc'], 'b-', label='Train Actor Acc', linewidth=2)
    axes[1, 0].plot(epochs, trainer.val_history['actor_acc'], 'r-', label='Val Actor Acc', linewidth=2)
    axes[1, 0].set_title('Actor Accuracy Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined accuracy curves
    axes[1, 1].plot(epochs, trainer.train_history['combined_acc'], 'b-', label='Train Combined Acc', linewidth=2)
    axes[1, 1].plot(epochs, trainer.val_history['combined_acc'], 'r-', label='Val Combined Acc', linewidth=2)
    axes[1, 1].set_title('Combined Accuracy Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'plots', 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_matrices(trainer, experiment_dir: str):
    """Create enhanced confusion matrices"""
    # Load test results if available
    test_results_path = os.path.join(experiment_dir, 'results', 'test_predictions.csv')
    if os.path.exists(test_results_path):
        results_df = pd.read_csv(test_results_path)
        
        # Action confusion matrix
        action_cm = confusion_matrix(
            [config.action_classes.index(action) for action in results_df['true_action']],
            [config.action_classes.index(action) for action in results_df['pred_action']]
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(action_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.action_classes,
                   yticklabels=config.action_classes)
        plt.title('Enhanced Action Classification Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'plots', 'action_confusion_matrix_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Actor confusion matrix
        actor_cm = confusion_matrix(
            [config.actor_classes.index(actor) for actor in results_df['true_actor']],
            [config.actor_classes.index(actor) for actor in results_df['pred_actor']]
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(actor_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.actor_classes,
                   yticklabels=config.actor_classes)
        plt.title('Enhanced Actor Classification Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'plots', 'actor_confusion_matrix_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.close()


def save_experiment_results(trainer, experiment_dir: str, test_metrics: Dict):
    """Save comprehensive experiment results"""
    print("💾 Saving enhanced experiment results...")
    
    # Save final metrics
    final_metrics = {
        'experiment_dir': experiment_dir,
        'best_val_accuracy': trainer.best_val_accuracy,
        'best_action_accuracy': trainer.best_action_accuracy,
        'best_actor_accuracy': trainer.best_actor_accuracy,
        'test_metrics': test_metrics,
        'train_history': trainer.train_history,
        'val_history': trainer.val_history,
        'config': config.__dict__,
        'dataset_stats': trainer.dataset_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(experiment_dir, 'results', 'final_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    
    # Save training history as CSV
    train_df = pd.DataFrame({
        'epoch': range(1, len(trainer.train_history['loss']) + 1),
        'train_loss': trainer.train_history['loss'],
        'train_action_acc': trainer.train_history['action_acc'],
        'train_actor_acc': trainer.train_history['actor_acc'],
        'train_combined_acc': trainer.train_history['combined_acc'],
        'val_loss': trainer.val_history['loss'],
        'val_action_acc': trainer.val_history['action_acc'],
        'val_actor_acc': trainer.val_history['actor_acc'],
        'val_combined_acc': trainer.val_history['combined_acc']
    })
    
    history_path = os.path.join(experiment_dir, 'results', 'training_history.csv')
    train_df.to_csv(history_path, index=False)
    
    # Create summary report
    create_summary_report(trainer, experiment_dir, test_metrics)
    
    print(f"✅ Results saved to: {experiment_dir}")


def create_summary_report(trainer, experiment_dir: str, test_metrics: Dict):
    """Create comprehensive summary report"""
    report_path = os.path.join(experiment_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENHANCED TWO-STEP ACTION RECOGNITION EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Experiment Directory: {experiment_dir}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Backbone: {config.backbone}\n")
        f.write(f"Shared Backbone: {config.shared_backbone}\n")
        f.write(f"Number of Frames: {config.num_frames}\n")
        f.write(f"Frame Size: {config.frame_size}\n")
        f.write(f"Hidden Dimension: {config.hidden_dim}\n")
        f.write(f"Number of Heads: {config.num_heads}\n")
        f.write(f"Number of Layers: {config.num_layers}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs: {config.num_epochs}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Optimizer: {config.optimizer}\n")
        f.write(f"Scheduler: {config.scheduler}\n")
        f.write(f"Action Loss: {config.action_loss_type}\n")
        f.write(f"Actor Loss: {config.actor_loss_type}\n")
        f.write(f"Mixed Precision: {config.mixed_precision}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train Samples: {trainer.dataset_stats['train_samples']}\n")
        f.write(f"Validation Samples: {trainer.dataset_stats['val_samples']}\n")
        f.write(f"Test Samples: {trainer.dataset_stats['test_samples']}\n")
        f.write(f"Action Classes: {len(config.action_classes)}\n")
        f.write(f"Actor Classes: {len(config.actor_classes)}\n\n")
        
        f.write("BEST VALIDATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Combined Accuracy: {trainer.best_val_accuracy:.2f}%\n")
        f.write(f"Best Action Accuracy: {trainer.best_action_accuracy:.2f}%\n")
        f.write(f"Best Actor Accuracy: {trainer.best_actor_accuracy:.2f}%\n\n")
        
        f.write("TEST RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test Action Accuracy: {test_metrics['action_acc']:.2f}%\n")
        f.write(f"Test Actor Accuracy: {test_metrics['actor_acc']:.2f}%\n")
        f.write(f"Test Combined Accuracy: {test_metrics['combined_acc']:.2f}%\n")
        f.write(f"Test Action F1: {test_metrics['action_f1']:.4f}\n")
        f.write(f"Test Actor F1: {test_metrics['actor_f1']:.4f}\n\n")
        
        f.write("ACTION-ACTOR MAPPINGS:\n")
        f.write("-" * 40 + "\n")
        for action, actors in config.action_to_actors.items():
            f.write(f"{action}: {', '.join(actors)}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("EXPERIMENT COMPLETED SUCCESSFULLY\n")
        f.write("="*80 + "\n")
    
    print(f"📄 Summary report saved to: {report_path}")
