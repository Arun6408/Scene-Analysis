import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import logging
from tqdm import tqdm

from config_enhanced import config
from models_enhanced import create_enhanced_model, create_simplified_model, calculate_model_complexity
from dataset_enhanced import create_enhanced_data_loaders, get_dataset_statistics


class EnhancedTrainer:
    """
    Enhanced Trainer for Two-Step Action Recognition
    Optimized for maximum accuracy with advanced training techniques
    """
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create data loaders
        print("Creating enhanced data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_enhanced_data_loaders()
        
        # Create model and loss function
        print("Creating simplified model for faster training...")
        self.model, self.loss_fn = create_simplified_model()
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_action_accuracy = 0.0
        self.best_actor_accuracy = 0.0
        self.train_history = {'loss': [], 'action_acc': [], 'actor_acc': [], 'combined_acc': []}
        self.val_history = {'loss': [], 'action_acc': [], 'actor_acc': [], 'combined_acc': []}
        
        # Logging
        self.setup_logging()
        
        # Calculate model complexity
        dummy_input = torch.randn(1, config.num_frames, 3, *config.frame_size).to(self.device)
        complexity = calculate_model_complexity(self.model, dummy_input.shape)
        print(f"Model Parameters: {complexity['total_parameters']:,}")
        print(f"Model FLOPs: {complexity['flops']:,}")
        
        # Dataset statistics
        self.dataset_stats = get_dataset_statistics()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create enhanced optimizer"""
        if config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create enhanced learning rate scheduler"""
        if config.scheduler == 'cosine_warmup':
            T_0 = max(1, config.num_epochs // 4)  # Ensure T_0 is at least 1
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=2,
                eta_min=config.min_lr
            )
        elif config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.min_lr
            )
        elif config.scheduler == 'step':
            return StepLR(
                self.optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        elif config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def setup_logging(self):
        """Setup simplified logging for faster training"""
        # Create log directories
        os.makedirs(os.path.join(self.experiment_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'results'), exist_ok=True)
        
        # Disable TensorBoard for speed
        self.tensorboard_writer = None
        
        # Setup simple logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.experiment_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with enhanced techniques"""
        self.model.train()
        total_loss = 0.0
        action_correct = 0
        actor_correct = 0
        combined_correct = 0  # Both action and actor must be correct
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            frames = batch['frames'].to(self.device)
            action_labels = batch['action_label'].to(self.device)
            actor_labels = batch['actor_label'].to(self.device)
            
            # Forward pass with mixed precision
            if config.mixed_precision:
                with autocast('cuda'):
                    outputs = self.model(frames)
                    loss_dict = self.loss_fn(
                        outputs['action_logits'],
                        outputs['actor_logits'],
                        action_labels,
                        actor_labels,
                        outputs['action_probs'],
                        outputs['actor_probs']
                    )
            else:
                outputs = self.model(frames)
                loss_dict = self.loss_fn(
                    outputs['action_logits'],
                    outputs['actor_logits'],
                    action_labels,
                    actor_labels,
                    outputs['action_probs'],
                    outputs['actor_probs']
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if config.mixed_precision:
                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total_loss'].backward()
                self.optimizer.step()
            
            # Calculate accuracy
            action_preds = torch.argmax(outputs['action_logits'], dim=1)
            actor_preds = torch.argmax(outputs['actor_logits'], dim=1)
            
            action_correct += (action_preds == action_labels).sum().item()
            actor_correct += (actor_preds == actor_labels).sum().item()
            
            # Combined accuracy: both action and actor must be correct
            combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
            combined_correct += combined_mask.sum().item()
            
            total_samples += action_labels.size(0)
            
            total_loss += loss_dict['total_loss'].item()
            
            # Update progress bar
            action_acc = action_correct / total_samples * 100
            actor_acc = actor_correct / total_samples * 100
            combined_acc = combined_correct / total_samples * 100
            
            progress_bar.set_postfix({
                'Loss': f"{total_loss / (batch_idx + 1):.4f}",
                'Action Acc': f"{action_acc:.2f}%",
                'Actor Acc': f"{actor_acc:.2f}%",
                'Combined Acc': f"{combined_acc:.2f}%"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_action_acc = action_correct / total_samples * 100
        epoch_actor_acc = actor_correct / total_samples * 100
        epoch_combined_acc = combined_correct / total_samples * 100
        
        return {
            'loss': epoch_loss,
            'action_acc': epoch_action_acc,
            'actor_acc': epoch_actor_acc,
            'combined_acc': epoch_combined_acc
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        action_correct = 0
        actor_correct = 0
        combined_correct = 0  # Both action and actor must be correct
        total_samples = 0
        
        all_action_preds = []
        all_action_labels = []
        all_actor_preds = []
        all_actor_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                frames = batch['frames'].to(self.device)
                action_labels = batch['action_label'].to(self.device)
                actor_labels = batch['actor_label'].to(self.device)
                
                # Forward pass
                if config.mixed_precision:
                    with autocast('cuda'):
                        outputs = self.model(frames)
                        loss_dict = self.loss_fn(
                            outputs['action_logits'],
                            outputs['actor_logits'],
                            action_labels,
                            actor_labels
                        )
                else:
                    outputs = self.model(frames)
                    loss_dict = self.loss_fn(
                        outputs['action_logits'],
                        outputs['actor_logits'],
                        action_labels,
                        actor_labels
                    )
                
                # Calculate accuracy
                action_preds = torch.argmax(outputs['action_logits'], dim=1)
                actor_preds = torch.argmax(outputs['actor_logits'], dim=1)
                
                action_correct += (action_preds == action_labels).sum().item()
                actor_correct += (actor_preds == actor_labels).sum().item()
                
                # Combined accuracy: both action and actor must be correct
                combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
                combined_correct += combined_mask.sum().item()
                
                total_samples += action_labels.size(0)
                
                total_loss += loss_dict['total_loss'].item()
                
                # Store predictions for detailed analysis
                all_action_preds.extend(action_preds.cpu().numpy())
                all_action_labels.extend(action_labels.cpu().numpy())
                all_actor_preds.extend(actor_preds.cpu().numpy())
                all_actor_labels.extend(actor_labels.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.val_loader)
        epoch_action_acc = action_correct / total_samples * 100
        epoch_actor_acc = actor_correct / total_samples * 100
        epoch_combined_acc = combined_correct / total_samples * 100
        
        # Calculate F1 scores
        action_f1 = self._calculate_f1_score(all_action_labels, all_action_preds)
        actor_f1 = self._calculate_f1_score(all_actor_labels, all_actor_preds)
        
        return {
            'loss': epoch_loss,
            'action_acc': epoch_action_acc,
            'actor_acc': epoch_actor_acc,
            'combined_acc': epoch_combined_acc,
            'action_f1': action_f1,
            'actor_f1': actor_f1,
            'action_preds': all_action_preds,
            'action_labels': all_action_labels,
            'actor_preds': all_actor_preds,
            'actor_labels': all_actor_labels
        }
    
    def _calculate_f1_score(self, true_labels: List[int], pred_labels: List[int]) -> float:
        """Calculate F1 score"""
        from sklearn.metrics import f1_score
        return f1_score(true_labels, pred_labels, average='weighted')
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save checkpoint with enhanced metadata and error handling"""
        try:
            # Create a lighter checkpoint for faster saving
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'best_val_accuracy': self.best_val_accuracy,
                'best_action_accuracy': self.best_action_accuracy,
                'best_actor_accuracy': self.best_actor_accuracy,
                'train_history': self.train_history,
                'val_history': self.val_history,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
                'dataset_stats': self.dataset_stats
            }
            
            # Save latest checkpoint
            latest_path = os.path.join(self.experiment_dir, 'checkpoints', 'latest_checkpoint.pth')
            torch.save(checkpoint, latest_path, _use_new_zipfile_serialization=False)
            
            # Save epoch checkpoint (only every 5 epochs to save space)
            if epoch % 5 == 0:
                epoch_path = os.path.join(self.experiment_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, epoch_path, _use_new_zipfile_serialization=False)
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(self.experiment_dir, 'checkpoints', 'best_checkpoint.pth')
                torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
                self.logger.info(f"Saved best checkpoint with accuracy: {self.best_val_accuracy:.2f}%")
            
            # Save final checkpoint
            if is_final:
                final_path = os.path.join(self.experiment_dir, 'checkpoints', 'final_checkpoint.pth')
                torch.save(checkpoint, final_path, _use_new_zipfile_serialization=False)
                self.logger.info("Saved final checkpoint")
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            # Try to save a minimal checkpoint
            try:
                minimal_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_accuracy': self.best_val_accuracy,
                }
                minimal_path = os.path.join(self.experiment_dir, 'checkpoints', f'minimal_checkpoint_epoch_{epoch}.pth')
                torch.save(minimal_checkpoint, minimal_path, _use_new_zipfile_serialization=False)
                self.logger.info("Saved minimal checkpoint as fallback")
            except Exception as e2:
                self.logger.error(f"Failed to save minimal checkpoint: {e2}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[bool, int]:
        """Load checkpoint with error handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if checkpoint['scaler_state_dict'] and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_accuracy = checkpoint['best_val_accuracy']
            self.best_action_accuracy = checkpoint['best_action_accuracy']
            self.best_actor_accuracy = checkpoint['best_actor_accuracy']
            self.train_history = checkpoint['train_history']
            self.val_history = checkpoint['val_history']
            
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True, self.current_epoch
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False, 0
    
    def train(self):
        """Simplified training loop for faster execution with error handling"""
        self.logger.info("Starting simplified training...")
        self.logger.info(f"Dataset: {self.dataset_stats['train_samples']} train, "
                         f"{self.dataset_stats['val_samples']} val, "
                         f"{self.dataset_stats['test_samples']} test")
        
        try:
            for epoch in range(self.current_epoch, config.num_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                try:
                    train_metrics = self.train_epoch()
                except Exception as e:
                    self.logger.error(f"Training epoch {epoch} failed: {e}")
                    continue
                
                # Validate epoch
                try:
                    val_metrics = self.validate_epoch()
                except Exception as e:
                    self.logger.error(f"Validation epoch {epoch} failed: {e}")
                    continue
                
                # Update learning rate
                if self.scheduler:
                    try:
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['combined_acc'])
                        else:
                            self.scheduler.step()
                    except Exception as e:
                        self.logger.error(f"Learning rate update failed: {e}")
                
                # Update best metrics
                if val_metrics['combined_acc'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['combined_acc']
                    self.best_action_accuracy = val_metrics['action_acc']
                    self.best_actor_accuracy = val_metrics['actor_acc']
                    try:
                        self.save_checkpoint(epoch, is_best=True)
                    except Exception as e:
                        self.logger.error(f"Failed to save best checkpoint: {e}")
                
                # Update histories
                self.train_history['loss'].append(train_metrics['loss'])
                self.train_history['action_acc'].append(train_metrics['action_acc'])
                self.train_history['actor_acc'].append(train_metrics['actor_acc'])
                self.train_history['combined_acc'].append(train_metrics['combined_acc'])
                
                self.val_history['loss'].append(val_metrics['loss'])
                self.val_history['action_acc'].append(val_metrics['action_acc'])
                self.val_history['actor_acc'].append(val_metrics['actor_acc'])
                self.val_history['combined_acc'].append(val_metrics['combined_acc'])
                
                # Simple logging
                self.logger.info(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Action Acc: {val_metrics['action_acc']:.2f}%, "
                    f"Val Actor Acc: {val_metrics['actor_acc']:.2f}%, "
                    f"Val Combined Acc: {val_metrics['combined_acc']:.2f}%"
                )
                
                # Save checkpoint periodically
                if (epoch + 1) % config.save_frequency == 0:
                    try:
                        self.save_checkpoint(epoch)
                    except Exception as e:
                        self.logger.error(f"Failed to save periodic checkpoint: {e}")
        
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            # Try to save a final checkpoint even if training failed
            try:
                self.save_checkpoint(self.current_epoch, is_final=True)
            except Exception as e2:
                self.logger.error(f"Failed to save final checkpoint: {e2}")
        else:
            # Save final checkpoint if training completed successfully
            try:
                self.save_checkpoint(self.current_epoch, is_final=True)
                self.logger.info("Training completed successfully!")
            except Exception as e:
                self.logger.error(f"Failed to save final checkpoint: {e}")
                self.logger.info("Training completed but final checkpoint save failed!")
    
    def test(self) -> Dict[str, float]:
        """Comprehensive test evaluation"""
        self.logger.info("Running comprehensive test evaluation...")
        
        # Load best model
        best_checkpoint_path = os.path.join(self.experiment_dir, 'checkpoints', 'best_checkpoint.pth')
        if os.path.exists(best_checkpoint_path):
            success, _ = self.load_checkpoint(best_checkpoint_path)
            if not success:
                self.logger.warning("Could not load best checkpoint, using current model")
        else:
            self.logger.warning("No best checkpoint found, using current model")
        
        self.model.eval()
        total_loss = 0.0
        action_correct = 0
        actor_correct = 0
        combined_correct = 0  # Both action and actor must be correct
        total_samples = 0
        
        all_action_preds = []
        all_action_labels = []
        all_actor_preds = []
        all_actor_labels = []
        all_video_ids = []
        all_action_names = []
        all_actor_names = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                frames = batch['frames'].to(self.device)
                action_labels = batch['action_label'].to(self.device)
                actor_labels = batch['actor_label'].to(self.device)
                video_ids = batch['video_id']
                action_names = batch['action_name']
                actor_names = batch['actor_name']
                
                # Forward pass
                if config.mixed_precision:
                    with autocast('cuda'):
                        outputs = self.model(frames)
                        loss_dict = self.loss_fn(
                            outputs['action_logits'],
                            outputs['actor_logits'],
                            action_labels,
                            actor_labels
                        )
                else:
                    outputs = self.model(frames)
                    loss_dict = self.loss_fn(
                        outputs['action_logits'],
                        outputs['actor_logits'],
                        action_labels,
                        actor_labels
                    )
                
                # Calculate accuracy
                action_preds = torch.argmax(outputs['action_logits'], dim=1)
                actor_preds = torch.argmax(outputs['actor_logits'], dim=1)
                
                action_correct += (action_preds == action_labels).sum().item()
                actor_correct += (actor_preds == actor_labels).sum().item()
                
                # Combined accuracy: both action and actor must be correct
                combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
                combined_correct += combined_mask.sum().item()
                
                total_samples += action_labels.size(0)
                
                total_loss += loss_dict['total_loss'].item()
                
                # Store predictions for detailed analysis
                all_action_preds.extend(action_preds.cpu().numpy())
                all_action_labels.extend(action_labels.cpu().numpy())
                all_actor_preds.extend(actor_preds.cpu().numpy())
                all_actor_labels.extend(actor_labels.cpu().numpy())
                all_video_ids.extend(video_ids)
                all_action_names.extend(action_names)
                all_actor_names.extend(actor_names)
        
        # Calculate test metrics
        test_loss = total_loss / len(self.test_loader)
        test_action_acc = action_correct / total_samples * 100
        test_actor_acc = actor_correct / total_samples * 100
        test_combined_acc = combined_correct / total_samples * 100
        
        # Calculate F1 scores
        test_action_f1 = self._calculate_f1_score(all_action_labels, all_action_preds)
        test_actor_f1 = self._calculate_f1_score(all_actor_labels, all_actor_preds)
        
        # Save detailed results
        self._save_test_results(
            all_action_preds, all_action_labels, all_actor_preds, all_actor_labels,
            all_video_ids, all_action_names, all_actor_names
        )
        
        test_metrics = {
            'loss': test_loss,
            'action_acc': test_action_acc,
            'actor_acc': test_actor_acc,
            'combined_acc': test_combined_acc,
            'action_f1': test_action_f1,
            'actor_f1': test_actor_f1
        }
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Loss: {test_loss:.4f}")
        self.logger.info(f"  Action Accuracy: {test_action_acc:.2f}%")
        self.logger.info(f"  Actor Accuracy: {test_actor_acc:.2f}%")
        self.logger.info(f"  Combined Accuracy: {test_combined_acc:.2f}%")
        self.logger.info(f"  Action F1: {test_action_f1:.4f}")
        self.logger.info(f"  Actor F1: {test_actor_f1:.4f}")
        
        return test_metrics
    
    def _save_test_results(self, action_preds, action_labels, actor_preds, actor_labels,
                          video_ids, action_names, actor_names):
        """Save detailed test results"""
        results_df = pd.DataFrame({
            'video_id': video_ids,
            'true_action': action_names,
            'pred_action': [config.action_classes[p] for p in action_preds],
            'true_actor': actor_names,
            'pred_actor': [config.actor_classes[p] for p in actor_labels],
            'action_correct': [a == b for a, b in zip(action_preds, action_labels)],
            'actor_correct': [a == b for a, b in zip(actor_preds, actor_labels)]
        })
        
        results_path = os.path.join(self.experiment_dir, 'results', 'test_predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save confusion matrices
        self._save_confusion_matrices(action_preds, action_labels, actor_preds, actor_labels)
        
        # Save classification reports
        self._save_classification_reports(action_preds, action_labels, actor_preds, actor_labels)
    
    def _save_confusion_matrices(self, action_preds, action_labels, actor_preds, actor_labels):
        """Save confusion matrices"""
        # Action confusion matrix
        action_cm = confusion_matrix(action_labels, action_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(action_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.action_classes,
                   yticklabels=config.action_classes)
        plt.title('Action Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'plots', 'action_confusion_matrix.png'))
        plt.close()
        
        # Actor confusion matrix
        actor_cm = confusion_matrix(actor_labels, actor_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(actor_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.actor_classes,
                   yticklabels=config.actor_classes)
        plt.title('Actor Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'plots', 'actor_confusion_matrix.png'))
        plt.close()
    
    def _save_classification_reports(self, action_preds, action_labels, actor_preds, actor_labels):
        """Save classification reports"""
        # Action classification report
        action_report = classification_report(
            action_labels, action_preds,
            target_names=config.action_classes,
            output_dict=True
        )
        
        # Actor classification report
        actor_report = classification_report(
            actor_labels, actor_preds,
            target_names=config.actor_classes,
            output_dict=True
        )
        
        # Save as CSV
        action_df = pd.DataFrame(action_report).transpose()
        actor_df = pd.DataFrame(actor_report).transpose()
        
        action_df.to_csv(os.path.join(self.experiment_dir, 'results', 'action_classification_report.csv'))
        actor_df.to_csv(os.path.join(self.experiment_dir, 'results', 'actor_classification_report.csv'))


def create_enhanced_trainer(experiment_dir: str) -> EnhancedTrainer:
    """Create enhanced trainer instance"""
    return EnhancedTrainer(experiment_dir)
