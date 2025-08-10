#!/usr/bin/env python3
"""
Enhanced Two-Step Action Recognition Test Script
Comprehensive testing for the enhanced system
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import json

from config_enhanced import config
from models_enhanced import create_enhanced_model, calculate_model_complexity
from dataset_enhanced import create_enhanced_data_loaders, get_dataset_statistics
from trainer_enhanced import create_enhanced_trainer


def test_data_loader():
    """Test data loader creation"""
    print("🧪 Test 1: Data Loader Creation")
    try:
        train_loader, val_loader, test_loader = create_enhanced_data_loaders()
        print(f"✅ Train loader: {len(train_loader)} batches")
        print(f"✅ Val loader: {len(val_loader)} batches")
        print(f"✅ Test loader: {len(test_loader)} batches")
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"✅ Batch shape: {batch['frames'].shape}")
        print(f"✅ Action labels: {batch['action_label'].shape}")
        print(f"✅ Actor labels: {batch['actor_label'].shape}")
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\n🧪 Test 2: Model Creation")
    try:
        model, loss_fn = create_enhanced_model()
        print(f"✅ Model created successfully")
        print(f"✅ Loss function created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, config.num_frames, 3, *config.frame_size)
        outputs = model(dummy_input)
        print(f"✅ Forward pass successful")
        print(f"✅ Action logits shape: {outputs['action_logits'].shape}")
        print(f"✅ Actor logits shape: {outputs['actor_logits'].shape}")
        
        # Test loss calculation
        action_labels = torch.randint(0, len(config.action_classes), (2,))
        actor_labels = torch.randint(0, len(config.actor_classes), (2,))
        loss_dict = loss_fn(outputs['action_logits'], outputs['actor_logits'], 
                           action_labels, actor_labels)
        print(f"✅ Loss calculation successful")
        print(f"✅ Total loss: {loss_dict['total_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_experiment_directory():
    """Test experiment directory creation"""
    print("\n🧪 Test 3: Experiment Directory Creation")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join("experiments", f"test_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["checkpoints", "plots", "logs", "results", "models"]
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
        print(f"✅ Experiment directory created: {experiment_dir}")
        return experiment_dir
    except Exception as e:
        print(f"❌ Experiment directory test failed: {e}")
        return None


def test_trainer_creation(experiment_dir):
    """Test trainer creation"""
    print("\n🧪 Test 4: Trainer Creation")
    try:
        trainer = create_enhanced_trainer(experiment_dir)
        print(f"✅ Trainer created successfully")
        print(f"✅ Model device: {trainer.model.device}")
        print(f"✅ Optimizer: {type(trainer.optimizer).__name__}")
        print(f"✅ Scheduler: {type(trainer.scheduler).__name__}")
        return trainer
    except Exception as e:
        print(f"❌ Trainer creation test failed: {e}")
        return None


def test_forward_pass(trainer):
    """Test forward pass"""
    print("\n🧪 Test 5: Forward Pass")
    try:
        # Get one batch
        batch = next(iter(trainer.train_loader))
        frames = batch['frames'].to(trainer.device)
        action_labels = batch['action_label'].to(trainer.device)
        actor_labels = batch['actor_label'].to(trainer.device)
        
        # Forward pass
        outputs = trainer.model(frames)
        print(f"✅ Forward pass successful")
        print(f"✅ Output shapes: {outputs['action_logits'].shape}, {outputs['actor_logits'].shape}")
        
        # Test loss calculation
        loss_dict = trainer.loss_fn(outputs['action_logits'], outputs['actor_logits'],
                                   action_labels, actor_labels)
        print(f"✅ Loss calculation successful")
        print(f"✅ Loss values: {loss_dict}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return False


def test_training_step(trainer):
    """Test one training step"""
    print("\n🧪 Test 6: Training Step")
    try:
        # Get one batch
        batch = next(iter(trainer.train_loader))
        frames = batch['frames'].to(trainer.device)
        action_labels = batch['action_label'].to(trainer.device)
        actor_labels = batch['actor_label'].to(trainer.device)
        
        # Training step
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        outputs = trainer.model(frames)
        loss_dict = trainer.loss_fn(outputs['action_logits'], outputs['actor_logits'],
                                   action_labels, actor_labels)
        
        loss_dict['total_loss'].backward()
        trainer.optimizer.step()
        
        print(f"✅ Training step successful")
        print(f"✅ Loss: {loss_dict['total_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        return False


def test_validation_step(trainer):
    """Test one validation step"""
    print("\n🧪 Test 7: Validation Step")
    try:
        # Get one batch
        batch = next(iter(trainer.val_loader))
        frames = batch['frames'].to(trainer.device)
        action_labels = batch['action_label'].to(trainer.device)
        actor_labels = batch['actor_label'].to(trainer.device)
        
        # Validation step
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(frames)
            loss_dict = trainer.loss_fn(outputs['action_logits'], outputs['actor_logits'],
                                       action_labels, actor_labels)
        
        # Calculate accuracy
        action_preds = torch.argmax(outputs['action_logits'], dim=1)
        actor_preds = torch.argmax(outputs['actor_logits'], dim=1)
        
        action_acc = (action_preds == action_labels).float().mean() * 100
        actor_acc = (actor_preds == actor_labels).float().mean() * 100
        
        print(f"✅ Validation step successful")
        print(f"✅ Loss: {loss_dict['total_loss'].item():.4f}")
        print(f"✅ Action accuracy: {action_acc:.2f}%")
        print(f"✅ Actor accuracy: {actor_acc:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Validation step test failed: {e}")
        return False


def test_training_epoch(trainer):
    """Test one training epoch"""
    print("\n🧪 Test 8: Training Epoch")
    try:
        # Run one epoch
        train_metrics = trainer.train_epoch()
        print(f"✅ Training epoch successful")
        print(f"✅ Train metrics: {train_metrics}")
        return True
    except Exception as e:
        print(f"❌ Training epoch test failed: {e}")
        return False


def test_validation_epoch(trainer):
    """Test one validation epoch"""
    print("\n🧪 Test 9: Validation Epoch")
    try:
        # Run one epoch
        val_metrics = trainer.validate_epoch()
        print(f"✅ Validation epoch successful")
        print(f"✅ Val metrics: {val_metrics}")
        return True
    except Exception as e:
        print(f"❌ Validation epoch test failed: {e}")
        return False


def test_checkpoint_saving(trainer):
    """Test checkpoint saving and loading"""
    print("\n🧪 Test 10: Checkpoint Saving/Loading")
    try:
        # Save checkpoint
        trainer.save_checkpoint(0, is_best=True)
        print(f"✅ Checkpoint saved successfully")
        
        # Load checkpoint
        success, epoch = trainer.load_checkpoint(
            os.path.join(trainer.experiment_dir, 'checkpoints', 'best_checkpoint.pth')
        )
        print(f"✅ Checkpoint loaded successfully: epoch {epoch}")
        
        return True
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False


def test_model_complexity():
    """Test model complexity calculation"""
    print("\n🧪 Test 11: Model Complexity")
    try:
        model, _ = create_enhanced_model()
        dummy_input = torch.randn(1, config.num_frames, 3, *config.frame_size)
        complexity = calculate_model_complexity(model, dummy_input.shape)
        
        print(f"✅ Model complexity calculated")
        print(f"✅ Total parameters: {complexity['total_parameters']:,}")
        print(f"✅ FLOPs: {complexity['flops']:,}")
        
        return True
    except Exception as e:
        print(f"❌ Model complexity test failed: {e}")
        return False


def test_dataset_statistics():
    """Test dataset statistics"""
    print("\n🧪 Test 12: Dataset Statistics")
    try:
        stats = get_dataset_statistics()
        print(f"✅ Dataset statistics calculated")
        print(f"✅ Train samples: {stats['train_samples']}")
        print(f"✅ Val samples: {stats['val_samples']}")
        print(f"✅ Test samples: {stats['test_samples']}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset statistics test failed: {e}")
        return False


def test_configuration():
    """Test configuration"""
    print("\n🧪 Test 13: Configuration")
    try:
        print(f"✅ Backbone: {config.backbone}")
        print(f"✅ Frame size: {config.frame_size}")
        print(f"✅ Number of frames: {config.num_frames}")
        print(f"✅ Batch size: {config.batch_size}")
        print(f"✅ Learning rate: {config.learning_rate}")
        print(f"✅ Action classes: {len(config.action_classes)}")
        print(f"✅ Actor classes: {len(config.actor_classes)}")
        print(f"✅ Valid labels: {len(config.valid_labels)}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Enhanced Two-Step Action Recognition Tests")
    print("="*80)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Model Creation", test_model_creation),
        ("Configuration", test_configuration),
        ("Dataset Statistics", test_dataset_statistics),
        ("Model Complexity", test_model_complexity),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    # Advanced tests that require trainer
    experiment_dir = test_experiment_directory()
    if experiment_dir:
        trainer = test_trainer_creation(experiment_dir)
        if trainer:
            advanced_tests = [
                ("Forward Pass", lambda: test_forward_pass(trainer)),
                ("Training Step", lambda: test_training_step(trainer)),
                ("Validation Step", lambda: test_validation_step(trainer)),
                ("Training Epoch", lambda: test_training_epoch(trainer)),
                ("Validation Epoch", lambda: test_validation_epoch(trainer)),
                ("Checkpoint Saving", lambda: test_checkpoint_saving(trainer)),
            ]
            
            for test_name, test_func in advanced_tests:
                try:
                    if test_func():
                        passed += 1
                        print(f"✅ {test_name} PASSED")
                    else:
                        print(f"❌ {test_name} FAILED")
                except Exception as e:
                    print(f"❌ {test_name} FAILED with exception: {e}")
                total += 1
    
    print("\n" + "="*80)
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced system is ready for training.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
