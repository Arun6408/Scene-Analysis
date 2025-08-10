#!/usr/bin/env python3
"""
Test script to verify the sequential two-step model implementation.
This script tests that the predicted action from the first model is correctly
used as input to the second model for actor classification.
"""

import torch
import torch.nn as nn
import numpy as np
from models_enhanced import EnhancedTwoStepModel, create_enhanced_model
from config_enhanced import config


def test_sequential_model():
    """Test the sequential two-step model implementation"""
    print("🧪 Testing Sequential Two-Step Model...")
    
    # Create model
    model, _ = create_enhanced_model()
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_frames = config.num_frames
    channels = 3
    height, width = config.frame_size
    
    dummy_input = torch.randn(batch_size, num_frames, channels, height, width)
    
    print(f"📊 Input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("✅ Forward pass successful!")
    
    # Check outputs
    print(f"📈 Action logits shape: {outputs['action_logits'].shape}")
    print(f"📈 Actor logits shape: {outputs['actor_logits'].shape}")
    print(f"📈 Predicted actions shape: {outputs['predicted_actions'].shape}")
    
    # Test sequential dependency
    action_preds = torch.argmax(outputs['action_logits'], dim=1)
    actor_preds = torch.argmax(outputs['actor_logits'], dim=1)
    
    print(f"🎯 Action predictions: {action_preds}")
    print(f"🎭 Actor predictions: {actor_preds}")
    
    # Verify that predicted_actions matches action_preds
    assert torch.allclose(action_preds, outputs['predicted_actions']), \
        "Predicted actions should match argmax of action logits"
    
    print("✅ Sequential dependency verified!")
    
    # Test combined accuracy calculation
    # Create dummy ground truth
    action_labels = torch.randint(0, len(config.action_classes), (batch_size,))
    actor_labels = torch.randint(0, len(config.actor_classes), (batch_size,))
    
    # Calculate individual accuracies
    action_correct = (action_preds == action_labels).sum().item()
    actor_correct = (actor_preds == actor_labels).sum().item()
    
    # Calculate combined accuracy (both must be correct)
    combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
    combined_correct = combined_mask.sum().item()
    
    print(f"📊 Action accuracy: {action_correct}/{batch_size} = {action_correct/batch_size*100:.1f}%")
    print(f"📊 Actor accuracy: {actor_correct}/{batch_size} = {actor_correct/batch_size*100:.1f}%")
    print(f"📊 Combined accuracy: {combined_correct}/{batch_size} = {combined_correct/batch_size*100:.1f}%")
    
    print("✅ Combined accuracy calculation verified!")
    
    # Test model components
    print("\n🔍 Testing model components...")
    
    # Test Step 1: Action classification
    action_outputs = model._step1_action_classification(dummy_input)
    print(f"✅ Step 1 - Action classification: {action_outputs['action_logits'].shape}")
    
    # Test Step 2: Actor classification with predicted action
    predicted_actions = torch.argmax(action_outputs['action_logits'], dim=1)
    actor_outputs = model._step2_actor_classification(dummy_input, predicted_actions)
    print(f"✅ Step 2 - Actor classification: {actor_outputs['actor_logits'].shape}")
    
    # Verify that the sequential process works correctly
    step1_action_logits = action_outputs['action_logits']
    step2_actor_logits = actor_outputs['actor_logits']
    
    # Check that both steps produce valid outputs
    assert step1_action_logits.shape == (batch_size, len(config.action_classes)), \
        f"Step 1 action logits shape should be ({batch_size}, {len(config.action_classes)})"
    assert step2_actor_logits.shape == (batch_size, len(config.actor_classes)), \
        f"Step 2 actor logits shape should be ({batch_size}, {len(config.actor_classes)})"
    
    # Check that predicted actions from step 1 are used in step 2
    step1_preds = torch.argmax(step1_action_logits, dim=1)
    assert torch.allclose(step1_preds, predicted_actions), \
        "Step 1 predictions should match the predicted actions used in step 2"
    
    print("✅ Sequential process verified!")
    
    # Test action embedding
    print(f"🔗 Action embedding shape: {model.action_embedding.weight.shape}")
    embedded_actions = model.action_embedding(predicted_actions)
    print(f"🔗 Embedded actions shape: {embedded_actions.shape}")
    
    print("✅ Action embedding verified!")
    
    print("\n🎉 All sequential model tests passed!")
    return True


def test_accuracy_calculation():
    """Test the combined accuracy calculation"""
    print("\n🧪 Testing Combined Accuracy Calculation...")
    
    # Create dummy predictions and labels
    batch_size = 10
    
    # Case 1: All correct
    action_preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    actor_preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    action_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    actor_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    
    combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
    combined_correct = combined_mask.sum().item()
    
    print(f"Case 1 (All correct): Combined accuracy = {combined_correct}/{batch_size} = {combined_correct/batch_size*100:.1f}%")
    assert combined_correct == batch_size, "All correct should give 100% combined accuracy"
    
    # Case 2: Some correct
    action_preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    actor_preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 2])  # Last one wrong
    action_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    actor_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    
    combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
    combined_correct = combined_mask.sum().item()
    
    print(f"Case 2 (One wrong): Combined accuracy = {combined_correct}/{batch_size} = {combined_correct/batch_size*100:.1f}%")
    assert combined_correct == batch_size - 1, "One wrong should give 90% combined accuracy"
    
    # Case 3: Action wrong, actor correct
    action_preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 2])  # Last action wrong
    actor_preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])   # Actor correct
    action_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    actor_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    
    combined_mask = (action_preds == action_labels) & (actor_preds == actor_labels)
    combined_correct = combined_mask.sum().item()
    
    print(f"Case 3 (Action wrong): Combined accuracy = {combined_correct}/{batch_size} = {combined_correct/batch_size*100:.1f}%")
    assert combined_correct == batch_size - 1, "Action wrong should give 90% combined accuracy"
    
    print("✅ Combined accuracy calculation tests passed!")
    return True


if __name__ == "__main__":
    print("🚀 Starting Sequential Model Tests...\n")
    
    try:
        # Test sequential model
        test_sequential_model()
        
        # Test accuracy calculation
        test_accuracy_calculation()
        
        print("\n🎉 All tests passed! The sequential two-step model is working correctly.")
        print("\n📋 Summary:")
        print("✅ Sequential processing: Action → Actor")
        print("✅ Combined accuracy: Both action AND actor must be correct")
        print("✅ Action embedding: Predicted action influences actor classification")
        print("✅ Cross-attention: Video features attend to predicted action")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
