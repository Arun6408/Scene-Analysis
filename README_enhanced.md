# Enhanced Two-Step Action Recognition System

A state-of-the-art sequential two-step action recognition system that achieves maximum accuracy through advanced deep learning techniques.

## 🎯 Overview

This system implements a **sequential two-step recognition process**:

1. **Step 1 - Action Classification**: Takes a video as input and predicts the action (climbing, crawling, eating, etc.)
2. **Step 2 - Actor Classification**: Takes the **predicted action from Step 1** along with the original video and predicts the actor (adult, baby, bird, etc.)

**Combined Accuracy**: A prediction is counted as correct **only if both the action and actor match the ground truth**.

## 🏗️ Architecture

### Sequential Two-Step Model
- **First Model**: Video → Action prediction
- **Second Model**: (Predicted Action + Video) → Actor prediction
- **Action Embedding**: The predicted action is embedded and used as context for actor classification
- **Cross-Attention**: Video features attend to the predicted action embedding for better actor prediction

### Key Components
- **Backbone Networks**: EfficientNet-B3/B4 or ResNet-50/101 (separate backbones for action and actor)
- **Multi-Head Attention**: Enhanced with relative positional encoding
- **Temporal Transformer**: Advanced temporal modeling for video understanding
- **Cross-Attention**: Interaction between action and actor features
- **Feature Fusion**: Sophisticated fusion of spatial, temporal, and cross-modal features

## 🚀 Features

### Advanced Training Techniques
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Enhanced Loss Functions**: Focal Loss with adaptive alpha, Label Smoothing
- **Consistency Loss**: Ensures action-actor relationship consistency
- **Advanced Optimizers**: AdamW with cosine annealing warmup
- **Comprehensive Augmentation**: Spatial and temporal augmentations

### Experiment Management
- **Checkpointing**: Automatic saving of best and periodic checkpoints
- **Resume Training**: Seamless training continuation
- **TensorBoard Logging**: Real-time training visualization
- **Weights & Biases**: Optional integration for experiment tracking

### Evaluation & Analysis
- **Sequential Accuracy**: Combined accuracy requiring both action and actor to be correct
- **Detailed Metrics**: Action accuracy, actor accuracy, combined accuracy, F1 scores
- **Visualization**: Training curves, confusion matrices, classification reports
- **Performance Analysis**: Comprehensive model evaluation

## 📊 Dataset

Uses the A2D (Actor-Action) dataset with:
- **Video Clips**: 320H resolution video clips
- **Annotations**: Detailed actor-action labels
- **Label Encoding**: Two-digit integers (e.g., 11 for adult-climbing)
- **Train/Test Split**: Optimized data distribution

## 🛠️ Quick Start

### Installation
```bash
pip install -r requirements_enhanced.txt
```

### Training
```bash
python main_enhanced.py --mode train --experiment-name my_experiment
```

### Testing
```bash
python main_enhanced.py --mode test --experiment-name my_experiment
```

### Resume Training
```bash
python main_enhanced.py --mode train --experiment-name my_experiment --resume
```

## 📁 Project Structure

```
├── config_enhanced.py          # Enhanced configuration
├── models_enhanced.py          # Sequential two-step model architecture
├── dataset_enhanced.py         # Enhanced data loading and augmentation
├── trainer_enhanced.py         # Advanced training with mixed precision
├── main_enhanced.py           # Main training/testing script
├── utils_enhanced.py          # Visualization and analysis utilities
├── test_enhanced.py           # Comprehensive testing suite
├── requirements_enhanced.txt   # Python dependencies
└── README_enhanced.md         # This file
```

## ⚙️ Configuration

Key parameters in `config_enhanced.py`:
- **Model**: `backbone`, `shared_backbone`, `hidden_dim`
- **Training**: `batch_size`, `learning_rate`, `num_epochs`
- **Data**: `frame_size`, `num_frames`, `temporal_stride`
- **Advanced**: `mixed_precision`, `use_attention`, `use_temporal_modeling`

## 📈 Performance

The enhanced system achieves:
- **Sequential Processing**: True two-step recognition with action-to-actor dependency
- **High Accuracy**: Optimized for maximum combined accuracy
- **Robust Training**: Advanced techniques for stable convergence
- **Comprehensive Evaluation**: Detailed analysis of model performance

## 🔬 Advanced Features

### Sequential Dependencies
- **Action → Actor**: The predicted action influences actor classification
- **Cross-Modal Attention**: Video features attend to predicted action
- **Consistency Loss**: Ensures logical action-actor relationships

### Training Optimizations
- **Mixed Precision**: 16-bit training for speed and memory efficiency
- **Gradient Scaling**: Prevents underflow in mixed precision
- **Advanced Schedulers**: Cosine annealing with warm restarts
- **Regularization**: Dropout, stochastic depth, label smoothing

### Evaluation Metrics
- **Combined Accuracy**: Only correct if both action AND actor match ground truth
- **Individual Metrics**: Separate action and actor accuracy tracking
- **F1 Scores**: Balanced precision and recall evaluation
- **Confusion Matrices**: Detailed error analysis

## 🎯 Usage Examples

### Basic Training
```python
from main_enhanced import main

# Train the sequential two-step model
main(mode='train', experiment_name='sequential_experiment')
```

### Testing with Best Model
```python
# Test using the best checkpoint
main(mode='test', experiment_name='sequential_experiment')
```

### Resume Training
```python
# Resume from checkpoint
main(mode='train', experiment_name='sequential_experiment', resume=True)
```

## 📊 Results Analysis

The system provides comprehensive analysis:
- **Training Curves**: Loss and accuracy progression
- **Confusion Matrices**: Action and actor classification errors
- **Classification Reports**: Precision, recall, F1 for each class
- **Performance Comparison**: Before/after enhancement analysis

## 🔧 Customization

### Model Architecture
- Change backbone networks in `config_enhanced.py`
- Modify attention mechanisms in `models_enhanced.py`
- Adjust temporal modeling parameters

### Training Parameters
- Adjust learning rate and batch size
- Modify augmentation strategies
- Change loss function weights

### Data Processing
- Customize frame extraction in `dataset_enhanced.py`
- Modify augmentation pipelines
- Adjust data loading parameters

## 🚨 Important Notes

1. **Sequential Nature**: The model processes action first, then uses the predicted action for actor classification
2. **Combined Accuracy**: A prediction is only correct if both action and actor match ground truth
3. **Memory Requirements**: Mixed precision training reduces memory usage
4. **Checkpointing**: Always save checkpoints for resume capability
5. **Evaluation**: Use combined accuracy as the primary metric

## 📝 License

This enhanced two-step action recognition system is designed for research and educational purposes.
