import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class EnhancedConfig:
    # Dataset paths
    dataset_root: str = "FYP Dataset"
    videos_dir: str = "FYP Dataset/clips320H"
    annotations_dir: str = "FYP Dataset/Annotations"
    csv_file: str = "FYP Dataset/videoset.csv"
    
    # Optimized model parameters for RTX 3050 4GB
    input_channels: int = 3
    frame_size: Tuple[int, int] = (192, 192)  # Reduced resolution for speed
    num_frames: int = 16  # Reduced frames for faster training
    temporal_stride: int = 1  # Increased stride for faster processing
    
    # Training parameters optimized for speed
    batch_size: int = 4  # Increased batch size for RTX 3050
    num_epochs: int = 30  
    learning_rate: float = 1e-4  # Slightly higher for faster convergence
    weight_decay: float = 1e-4
    warmup_epochs: int = 5  # No warmup for 1 epoch
    
    # Simplified model architecture for speed
    backbone: str = "efficientnet_b4"  
    shared_backbone: bool = False  # Shared backbone to save memory
    use_attention: bool = True  # Disable attention for speed
    use_temporal_modeling: bool = True  # Disable temporal modeling for speed
    use_feature_fusion: bool = True  # Disable feature fusion for speed
    
   # Enhanced data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5        # Common strong augmentation
    color_jitter_prob: float = 0.4           # Slightly higher for better color variation
    random_crop_prob: float = 0.6            # Good for spatial robustness
    mixup_alpha: float = 0.2                 # Light mixup, helps regularization
    cutmix_prob: float = 0.3                  # Occasional cutmix for diversity
    auto_augment: bool = True                 # Let model learn strong aug policies
    random_erasing: bool = True               # Helps occlusion robustness
    temporal_augmentation: bool = True        # Important for video datasets

    
    # Simplified loss functions
    action_loss_type: str = "focal"  # Enchanced loss for performance
    actor_loss_type: str = "focal"  # Enchanced loss for performance
    label_smoothing: float = 0.1  # Disable label smoothing for speed
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    action_weight: float = 1.0
    actor_weight: float = 1.0
    
    # Simplified optimization
    optimizer: str = "adamw"  # Simpler optimizer for speed
    scheduler: str = "cosine_warmup"  # Simpler scheduler for speed
    step_size: int = 30
    gamma: float = 0.1
    min_lr: float = 1e-6
    # Enhanced regularization
    dropout_rate: float = 0.4          # Slightly higher to improve generalization
    stochastic_depth: float = 0.1      # Light stochastic depth for deeper models
    drop_path_rate: float = 0.1        # Mild drop path for ResNet/EfficientNet

    # Evaluation and saving
    eval_frequency: int = 1
    save_frequency: int = 1  # Save every epoch for 1 epoch test
    
    # Logging and experiment management
    log_dir: str = "logs_enhanced"
    model_dir: str = "models_enhanced"
    experiment_name: str = "two_step"  # Fixed experiment name
    use_wandb: bool = False
    use_tensorboard: bool = False  # Disable tensorboard for speed
    
    # Hardware optimization for RTX 3050 4GB
    device: str = 'cuda'
    num_workers: int = 4  # Increased workers for faster data loading
    pin_memory: bool = True
    mixed_precision: bool = True  # Keep mixed precision for memory efficiency
    
    # Data splits
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Class mappings for enhanced two-step approach
    action_classes: List[str] = None
    actor_classes: List[str] = None
    action_to_actors: Dict[str, List[str]] = None
    actor_to_actions: Dict[str, List[str]] = None
    valid_labels: List[int] = None

    # Enhanced model parameters
    hidden_dim: int = 1024      # More expressive feature space
    num_heads: int = 8         # Standard for good attention resolution
    num_layers: int = 6        # Deeper for richer representation
    mlp_ratio: float = 4.0     # Common choice for transformer blocks

    
    def __post_init__(self):
        # Define action and actor classes
        self.action_classes = ["climbing", "crawling", "eating", "flying", 
                             "jumping", "rolling", "running", "walking"]
        self.actor_classes = ["adult", "baby", "ball", "bird", "car", "cat", "dog"]
        
        # Enhanced action-actor mappings with validation
        self.action_to_actors = {
            "climbing": ["adult", "baby", "bird", "cat"],
            "crawling": ["adult", "baby", "dog"],
            "eating": ["adult", "bird", "cat", "dog"],
            "flying": ["ball", "bird", "car"],
            "jumping": ["adult", "ball", "bird", "car", "cat", "dog"],
            "rolling": ["adult", "baby", "ball", "bird", "car", "cat", "dog"],
            "running": ["adult", "car", "cat", "dog"],
            "walking": ["adult", "baby", "bird", "cat", "dog"]
        }
        
        # Reverse mapping for enhanced training
        self.actor_to_actions = {}
        for actor in self.actor_classes:
            self.actor_to_actions[actor] = []
            for action, actors in self.action_to_actors.items():
                if actor in actors:
                    self.actor_to_actions[actor].append(action)
        
        # Valid actor-action combinations from dataset
        self.valid_labels = [
            11, 12, 13, 15, 16, 17, 18, 19,  # adult actions
            21, 22, 26, 28, 29,  # baby actions
            34, 35, 36, 39,  # ball actions
            41, 43, 44, 45, 46, 48, 49,  # bird actions
            54, 55, 56, 57, 59,  # car actions
            61, 63, 65, 66, 67, 68, 69,  # cat actions
            72, 73, 75, 76, 77, 78, 79   # dog actions
        ]
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

# Global config instance
config = EnhancedConfig()
