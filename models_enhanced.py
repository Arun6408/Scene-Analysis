import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, efficientnet_b3, efficientnet_b4, efficientnet_b0
import timm
from typing import Dict, Tuple, Optional
import math
from config_enhanced import config


class MultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with relative positional encoding"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Relative positional encoding
        self.rel_pos_enc = nn.Parameter(torch.randn(2 * 32 - 1, self.head_dim))
    
    def forward(self, x: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention
        
        Args:
            x: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim] (optional, defaults to x for self-attention)
            value: Value tensor [batch_size, seq_len, embed_dim] (optional, defaults to x for self-attention)
        """
        # Use x as key and value for self-attention if not provided
        if key is None:
            key = x
        if value is None:
            value = x
            
        batch_size, seq_len, embed_dim = x.shape
        key_len = key.size(1)
        value_len = value.size(1)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, value_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative positional encoding (only for self-attention)
        if key is x and value is x:
            rel_pos = self._get_relative_positions(seq_len)
            rel_pos_scores = torch.matmul(q, rel_pos.transpose(-2, -1))
            scores = scores + rel_pos_scores
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Get relative positional encodings"""
        positions = torch.arange(seq_len, device=self.rel_pos_enc.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions += seq_len - 1  # Shift to non-negative
        return self.rel_pos_enc[relative_positions]


class TemporalTransformer(nn.Module):
    """Enhanced Temporal Transformer for video understanding"""
    
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Temporal positional encoding
        self.temporal_pos_enc = nn.Parameter(torch.randn(1, 100, embed_dim))  # Support up to 100 frames
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            }) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Add temporal positional encoding
        if seq_len <= self.temporal_pos_enc.size(1):
            pos_enc = self.temporal_pos_enc[:, :seq_len, :]
        else:
            # Interpolate if sequence is longer
            pos_enc = F.interpolate(
                self.temporal_pos_enc.transpose(1, 2), 
                size=seq_len, 
                mode='linear'
            ).transpose(1, 2)
        
        x = x + pos_enc
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            x = layer['attention'](x)
            x = residual + x
            
            # MLP
            residual = x
            x = layer['norm2'](x)
            x = layer['mlp'](x)
            x = residual + x
        
        x = self.final_norm(x)
        return x


class EnhancedTwoStepModel(nn.Module):
    """
    Enhanced Two-Step Action Recognition Model with Sequential Processing
    Step 1: Classify action (climbing, crawling, eating, etc.)
    Step 2: Classify actor (adult, baby, bird, etc.) given the predicted action and video
    """
    
    def __init__(self, 
                 backbone: str = 'efficientnet_b3',
                 num_action_classes: int = 8,
                 num_actor_classes: int = 7,
                 shared_backbone: bool = False,
                 dropout_rate: float = 0.3,
                 hidden_dim: int = 1024,
                num_heads: int = 8,
                 num_layers: int = 4):
        super(EnhancedTwoStepModel, self).__init__()
        
        self.backbone = backbone
        self.num_action_classes = num_action_classes
        self.num_actor_classes = num_actor_classes
        self.shared_backbone = shared_backbone
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        
        # Create backbones
        if shared_backbone:
            self.feature_extractor = self._create_backbone()
            self.action_classifier = self._create_action_classifier()
            self.actor_classifier = self._create_actor_classifier()
        else:
            # Separate backbones for better specialization
            self.action_backbone = self._create_backbone()
            self.actor_backbone = self._create_backbone()
            self.action_classifier = self._create_action_classifier()
            self.actor_classifier = self._create_actor_classifier()
        
        # Enhanced attention mechanisms
        self.action_attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.actor_attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        
        # Temporal modeling
        self.action_temporal = TemporalTransformer(hidden_dim, num_heads, num_layers, dropout=dropout_rate)
        self.actor_temporal = TemporalTransformer(hidden_dim, num_heads, num_layers, dropout=dropout_rate)
        
        # Action embedding for actor model
        self.action_embedding = nn.Embedding(num_action_classes, hidden_dim)
        
        # Feature fusion layers
        self.action_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.actor_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # video + action + cross-attention
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-attention between action and actor
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        
        # Final classifiers
        self.action_final = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_action_classes)
        )
        
        self.actor_final = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_actor_classes)
        )
    
    def _create_backbone(self) -> nn.Module:
        """Create enhanced backbone network"""
        if self.backbone == 'efficientnet_b0':
            model = efficientnet_b0(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'efficientnet_b3':
            model = efficientnet_b3(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'efficientnet_b4':
            model = efficientnet_b4(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'resnet101':
            model = resnet101(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'resnet50':
            model = resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        return model
    
    def _create_action_classifier(self) -> nn.Module:
        """Create action classifier"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 4, self.num_action_classes)
        )
    
    def _create_actor_classifier(self) -> nn.Module:
        """Create actor classifier"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 4, self.num_actor_classes)
        )
    
    def _extract_features(self, x: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        """Extract features from backbone with enhanced processing"""
        batch_size, num_frames, channels, height, width = x.size()
        
        # Reshape for batch processing
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features
        features = backbone(x)
        
        # Handle different backbone output shapes
        if features.dim() == 4:
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(batch_size, num_frames, -1)
        else:
            features = features.view(batch_size, num_frames, -1)
        
        # Project to consistent hidden dimension
        if features.size(-1) != self.hidden_dim:
            features = nn.Linear(features.size(-1), self.hidden_dim).to(features.device)(features)
        
        return features
    
    def _step1_action_classification(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Step 1: Classify action from video"""
        if self.shared_backbone:
            features = self._extract_features(x, self.feature_extractor)
        else:
            features = self._extract_features(x, self.action_backbone)
        
        # Apply attention for action classification
        action_features = self.action_attention(features)
        action_temporal = self.action_temporal(action_features)
        
        # Global temporal pooling
        action_global = torch.mean(action_temporal, dim=1)  # [batch_size, hidden_dim]
        
        # Feature fusion for action
        action_fused = self.action_fusion(torch.cat([action_global, action_global], dim=1))
        
        # Final action classification
        action_logits = self.action_final(action_fused)
        
        return {
            'action_logits': action_logits,
            'action_probs': F.softmax(action_logits, dim=1),
            'action_features': action_fused
        }
    
    def _step2_actor_classification(self, x: torch.Tensor, predicted_action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Step 2: Classify actor given video and predicted action"""
        if self.shared_backbone:
            features = self._extract_features(x, self.feature_extractor)
        else:
            features = self._extract_features(x, self.actor_backbone)
        
        # Apply attention for actor classification
        actor_features = self.actor_attention(features)
        actor_temporal = self.actor_temporal(actor_features)
        
        # Global temporal pooling
        actor_global = torch.mean(actor_temporal, dim=1)  # [batch_size, hidden_dim]
        
        # Embed predicted action
        action_embed = self.action_embedding(predicted_action)  # [batch_size, hidden_dim]
        
        # Cross-attention between video features and action embedding
        actor_context = self.cross_attention(
            actor_global.unsqueeze(1),  # Query: video features
            action_embed.unsqueeze(1),  # Key: action embedding
            action_embed.unsqueeze(1)   # Value: action embedding
        ).squeeze(1)
        
        # Feature fusion for actor (video + action + cross-attention)
        actor_fused = self.actor_fusion(torch.cat([actor_global, action_embed, actor_context], dim=1))
        
        # Final actor classification
        actor_logits = self.actor_final(actor_fused)
        
        return {
            'actor_logits': actor_logits,
            'actor_probs': F.softmax(actor_logits, dim=1),
            'actor_features': actor_fused
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Sequential forward pass for two-step classification
        
        Args:
            x: Input tensor of shape [batch_size, num_frames, channels, height, width]
        
        Returns:
            Dictionary containing action and actor predictions
        """
        # Step 1: Action classification
        action_outputs = self._step1_action_classification(x)
        action_preds = torch.argmax(action_outputs['action_logits'], dim=1)
        
        # Step 2: Actor classification using predicted action
        actor_outputs = self._step2_actor_classification(x, action_preds)
        
        return {
            'action_logits': action_outputs['action_logits'],
            'actor_logits': actor_outputs['actor_logits'],
            'action_probs': action_outputs['action_probs'],
            'actor_probs': actor_outputs['actor_probs'],
            'action_features': action_outputs['action_features'],
            'actor_features': actor_outputs['actor_features'],
            'predicted_actions': action_preds  # Store for training/validation
        }


class EnhancedLoss(nn.Module):
    """
    Enhanced combined loss function for two-step classification
    """
    
    def __init__(self, 
                 action_loss_type: str = 'focal',
                 actor_loss_type: str = 'focal',
                 action_weight: float = 1.0,
                 actor_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 consistency_weight: float = 0.1):
        super(EnhancedLoss, self).__init__()
        
        self.action_weight = action_weight
        self.actor_weight = actor_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.consistency_weight = consistency_weight
        
        # Create loss functions
        self.action_loss = self._create_loss(action_loss_type)
        self.actor_loss = self._create_loss(actor_loss_type)
        
        # Consistency loss for action-actor relationship
        self.consistency_loss = nn.KLDivLoss(reduction='batchmean')
    
    def _create_loss(self, loss_type: str) -> nn.Module:
        """Create enhanced loss function"""
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif loss_type == 'focal':
            return EnhancedFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        elif loss_type == 'label_smoothing':
            return LabelSmoothingLoss(smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, 
                action_logits: torch.Tensor,
                actor_logits: torch.Tensor,
                action_labels: torch.Tensor,
                actor_labels: torch.Tensor,
                action_probs: torch.Tensor = None,
                actor_probs: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced combined loss
        
        Args:
            action_logits: Action classification logits
            actor_logits: Actor classification logits
            action_labels: Ground truth action labels
            actor_labels: Ground truth actor labels
            action_probs: Action probabilities (optional)
            actor_probs: Actor probabilities (optional)
        
        Returns:
            Dictionary containing individual and total losses
        """
        # Compute individual losses
        action_loss = self.action_loss(action_logits, action_labels)
        actor_loss = self.actor_loss(actor_logits, actor_labels)
        
        # Consistency loss for action-actor relationship
        consistency_loss = 0.0
        if action_probs is not None and actor_probs is not None:
            # Create target distribution based on action-actor relationships
            batch_size = action_probs.size(0)
            target_dist = torch.zeros_like(action_probs)
            
            for i in range(batch_size):
                action_idx = action_labels[i].item()
                actor_idx = actor_labels[i].item()
                
                # Create target distribution that encourages consistency
                target_dist[i, action_idx] = 1.0
            
            consistency_loss = self.consistency_loss(
                F.log_softmax(action_logits, dim=1),
                target_dist
            )
        
        # Combined loss
        total_loss = (self.action_weight * action_loss + 
                     self.actor_weight * actor_loss + 
                     self.consistency_weight * consistency_loss)
        
        return {
            'action_loss': action_loss,
            'actor_loss': actor_loss,
            'consistency_loss': consistency_loss,
            'total_loss': total_loss
        }


class EnhancedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with adaptive alpha
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(EnhancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced focal loss
        
        Args:
            inputs: Logits from model
            targets: Ground truth labels
        
        Returns:
            Enhanced focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Adaptive alpha based on class frequency
        alpha_t = self.alpha * (1 - pt) + (1 - self.alpha) * pt
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for better generalization
    """
    
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss
        
        Args:
            inputs: Logits from model
            targets: Ground truth labels
        
        Returns:
            Label smoothing loss
        """
        num_classes = inputs.size(-1)
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        log_prob = F.log_softmax(inputs, dim=1)
        loss = -(smooth_one_hot * log_prob).sum(dim=1).mean()
        
        return loss


def create_enhanced_model() -> Tuple['SimplifiedTwoStepModel', EnhancedLoss]:
    """Create simplified two-step model and loss function for fast training"""
    # Use simplified model for faster training
    model = SimplifiedTwoStepModel(
        backbone=config.backbone,
        num_action_classes=len(config.action_classes),
        num_actor_classes=len(config.actor_classes),
        shared_backbone=config.shared_backbone,
        dropout_rate=config.dropout_rate,
        hidden_dim=config.hidden_dim
    )
    
    loss_fn = EnhancedLoss(
        action_loss_type=config.action_loss_type,
        actor_loss_type=config.actor_loss_type,
        action_weight=config.action_weight,
        actor_weight=config.actor_weight,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing
    )
    
    return model, loss_fn


def create_simplified_model() -> Tuple['SimplifiedTwoStepModel', EnhancedLoss]:
    """Create simplified two-step model and loss function"""
    model = SimplifiedTwoStepModel(
        backbone=config.backbone,
        num_action_classes=len(config.action_classes),
        num_actor_classes=len(config.actor_classes),
        shared_backbone=config.shared_backbone,
        dropout_rate=config.dropout_rate,
        hidden_dim=config.hidden_dim
    )
    
    loss_fn = EnhancedLoss(
        action_loss_type=config.action_loss_type,
        actor_loss_type=config.actor_loss_type,
        action_weight=config.action_weight,
        actor_weight=config.actor_weight,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing
    )
    
    return model, loss_fn


def calculate_model_complexity(model, input_size: Tuple[int, int, int, int, int]) -> Dict[str, int]:
    """Calculate model complexity for both enhanced and simplified models"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def count_parameters_by_module(model):
        params_by_module = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if params > 0:
                    params_by_module[name] = params
        return params_by_module
    
    total_params = count_parameters(model)
    params_by_module = count_parameters_by_module(model)
    
    # Calculate FLOPs (approximate)
    batch_size, num_frames, channels, height, width = input_size
    
    # Different FLOP calculation based on model type
    if hasattr(model, 'hidden_dim'):
        hidden_dim = model.hidden_dim
    else:
        hidden_dim = config.hidden_dim
    
    flops = batch_size * num_frames * height * width * channels * hidden_dim * 2  # Approximate
    
    return {
        'total_parameters': total_params,
        'parameters_by_module': params_by_module,
        'flops': flops
    }


class SimplifiedTwoStepModel(nn.Module):
    """
    Simplified Two-Step Action Recognition Model for Fast Training
    Optimized for RTX 3050 4GB GPU
    """
    
    def __init__(self, 
                 backbone: str = 'efficientnet_b0',
                 num_action_classes: int = 8,
                 num_actor_classes: int = 7,
                 shared_backbone: bool = True,
                 dropout_rate: float = 0.1,
                 hidden_dim: int = 512):
        super(SimplifiedTwoStepModel, self).__init__()
        
        self.backbone = backbone
        self.num_action_classes = num_action_classes
        self.num_actor_classes = num_actor_classes
        self.shared_backbone = shared_backbone
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        
        # Create shared backbone
        self.feature_extractor = self._create_backbone()
        
        # Simple feature projection
        self.feature_projection = nn.Linear(1280, hidden_dim)  # EfficientNet-B0 output size
        
        # Simple classifiers
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_action_classes)
        )
        
        self.actor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_actor_classes)
        )
    
    def _create_backbone(self) -> nn.Module:
        """Create simplified backbone network"""
        if self.backbone == 'efficientnet_b0':
            model = efficientnet_b0(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'efficientnet_b3':
            model = efficientnet_b3(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'efficientnet_b4':
            model = efficientnet_b4(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'resnet101':
            model = resnet101(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.backbone == 'resnet50':
            model = resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        return model
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone with simplified processing"""
        batch_size, num_frames, channels, height, width = x.size()
        
        # Reshape for batch processing
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Handle different backbone output shapes
        if features.dim() == 4:
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.squeeze(-1).squeeze(-1)
        
        # Project to consistent hidden dimension
        if features.size(-1) != self.hidden_dim:
            features = self.feature_projection(features)
        
        # Reshape back to batch format
        features = features.view(batch_size, num_frames, -1)
        
        # Simple temporal pooling (mean)
        features = torch.mean(features, dim=1)  # [batch_size, hidden_dim]
        
        return features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for simplified two-step model
        
        Args:
            x: Input video tensor [batch_size, num_frames, channels, height, width]
        
        Returns:
            Dictionary containing action and actor logits and probabilities
        """
        # Extract features
        features = self._extract_features(x)
        
        # Action classification
        action_logits = self.action_classifier(features)
        action_probs = F.softmax(action_logits, dim=1)
        
        # Actor classification
        actor_logits = self.actor_classifier(features)
        actor_probs = F.softmax(actor_logits, dim=1)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'actor_logits': actor_logits,
            'actor_probs': actor_probs,
            'features': features
        }
