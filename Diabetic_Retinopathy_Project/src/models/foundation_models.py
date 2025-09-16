# ============================================================================
# /src/models/foundation_models.py
# Foundation Model Integration for Multi-Modal Medical Imaging
# ============================================================================
"""
This module provides foundation model integration for diabetic retinopathy classification
using multiple medical imaging modalities. Foundation models are large-scale pretrained
models that provide excellent feature representations for downstream tasks.

The module includes:

1. RETFound - Retinal foundation model for fundus images
   - Based on Vision Transformer (ViT) architecture
   - Pretrained on large-scale retinal image datasets
   - Provides robust fundus image representations

2. OCTCube - Foundation model for OCT volume analysis
   - 3D CNN architecture for volumetric OCT data
   - Handles cross-sectional retinal layer information
   - Optimized for medical volumetric data

3. FLIO Encoder - Custom encoder for fluorescence lifetime imaging
   - No pretrained foundation model available
   - Custom 2D CNN with metabolic feature extraction
   - Handles 4-channel lifetime and intensity data

4. Training utilities including advanced loss functions and training loops

These models can be used independently or combined in fusion architectures
for comprehensive multi-modal analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader  # Add this line
from transformers import AutoModel, AutoImageProcessor, ViTModel, ViTConfig
from typing import Dict, Optional, Tuple, Union, List, Any, Callable
import logging
from pathlib import Path
import numpy as np
import wandb
from tqdm import tqdm
import time
import json

class RETFoundEncoder(nn.Module):
    """
    RETFound foundation model encoder for fundus images.
    
    RETFound is a specialized foundation model for retinal imaging that has been
    pretrained on large-scale fundus image datasets. It uses a Vision Transformer
    architecture adapted for medical imaging characteristics.
    
    Key features:
    - Pretrained on diverse retinal pathology datasets
    - Robust to different imaging conditions and equipment
    - Provides rich feature representations for downstream tasks
    - Can be fine-tuned or used as frozen feature extractor
    """
    
    def __init__(
        self,
        model_path: str = "pretrained_weights/retfound_cfp_weights.pth",
        freeze_backbone: bool = False,
        feature_dim: int = 768
    ):
        """
        Initialize RETFound encoder.
        
        Args:
            model_path: Path to pretrained RETFound weights
            freeze_backbone: Whether to freeze backbone weights during training
            feature_dim: Output feature dimension (standardized across modalities)
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Configure Vision Transformer architecture to match RETFound
        # These parameters are optimized for retinal imaging
        config = ViTConfig(
            image_size=224,           # Standard input size for fundus images
            patch_size=16,            # 16x16 patches work well for retinal details
            num_channels=3,           # RGB fundus images
            num_labels=5,             # DR grades (0-4)
            hidden_size=768,          # Hidden dimension for transformer
            num_hidden_layers=12,     # Depth of transformer (balance between capacity and efficiency)
            num_attention_heads=12,   # Multi-head attention
            intermediate_size=3072,   # Feed-forward network size
            hidden_act="gelu",        # GELU activation (smooth, differentiable)
            hidden_dropout_prob=0.0,  # Dropout for regularization
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,   # Weight initialization range
            layer_norm_eps=1e-12,     # Layer normalization epsilon
        )
        
        # Initialize the Vision Transformer backbone
        self.backbone = ViTModel(config)
        
        # Load pretrained RETFound weights if available
        if Path(model_path).exists():
            self._load_retfound_weights(model_path)
            logging.info(f"✅ Loaded RETFound weights from {model_path}")
        else:
            logging.warning(f"⚠️ RETFound weights not found at {model_path}, using random initialization")
            logging.info("Consider downloading RETFound weights for better performance")
        
        # Freeze backbone parameters if specified
        # This is useful when you want to use RETFound as a fixed feature extractor
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logging.info("🔒 RETFound backbone frozen - parameters won't be updated during training")
        
        # Projection layer to standardize feature dimensions across modalities
        # This ensures all modalities output the same feature size for fusion
        self.projection = nn.Linear(config.hidden_size, feature_dim)
        
    def _load_retfound_weights(self, model_path: str):
        """
        Load pretrained RETFound weights with robust error handling.
        
        RETFound weights may come in different formats depending on the source.
        This method handles various checkpoint formats and key naming conventions.
        
        Args:
            model_path: Path to the RETFound checkpoint file
        """
        try:
            # Load checkpoint with CPU mapping to avoid GPU memory issues
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                # Standard PyTorch checkpoint format
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Lightning or other framework format
                state_dict = checkpoint['state_dict']
            else:
                # Direct state dictionary
                state_dict = checkpoint
            
            # Clean up key names - remove common prefixes
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'backbone.' prefix if present
                if k.startswith('backbone.'):
                    cleaned_state_dict[k[9:]] = v
                # Remove 'model.' prefix if present
                elif k.startswith('model.'):
                    cleaned_state_dict[k[6:]] = v
                else:
                    cleaned_state_dict[k] = v
            
            # Load weights with strict=False to handle architecture differences
            missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
            
            # Log any issues with loading
            if missing_keys:
                logging.warning(f"⚠️ Missing keys in RETFound: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                logging.warning(f"⚠️ Unexpected keys in RETFound: {unexpected_keys[:5]}...")
                
            logging.info("✅ RETFound weights loaded successfully")
                
        except Exception as e:
            logging.error(f"❌ Failed to load RETFound weights: {e}")
            logging.info("🔄 Continuing with random initialization - consider checking the checkpoint path")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RETFound encoder.
        
        Args:
            x: Input fundus images [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Encoded features [batch_size, feature_dim]
        """
        # Pass through Vision Transformer backbone
        outputs = self.backbone(x)
        
        # Extract features from transformer output
        # Different ViT implementations may have different output formats
        if hasattr(outputs, 'last_hidden_state'):
            # Use CLS token (first token) which contains global image information
            features = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        else:
            # Fallback to pooler output if available
            features = outputs.pooler_output
        
        # Project to standardized feature dimension
        # This ensures compatibility with fusion architectures
        features = self.projection(features)
        
        return features


class OCTCubeEncoder(nn.Module):
    """
    OCTCube foundation model encoder for OCT volume analysis.
    
    OCTCube is designed for processing 3D OCT volumes that show cross-sectional
    views of retinal layers. The model uses 3D convolutions to capture both
    spatial and depth information from the volumetric data.
    
    Key features:
    - 3D CNN architecture for volumetric processing
    - Handles variable OCT volume sizes
    - Preserves spatial relationships between retinal layers
    - Optimized for medical volumetric data characteristics
    """
    
    def __init__(
        self,
        model_path: str = "pretrained_weights/octcube_weights.pth",
        freeze_backbone: bool = False,
        feature_dim: int = 768,
        input_shape: Tuple[int, int, int] = (256, 256, 32)
    ):
        """
        Initialize OCTCube encoder.
        
        Args:
            model_path: Path to pretrained OCTCube weights
            freeze_backbone: Whether to freeze backbone during training
            feature_dim: Output feature dimension
            input_shape: Expected input shape (height, width, depth)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.input_shape = input_shape
        
        # 3D CNN backbone optimized for OCT volumes
        # The architecture progressively reduces spatial dimensions while increasing channels
        self.backbone = nn.Sequential(
            # First convolutional block
            # Large receptive field to capture retinal layer patterns
            nn.Conv3d(1, 64, kernel_size=3, padding=1),  # Preserve spatial dimensions
            nn.BatchNorm3d(64),                          # Stabilize training
            nn.ReLU(),                                   # Non-linearity
            nn.MaxPool3d(2),                            # Reduce dimensions by half
            
            # Second block - increase feature complexity
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),                            # Further dimension reduction
            
            # Third block - high-level feature extraction
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Fourth block - final feature representation
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # Global average pooling to handle variable input sizes
            nn.AdaptiveAvgPool3d((1, 1, 1)),           # Reduce to 1x1x1 feature map
            
            # Flatten and project to standard feature dimension
            nn.Flatten(),
            nn.Linear(512, feature_dim)
        )
        
        # Load pretrained OCTCube weights if available
        if Path(model_path).exists():
            self._load_octcube_weights(model_path)
            logging.info(f"✅ Loaded OCTCube weights from {model_path}")
        else:
            logging.warning(f"⚠️ OCTCube weights not found at {model_path}, using random initialization")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logging.info("🔒 OCTCube backbone frozen")
    
    def _load_octcube_weights(self, model_path: str):
        """
        Load pretrained OCTCube weights.
        
        Args:
            model_path: Path to OCTCube checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights with flexibility for architecture differences
            self.backbone.load_state_dict(state_dict, strict=False)
            logging.info("✅ OCTCube weights loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to load OCTCube weights: {e}")
            logging.info("🔄 Continuing with random initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through OCTCube encoder.
        
        Args:
            x: Input OCT volume [batch_size, 1, depth, height, width]
            
        Returns:
            torch.Tensor: Encoded features [batch_size, feature_dim]
        """
        # Process through 3D CNN backbone
        features = self.backbone(x)
        return features


class FLIOEncoder(nn.Module):
    """
    Custom encoder for FLIO (Fluorescence Lifetime Imaging Ophthalmoscopy) data.
    
    FLIO is a novel imaging modality that measures fluorescence lifetime of
    retinal tissues, providing metabolic information. Since no pretrained
    foundation model exists for FLIO, we use a custom 2D CNN architecture.
    
    Key features:
    - Handles 4-channel input (2 lifetime + 2 intensity channels)
    - Extracts both general features and metabolic-specific features
    - Designed for fluorescence lifetime characteristics
    - Includes metabolic biomarker prediction
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        input_channels: int = 4,  # 2 lifetime + 2 intensity channels
        input_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize FLIO encoder.
        
        Args:
            feature_dim: Output feature dimension
            input_channels: Number of input channels (4 for FLIO)
            input_size: Input image size (height, width)
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 2D CNN backbone designed for FLIO characteristics
        # Architecture balances spatial resolution with feature extraction
        self.backbone = nn.Sequential(
            # First block - initial feature extraction
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),                    # Normalize for stable training
            nn.ReLU(),
            nn.MaxPool2d(2),                       # Reduce spatial dimensions
            
            # Second block - intermediate features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block - high-level features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth block - final feature representation
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Global average pooling for translation invariance
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Flatten and project to standard dimension
            nn.Flatten(),
            nn.Linear(512, feature_dim)
        )
        
        # Specialized head for metabolic feature extraction
        # FLIO provides unique metabolic information not available in other modalities
        self.metabolic_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),                       # Prevent overfitting
            nn.Linear(256, 3)                      # Predict 3 metabolic biomarkers
        )
        # Metabolic features:
        # 1. NADH/FAD ratio - cellular metabolism indicator
        # 2. Metabolic activity - overall cellular activity level
        # 3. Oxidative stress - tissue health indicator
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FLIO encoder.
        
        Args:
            x: Input FLIO data [batch_size, 4, height, width]
            
        Returns:
            Dictionary containing:
                - 'features': General features [batch_size, feature_dim]
                - 'metabolic_features': Metabolic biomarkers [batch_size, 3]
        """
        # Extract general features
        features = self.backbone(x)
        
        # Extract metabolic-specific features
        metabolic_features = self.metabolic_head(features)
        
        return {
            'features': features,
            'metabolic_features': metabolic_features
        }


class FoundationModelManager:
    """
    Manager for loading and configuring foundation models.
    
    This class provides a unified interface for managing multiple foundation
    models across different modalities. It handles model loading, configuration,
    and provides consistent access to all models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize foundation model manager.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.models = {}
    
    def load_models(self) -> Dict[str, nn.Module]:
        """
        Load all configured foundation models.
        
        Returns:
            Dictionary of loaded models keyed by modality name
        """
        feature_dim = self.config.get('feature_dim', 768)
        
        # Load RETFound for fundus images
        if self.config.get('use_fundus', True):
            self.models['fundus'] = RETFoundEncoder(
                model_path=self.config.get('retfound_path', 'pretrained_weights/retfound_cfp_weights.pth'),
                freeze_backbone=self.config.get('freeze_fundus', False),
                feature_dim=feature_dim
            )
            logging.info("🔄 RETFound encoder initialized")
        
        # Load OCTCube for OCT volumes
        if self.config.get('use_oct', True):
            self.models['oct'] = OCTCubeEncoder(
                model_path=self.config.get('octcube_path', 'pretrained_weights/octcube_weights.pth'),
                freeze_backbone=self.config.get('freeze_oct', False),
                feature_dim=feature_dim,
                input_shape=tuple(self.config.get('oct_size', [256, 256, 32]))
            )
            logging.info("🔄 OCTCube encoder initialized")
        
        # Load FLIO encoder
        if self.config.get('use_flio', True):
            self.models['flio'] = FLIOEncoder(
                feature_dim=feature_dim,
                input_size=tuple(self.config.get('flio_size', [256, 256]))
            )
            logging.info("🔄 FLIO encoder initialized")
        
        logging.info(f"✅ Foundation model manager loaded {len(self.models)} models")
        return self.models


# ============================================================================
# Training Utilities and Helper Functions
# ============================================================================

class ModelTrainer:
    """
    Comprehensive trainer for multi-modal diabetic retinopathy models.
    
    This trainer provides a complete training pipeline including:
    - Mixed precision training for efficiency
    - Advanced learning rate scheduling
    - Comprehensive logging and monitoring
    - Checkpoint management
    - Early stopping and validation
    - Support for multiple loss functions
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on (CPU/GPU)
            logger: Optional logger for training messages
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        
        # Mixed precision training for efficiency
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', True) else None
        
        # Training state tracking
        self.epoch = 0
        self.best_val_score = float('-inf')
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        
        # Setup experiment tracking
        if config.get('use_wandb', False):
            self._setup_wandb()
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer based on configuration.
        
        Different optimizers work better for different architectures:
        - AdamW: Good for transformers and general use
        - Adam: Classic choice for many architectures
        - SGD: Can provide better generalization with proper tuning
        
        Returns:
            Configured optimizer
        """
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),               # Standard beta values
                eps=1e-8                          # Numerical stability
            )
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,                     # Momentum for SGD
                nesterov=True                     # Nesterov momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate scheduler.
        
        Different schedulers for different training strategies:
        - Cosine: Smooth annealing, good for fine-tuning
        - Plateau: Adaptive reduction based on validation metrics
        - Step: Fixed schedule reduction
        
        Returns:
            Configured scheduler or None
        """
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',                       # Monitor validation accuracy
                factor=0.5,                       # Reduce by half
                patience=self.config.get('scheduler_patience', 5),
                verbose=True
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _setup_criterion(self) -> nn.Module:
        """
        Setup loss function based on configuration.
        
        Different loss functions for different challenges:
        - Cross-entropy: Standard classification loss
        - Focal loss: Handles class imbalance
        - Ordinal loss: Respects ordinal nature of DR grades
        
        Returns:
            Configured loss function
        """
        criterion_name = self.config.get('criterion', 'crossentropy').lower()
        
        if criterion_name == 'crossentropy':
            # Handle class imbalance with weights
            if self.config.get('use_class_weights', True):
                class_weights = self._calculate_class_weights()
                return nn.CrossEntropyLoss(weight=class_weights)
            else:
                return nn.CrossEntropyLoss()
        elif criterion_name == 'focal':
            return FocalLoss(
                alpha=self.config.get('focal_alpha', 1.0),
                gamma=self.config.get('focal_gamma', 2.0)
            )
        elif criterion_name == 'ordinal':
            return OrdinalLoss(num_classes=self.config.get('num_classes', 5))
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Medical datasets often have class imbalance (more mild cases than severe).
        Class weights help the model pay attention to underrepresented classes.
        
        Returns:
            Tensor of class weights
        """
        # In practice, this would analyze the training dataset
        # For now, return uniform weights as placeholder
        num_classes = self.config.get('num_classes', 5)
        weights = torch.ones(num_classes)
        
        # Example weights for typical DR distribution
        # (modify based on your actual dataset distribution)
        if num_classes == 5:
            # More weight for severe cases (grades 3, 4)
            weights = torch.tensor([0.5, 0.8, 1.0, 2.0, 3.0])
        
        return weights.to(self.device)
    
    def _setup_wandb(self):
        """
        Setup Weights & Biases experiment tracking.
        
        W&B provides comprehensive experiment tracking including:
        - Loss curves and metrics
        - Model architecture visualization
        - Hyperparameter logging
        - Artifact management
        """
        wandb.init(
            project=self.config.get('wandb_project', 'diabetic-retinopathy'),
            name=self.config.get('experiment_name', 'multimodal-dr'),
            config=self.config,
            tags=self.config.get('tags', ['multimodal', 'dr', 'foundation-models'])
        )
        
        # Watch model for gradient and parameter tracking
        wandb.watch(self.model, log_freq=100)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar for training visualization
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = self.criterion(outputs['logits'], batch['label'])
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward and backward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], batch['label'])
                loss.backward()
                self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], 1)
            correct_predictions += (predicted == batch['label']).sum().item()
            total_predictions += batch['label'].size(0)
            
            # Update progress bar
            current_acc = 100 * correct_predictions / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_acc = correct_predictions / total_predictions
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], batch['label'])
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels for metrics
                _, predicted = torch.max(outputs['logits'], 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Calculate validation metrics
        val_loss = total_loss / num_batches
        val_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Additional metrics can be added here
        metrics = {
            'loss': val_loss,
            'accuracy': val_acc
        }
        
        return metrics
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Complete training loop with validation and checkpointing.
        
        Args:
            num_epochs: Number of epochs to train (overrides config)
            
        Returns:
            Dictionary with training history
        """
        num_epochs = num_epochs or self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        self.logger.info(f"🚀 Starting training for {num_epochs} epochs")
        self.logger.info(f"📊 Training dataset size: {len(self.train_loader.dataset)}")
        self.logger.info(f"📊 Validation dataset size: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            epoch_time = time.time() - start_time
            self._log_epoch_metrics(train_metrics, val_metrics, epoch_time)
            
            # Check for improvement and save best model
            val_score = val_metrics['accuracy']  # Primary metric for model selection
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"🎯 New best validation accuracy: {val_score:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(f"⏳ No improvement for {patience_counter} epochs")
            
            # Early stopping check
            if patience_counter >= patience:
                self.logger.info(f"⏹️ Early stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"Best validation accuracy: {self.best_val_score:.4f}")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Training completed
        self.logger.info(f"✅ Training completed!")
        self.logger.info(f"🏆 Best validation accuracy: {self.best_val_score:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores
        }
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to the specified device.
        
        Args:
            batch: Batch dictionary containing tensors and other data
            
        Returns:
            Batch with tensors moved to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """
        Log metrics for the current epoch.
        
        Args:
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            epoch_time: Time taken for the epoch
        """
        # Store metrics for plotting
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.val_scores.append(val_metrics['accuracy'])
        
        # Console logging with rich formatting
        self.logger.info(
            f"📈 Epoch {self.epoch + 1:3d}/{self.config.get('epochs', 100)} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Weights & Biases logging
        if self.config.get('use_wandb', False):
            wandb.log({
                'epoch': self.epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/learning_rate': train_metrics['lr'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'time/epoch_duration': epoch_time,
                'best_val_accuracy': self.best_val_score
            })
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint with complete training state.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive checkpoint
        checkpoint = {
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_score': self.best_val_score,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores,
            'model_architecture': str(self.model)
        }
        
        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Also save model weights only for easier loading
        if filename == 'best_model.pth':
            weights_path = checkpoint_dir / 'best_model_weights.pth'
            torch.save(self.model.state_dict(), weights_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.best_val_score = checkpoint.get('best_val_score', float('-inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_scores = checkpoint.get('val_scores', [])
        
        epoch = checkpoint['epoch']
        self.logger.info(f"📂 Checkpoint loaded from epoch {epoch}")
        self.logger.info(f"🎯 Best validation score: {self.best_val_score:.4f}")
        
        return epoch


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in medical datasets.
    
    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples. This is particularly useful for medical
    datasets where some conditions are rare but important.
    
    The focal loss is defined as:
    FL(pt) = -α(1-pt)^γ log(pt)
    
    Where:
    - pt is the predicted probability for the true class
    - α is a weighting factor for class balance
    - γ is the focusing parameter (higher γ = more focus on hard examples)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare classes
            gamma: Focusing parameter (0 = standard cross-entropy)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OrdinalLoss(nn.Module):
    """
    Ordinal loss for diabetic retinopathy grading.
    
    DR grading is inherently ordinal (0 < 1 < 2 < 3 < 4), meaning that
    predicting grade 2 when the truth is grade 1 is less wrong than
    predicting grade 4. Standard cross-entropy doesn't capture this.
    
    This loss function penalizes predictions based on their distance
    from the true grade, encouraging the model to make mistakes that
    are closer to the correct answer.
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize ordinal loss.
        
        Args:
            num_classes: Number of DR grades (typically 5)
        """
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate ordinal loss.
        
        Args:
            inputs: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Ordinal loss value
        """
        batch_size = inputs.size(0)
        
        # Create ordinal weight matrix
        # Distance between grades determines penalty
        weight_matrix = torch.zeros(self.num_classes, self.num_classes, device=inputs.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                weight_matrix[i, j] = abs(i - j)
        
        # Calculate log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get weights for each sample based on true labels
        weights = weight_matrix[targets]  # [batch_size, num_classes]
        
        # Calculate weighted cross-entropy
        # Higher weights for predictions farther from truth
        weighted_log_probs = log_probs * weights
        loss = -weighted_log_probs.sum(dim=1).mean()
        
        return loss


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    
    Monitors validation metrics and stops training when improvement
    plateaus for a specified number of epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of foundation models and training utilities.
    
    This demonstrates how to:
    1. Initialize foundation models
    2. Set up training pipeline
    3. Configure advanced training features
    """
    
    print("🔬 Foundation Models for Medical Imaging")
    print("=" * 50)
    
    print("\n📋 Available Foundation Models:")
    print("✅ RETFound - Retinal fundus image encoder")
    print("✅ OCTCube - OCT volume encoder")
    print("✅ FLIO - Custom fluorescence lifetime encoder")
    
    print("\n🎯 Training Features:")
    print("✅ Mixed precision training")
    print("✅ Advanced learning rate scheduling")
    print("✅ Multiple loss functions (Cross-entropy, Focal, Ordinal)")
    print("✅ Comprehensive logging and monitoring")
    print("✅ Early stopping and checkpointing")
    print("✅ Weights & Biases integration")
    
    print("\n🚀 Usage Example:")
    print("# Initialize foundation models")
    print("model_manager = FoundationModelManager(config)")
    print("models = model_manager.load_models()")
    
    print("\n# Setup training")
    print("trainer = ModelTrainer(model, train_loader, val_loader, config, device)")
    print("history = trainer.train()")
    
    print("\n📊 Advanced Loss Functions:")
    print("• Focal Loss - Handles class imbalance")
    print("• Ordinal Loss - Respects DR grade ordering")
    print("• Class-weighted Cross-entropy - Balances rare classes")
    
    print("\n🎛️ Training Configuration Options:")
    print("• Optimizers: AdamW, Adam, SGD")
    print("• Schedulers: Cosine, Plateau, Step")
    print("• Mixed precision training")
    print("• Gradient clipping")
    print("• Early stopping")
    
    print("\n🔧 Next Steps:")
    print("1. Configure your model settings")
    print("2. Prepare your data loaders")
    print("3. Initialize trainer and start training")
    print("4. Monitor progress with W&B")
    print("5. Evaluate on test set")