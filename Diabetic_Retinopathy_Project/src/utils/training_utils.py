# ============================================================================
# /src/utils/training_utils.py
# Training Utilities and Helper Functions
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
import wandb
from pathlib import Path
import json
from tqdm import tqdm
import time

class ModelTrainer:
    """Comprehensive trainer for multi-modal DR models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', True) else None
        
        # Training state
        self.epoch = 0
        self.best_val_score = float('-inf')
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        
        # Setup logging
        if config.get('use_wandb', False):
            self._setup_wandb()
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
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
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
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
                mode='max',
                factor=0.5,
                patience=self.config.get('scheduler_patience', 5),
                verbose=True
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            return None
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function"""
        criterion_name = self.config.get('criterion', 'crossentropy').lower()
        
        if criterion_name == 'crossentropy':
            # Handle class imbalance
            if self.config.get('use_class_weights', True):
                # Calculate class weights from training data
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
        """Calculate class weights for imbalanced dataset"""
        # This would typically be done on the dataset
        # For now, return uniform weights
        num_classes = self.config.get('num_classes', 5)
        return torch.ones(num_classes).to(self.device)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.config.get('wandb_project', 'diabetic-retinopathy'),
            name=self.config.get('experiment_name', 'multimodal-dr'),
            config=self.config
        )
        wandb.watch(self.model)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = self.criterion(outputs['logits'], batch['label'])
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], batch['label'])
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Accuracy calculation
            _, predicted = torch.max(outputs['logits'], 1)
            correct_predictions += (predicted == batch['label']).sum().item()
            total_predictions += batch['label'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
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
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], batch['label'])
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels
                _, predicted = torch.max(outputs['logits'], 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / num_batches
        val_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Calculate additional metrics (if metrics module available)
        metrics = {
            'loss': val_loss,
            'accuracy': val_acc
        }
        
        return metrics
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Full training loop"""
        num_epochs = num_epochs or self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            self._log_epoch_metrics(train_metrics, val_metrics, epoch_time)
            
            # Check for improvement
            val_score = val_metrics['accuracy']  # or any other metric
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Training completed
        self.logger.info(f"Training completed. Best validation score: {self.best_val_score:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores
        }
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Log metrics for the epoch"""
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.val_scores.append(val_metrics['accuracy'])
        
        # Console logging
        self.logger.info(
            f"Epoch {self.epoch + 1} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Wandb logging
        if self.config.get('use_wandb', False):
            wandb.log({
                'epoch': self.epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/lr': train_metrics['lr'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'epoch_time': epoch_time
            })
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './models/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        self.logger.info(f"Checkpoint saved: {checkpoint_dir / filename}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_score = checkpoint.get('best_val_score', float('-inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_scores = checkpoint.get('val_scores', [])
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from epoch {epoch}")
        
        return epoch


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OrdinalLoss(nn.Module):
    """Ordinal loss for DR grading"""
    
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Ordinal loss that penalizes distant predictions more
        """
        import torch.nn.functional as F
        batch_size = inputs.size(0)
        
        # Create ordinal weight matrix
        weight_matrix = torch.zeros(self.num_classes, self.num_classes, device=inputs.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                weight_matrix[i, j] = abs(i - j)
        
        # Calculate weighted cross entropy
        log_probs = F.log_softmax(inputs, dim=1)
        weights = weight_matrix[targets]  # [batch_size, num_classes]
        
        weighted_log_probs = log_probs * weights
        loss = -weighted_log_probs.sum(dim=1).mean()
        
        return loss