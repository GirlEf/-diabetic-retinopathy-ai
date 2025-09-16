# ============================================================================
# /src/models/self_supervised_learning.py
# Self-Supervised Learning Models for Multi-Modal Diabetic Retinopathy
# ============================================================================
"""
This module implements advanced self-supervised learning (SSL) techniques for medical imaging.
SSL is crucial for medical AI because labeled medical data is scarce and expensive to obtain,
while unlabeled medical images are abundant.

The module provides two main SSL approaches:

1. Masked Autoencoder (MAE) - Learns to reconstruct masked portions of images
   - Based on "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
   - Particularly effective for learning spatial representations in medical images
   - Helps models understand anatomical structure without requiring labels

2. Contrastive Learning - Learns by comparing different modalities of the same patient
   - Encourages similar representations for different views of the same patient
   - Particularly powerful for multi-modal medical data (fundus, OCT, FLIO)
   - Helps learn cross-modal relationships without supervision

These SSL models can be used to:
- Pre-train encoders before fine-tuning on labeled data
- Learn robust feature representations from unlabeled medical images
- Improve generalization to new hospitals/imaging equipment
- Reduce dependence on manual annotations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from typing import Dict, Tuple, Optional, List
import numpy as np
import random

class MaskedAutoEncoder(nn.Module):
    """
    Masked Autoencoder for self-supervised learning on medical images.
    
    This implementation is based on the MAE paper (He et al., 2022) but adapted
    for medical imaging. The core idea is to:
    1. Randomly mask patches of the input image (typically 75%)
    2. Encode only the visible patches
    3. Decode to reconstruct the original image
    4. Compute loss only on the masked patches
    
    For medical imaging, this is particularly valuable because:
    - Models learn to understand spatial relationships in anatomy
    - No manual annotations required
    - Can pre-train on large amounts of unlabeled medical data
    - Learned representations transfer well to downstream tasks
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        input_size: Tuple[int, int],
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16
    ):
        """
        Initialize the Masked Autoencoder.
        
        Args:
            encoder: Pre-trained encoder (e.g., ViT, Swin Transformer)
            input_size: Input image dimensions (height, width)
            patch_size: Size of image patches (typically 16x16)
            mask_ratio: Fraction of patches to mask (0.75 = 75%)
            decoder_embed_dim: Hidden dimension for decoder
            decoder_depth: Number of transformer layers in decoder
            decoder_num_heads: Number of attention heads in decoder
        """
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Calculate total number of patches in the image
        # For a 224x224 image with 16x16 patches: (224/16) * (224/16) = 196 patches
        self.num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)
        
        # Decoder components
        # Projects encoder features to decoder dimension
        self.decoder_embed = nn.Linear(encoder.config.hidden_size, decoder_embed_dim)
        
        # Learnable mask token - represents masked patches during reconstruction
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embeddings for decoder (includes +1 for class token)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        
        # Final decoder layers
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # Predicts RGB pixel values for each patch (patch_size^2 * 3 channels)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3)
        
        # Initialize all weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize model weights following best practices.
        
        Proper initialization is crucial for training stability,
        especially in self-supervised learning where gradients
        can be unstable initially.
        """
        # Initialize positional embeddings with small random values
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Apply weight initialization to all modules
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """
        Initialize weights for different layer types.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Standard initialization for layer norm
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images into patches for processing.
        
        This transforms a batch of images into a sequence of patches,
        which is the standard input format for Vision Transformers.
        
        Args:
            imgs: Input images [batch_size, 3, height, width]
            
        Returns:
            Patches [batch_size, num_patches, patch_size^2 * 3]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        # Calculate patch grid dimensions
        h = w = imgs.shape[2] // p
        
        # Reshape image into patches
        # [B, 3, H, W] -> [B, 3, h, p, w, p] -> [B, h, w, p, p, 3] -> [B, h*w, p^2*3]
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images for visualization.
        
        This reverses the patchify operation to reconstruct
        the full image from patches.
        
        Args:
            x: Patches [batch_size, num_patches, patch_size^2 * 3]
            
        Returns:
            Images [batch_size, 3, height, width]
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        # Reshape patches back to image
        # [B, h*w, p^2*3] -> [B, h, w, p, p, 3] -> [B, 3, h, p, w, p] -> [B, 3, H, W]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.shape[0], 3, h * p, h * p)
        return x
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random masking to patches.
        
        This is the core of MAE - we randomly mask a high percentage of patches
        and only process the visible ones through the encoder. This forces
        the model to learn meaningful representations from partial information.
        
        Args:
            x: Input patches [batch_size, num_patches, embed_dim]
            
        Returns:
            x_masked: Visible patches only [batch_size, num_visible, embed_dim]
            mask: Binary mask [batch_size, num_patches] (0=keep, 1=remove)
            ids_restore: Indices to restore original patch order
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))  # Number of patches to keep
        
        # Generate random noise for each patch
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise to get random ordering
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # indices to restore original order
        
        # Keep the first subset (lowest noise values)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask in original patch order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder with masking.
        
        This processes only the visible patches through the encoder,
        significantly reducing computational cost (e.g., 4x speedup with 75% masking).
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            x: Encoded features [batch_size, num_visible+1, embed_dim]
            mask: Binary mask indicating masked patches
            ids_restore: Indices for restoring patch order
        """
        # Convert image to patches and add positional embeddings
        x = self.encoder.embeddings.patch_embeddings(x)  # Assuming ViT-like structure
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        # Add positional embeddings (excluding class token)
        x = x + self.encoder.embeddings.position_embeddings[:, 1:, :]
        
        # Apply random masking - this is where the magic happens!
        x, mask, ids_restore = self.random_masking(x)
        
        # Add class token with its positional embedding
        cls_token = self.encoder.embeddings.cls_token + self.encoder.embeddings.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Process through encoder transformer blocks
        for block in self.encoder.encoder.layer:
            x = block(x)[0]
        
        # Final layer normalization
        x = self.encoder.layernorm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder to reconstruct the image.
        
        The decoder takes the encoded visible patches and mask tokens
        to reconstruct the full image. This is where the model learns
        to understand spatial relationships and context.
        
        Args:
            x: Encoded features from encoder [batch_size, num_visible+1, embed_dim]
            ids_restore: Indices to restore original patch order
            
        Returns:
            Reconstructed patches [batch_size, num_patches, patch_size^2 * 3]
        """
        # Project encoder features to decoder dimension
        x = self.decoder_embed(x)
        
        # Create mask tokens for the masked patches
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        # Combine visible patches with mask tokens
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        
        # Restore original patch order
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Re-add class token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add positional embeddings
        x = x + self.decoder_pos_embed
        
        # Process through decoder transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values for each patch
        x = self.decoder_pred(x)
        
        # Remove class token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked patches.
        
        This is a key insight of MAE - we only compute loss on the masked patches,
        which prevents the model from simply copying visible patches and forces
        it to learn meaningful representations.
        
        Args:
            imgs: Original images [batch_size, 3, height, width]
            pred: Predicted patches [batch_size, num_patches, patch_size^2 * 3]
            mask: Binary mask [batch_size, num_patches]
            
        Returns:
            Reconstruction loss (scalar)
        """
        # Convert original images to patches
        target = self.patchify(imgs)
        
        # Compute per-patch loss (Mean Squared Error)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # Only compute loss on masked patches
        # This is crucial - it prevents the model from cheating by copying visible patches
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for training.
        
        Args:
            imgs: Input images [batch_size, 3, height, width]
            
        Returns:
            Dictionary containing:
                - loss: Reconstruction loss
                - pred: Predicted patches
                - mask: Binary mask
                - latent: Encoded features
        """
        # Encode visible patches
        latent, mask, ids_restore = self.forward_encoder(imgs)
        
        # Decode to reconstruct image
        pred = self.forward_decoder(latent, ids_restore)
        
        # Compute loss on masked patches
        loss = self.forward_loss(imgs, pred, mask)
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'latent': latent
        }


class TransformerBlock(nn.Module):
    """
    Standard Transformer block for the decoder.
    
    This implements the standard transformer architecture with
    multi-head attention and feed-forward networks.
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        """
        Initialize transformer block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to input dim
            dropout: Dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class ContrastiveLearning(nn.Module):
    """
    Contrastive learning for multi-modal self-supervision.
    
    This approach is particularly powerful for medical imaging because
    it can learn relationships between different modalities (fundus, OCT, FLIO)
    without requiring labels. The key idea is:
    
    1. For the same patient, different modalities should have similar representations
    2. For different patients, representations should be dissimilar
    3. This creates a rich multi-modal representation space
    
    For example:
    - Fundus and OCT of the same patient should be similar
    - Fundus and FLIO of the same patient should be similar
    - But different patients should be dissimilar across all modalities
    """
    
    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        feature_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.1
    ):
        """
        Initialize contrastive learning model.
        
        Args:
            encoders: Dictionary of modality encoders
            feature_dim: Dimension of encoder features
            projection_dim: Dimension of projection space
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.temperature = temperature
        
        # Projection heads map features to contrastive space
        # These are typically smaller than feature_dim to create a bottleneck
        self.projectors = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, projection_dim)
            )
            for modality in encoders.keys()
        })
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss between modalities.
        
        Args:
            batch: Dictionary containing data for each modality
            
        Returns:
            Dictionary containing losses and features
        """
        features = {}
        projections = {}
        
        # Extract and project features from each modality
        for modality, data in batch.items():
            if modality in self.encoders:
                # Extract features using modality-specific encoder
                feat = self.encoders[modality](data)
                features[modality] = feat
                
                # Project to contrastive space and normalize
                # Normalization is crucial for contrastive learning
                projections[modality] = F.normalize(self.projectors[modality](feat), dim=1)
        
        # Compute contrastive losses between all modality pairs
        losses = {}
        for mod1 in projections:
            for mod2 in projections:
                if mod1 != mod2:
                    # Symmetric loss - both directions
                    loss = self.contrastive_loss(projections[mod1], projections[mod2])
                    losses[f"{mod1}_{mod2}"] = loss
        
        # Average all pairwise losses
        total_loss = sum(losses.values()) / len(losses)
        
        return {
            'total_loss': total_loss,
            'individual_losses': losses,
            'features': features,
            'projections': projections
        }
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.
        
        This is the standard contrastive loss used in SimCLR and similar methods.
        The idea is to maximize similarity between positive pairs (same patient)
        while minimizing similarity to negative pairs (different patients).
        
        Args:
            z1: Projections from first modality [batch_size, proj_dim]
            z2: Projections from second modality [batch_size, proj_dim]
            
        Returns:
            Contrastive loss (scalar)
        """
        batch_size = z1.shape[0]
        
        # Compute similarity matrix between all pairs
        # Temperature scaling controls the "sharpness" of the distribution
        sim_matrix = torch.mm(z1, z2.T) / self.temperature
        
        # Create labels - positive pairs are on the diagonal
        # For same patient, different modalities should be similar
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute cross-entropy loss
        # This encourages high similarity for positive pairs
        # and low similarity for negative pairs
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class SSLTrainer:
    """
    Self-supervised learning trainer for medical imaging.
    
    This trainer handles the training loop for SSL models,
    including proper optimization, learning rate scheduling,
    and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize SSL trainer.
        
        Args:
            model: SSL model (MAE or ContrastiveLearning)
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer - AdamW works well for SSL
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('ssl_lr', 5e-4),  # Lower LR for SSL
            weight_decay=config.get('weight_decay', 0.05)  # Strong regularization
        )
        
        # Learning rate scheduler - cosine annealing is popular for SSL
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('ssl_epochs', 50)
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - handle different model types
            if isinstance(self.model, MaskedAutoEncoder):
                # MAE training on single modality
                outputs = self.model(batch['fundus'])  # Adapt for your modality
                loss = outputs['loss']
            elif isinstance(self.model, ContrastiveLearning):
                # Contrastive learning across modalities
                outputs = self.model(batch)
                loss = outputs['total_loss']
            else:
                raise ValueError(f"Unknown model type: {type(self.model)}")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def save_checkpoint(self, epoch: int, save_path: str):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            save_path: Path to save checkpoint
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, save_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']


# Example usage and benefits
if __name__ == "__main__":
    """
    Example usage of SSL models for medical imaging.
    
    SSL is particularly valuable in medical imaging because:
    1. Labeled data is expensive and requires expert annotation
    2. Unlabeled medical images are abundant
    3. SSL can learn robust representations that generalize well
    4. Pre-trained SSL models improve downstream task performance
    """
    print("Self-Supervised Learning for Medical Imaging")
    print("=" * 50)
    
    print("Benefits of SSL in Medical Imaging:")
    print("• Reduces dependence on labeled data")
    print("• Learns robust anatomical representations")
    print("• Improves generalization across hospitals")
    print("• Enables multi-modal learning without labels")
    print("• Provides better initialization for downstream tasks")
    
    print("\nImplemented SSL Methods:")
    print("• Masked Autoencoder (MAE) - Spatial representation learning")
    print("• Contrastive Learning - Multi-modal relationship learning")
    
    print("\nRecommended Usage:")
    print("1. Pre-train on large unlabeled medical image datasets")
    print("2. Fine-tune on smaller labeled datasets for specific tasks")
    print("3. Use learned representations for downstream classification")
    print("4. Combine with foundation models for best results")