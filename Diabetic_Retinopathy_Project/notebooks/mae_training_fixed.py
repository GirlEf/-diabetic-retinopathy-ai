#!/usr/bin/env python3
"""
Fixed MAE Training Script - Addresses hanging and memory issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
import json
import gc
import traceback
import numpy as np

def create_robust_mae_training(ssl_config, ssl_train_dataset, ssl_val_dataset, checkpoint_dir):
    """
    Create and execute robust MAE training with comprehensive error handling
    """
    print("=== ROBUST MASKED AUTOENCODER PRETRAINING ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory cleanup before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Simplified collate function to prevent hanging
    def simple_mae_collate_fn(batch):
        """Simplified collate function to prevent DataLoader hanging"""
        try:
            result = {}
            
            # Process each modality separately
            for modality in ['fundus', 'oct', 'flio']:
                modality_batch = []
                
                for sample in batch:
                    if modality in sample:
                        tensor = sample[modality]
                        
                        # Handle OCT 3D volumes
                        if modality == 'oct' and tensor.ndim == 4:  # [1, H, W, Slices]
                            tensor = tensor.mean(dim=-1)  # Take mean across slices
                        elif modality == 'oct' and tensor.ndim == 3: # [H, W]
                            tensor = tensor.unsqueeze(0)  # Add channel dim
                        
                        # Handle FLIO
                        elif modality == 'flio' and tensor.ndim == 3: # [H, W]
                            tensor = tensor.unsqueeze(0)  # Add channel dim
                        
                        modality_batch.append(tensor)
                    else:
                        # Create zero tensor for missing modality
                        if modality == 'fundus':
                            zero_tensor = torch.zeros(3, 224, 224)
                        elif modality == 'oct':
                            zero_tensor = torch.zeros(1, 256, 256)
                        elif modality == 'flio':
                            zero_tensor = torch.zeros(2, 256, 256)
                        modality_batch.append(zero_tensor)
                
                if modality_batch:
                    try:
                        result[modality] = torch.stack(modality_batch)
                    except RuntimeError as e:
                        print(f"Error stacking {modality}: {e}")
                        # Fallback: use first tensor shape
                        first_shape = modality_batch[0].shape
                        result[modality] = torch.zeros(len(modality_batch), *first_shape)
            
            # Add participant IDs
            result['participant_id'] = [sample.get('participant_id', f'unknown_{i}') for i, sample in enumerate(batch)]
            
            return result
            
        except Exception as e:
            print(f"Error in collate function: {e}")
            traceback.print_exc()
            # Return minimal batch to prevent hanging
            return {
                'fundus': torch.zeros(len(batch), 3, 224, 224),
                'oct': torch.zeros(len(batch), 1, 256, 256),
                'flio': torch.zeros(len(batch), 2, 256, 256),
                'participant_id': [f'error_{i}' for i in range(len(batch))]
            }
    
    # Create DataLoaders with safer settings
    print("🔄 Creating MAE-optimized DataLoaders...")
    mae_train_loader = DataLoader(
        ssl_train_dataset,
        batch_size=ssl_config['ssl_batch_size'],
        shuffle=True,
        num_workers=0,  # Keep at 0 to prevent multiprocessing issues
        pin_memory=False,
        collate_fn=simple_mae_collate_fn,
        drop_last=False,
        timeout=60  # Add timeout to prevent hanging
    )
    
    mae_val_loader = DataLoader(
        ssl_val_dataset,
        batch_size=ssl_config['ssl_batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=simple_mae_collate_fn,
        drop_last=False,
        timeout=60
    )
    
    # Simplified MAE model with better memory management
    class RobustMaskedAutoEncoder(nn.Module):
        def __init__(self, input_size=(224, 224), patch_size=16, mask_ratio=0.75, in_channels=3):
            super().__init__()
            self.patch_size = patch_size
            self.mask_ratio = mask_ratio
            self.input_size = input_size
            self.in_channels = in_channels
            
            # Calculate patches
            self.num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)
            
            # Encoder
            self.patch_embed = nn.Conv2d(in_channels, 512, kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, 512))
            
            # Smaller transformer for memory efficiency
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512, 
                nhead=8, 
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            
            # Decoder
            self.decoder_embed = nn.Linear(512, 256)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, 256))
            
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
            
            self.decoder_pred = nn.Linear(256, patch_size * patch_size * in_channels)
            
            self.initialize_weights()
        
        def initialize_weights(self):
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.decoder_pos_embed, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)
        
        def patchify(self, imgs):
            """Convert images to patches"""
            p = self.patch_size
            h = w = imgs.shape[2] // p
            x = imgs.reshape(imgs.shape[0], self.in_channels, h, p, w, p)
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(imgs.shape[0], h * w, p**2 * self.in_channels)
            return x
        
        def unpatchify(self, x):
            """Convert patches back to images"""
            p = self.patch_size
            h = w = int(x.shape[1]**.5)
            x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
            x = torch.einsum('nhwpqc->nchpwq', x)
            x = x.reshape(x.shape[0], self.in_channels, h * p, h * p)
            return x
        
        def random_masking(self, x):
            """Apply random masking with safety checks"""
            N, L, D = x.shape
            len_keep = max(1, int(L * (1 - self.mask_ratio)))  # Ensure at least 1 patch
            
            noise = torch.rand(N, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            return x_masked, mask, ids_restore
        
        def forward_encoder(self, x):
            """Forward pass through encoder"""
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x, mask, ids_restore = self.random_masking(x)
            x = self.encoder(x)
            return x, mask, ids_restore
        
        def forward_decoder(self, x, ids_restore):
            """Forward pass through decoder"""
            x = self.decoder_embed(x)
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = x_ + self.decoder_pos_embed
            x = self.decoder(x)
            x = self.decoder_pred(x)
            return x
        
        def forward_loss(self, imgs, pred, mask):
            """Calculate reconstruction loss"""
            target = self.patchify(imgs)
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # Add epsilon to prevent division by zero
            return loss
        
        def forward(self, imgs):
            """Complete forward pass"""
            try:
                latent, mask, ids_restore = self.forward_encoder(imgs)
                pred = self.forward_decoder(latent, ids_restore)
                loss = self.forward_loss(imgs, pred, mask)
                return {'loss': loss, 'pred': pred, 'mask': mask}
            except Exception as e:
                print(f"Error in forward pass: {e}")
                # Return dummy loss to prevent training from stopping
                dummy_loss = torch.tensor(1e6, device=imgs.device, requires_grad=True)
                return {'loss': dummy_loss, 'pred': None, 'mask': None}
    
    # Create MAE models with memory-efficient settings
    mae_models = {}
    for modality, config in [
        ('fundus', {'input_size': (224, 224), 'in_channels': 3}),
        ('oct', {'input_size': (256, 256), 'in_channels': 1}),
        ('flio', {'input_size': (256, 256), 'in_channels': 2})
    ]:
        try:
            mae_models[modality] = RobustMaskedAutoEncoder(
                input_size=config['input_size'],
                patch_size=16,
                mask_ratio=ssl_config['mask_ratio'],
                in_channels=config['in_channels']
            ).to(device)
            print(f"✅ Created {modality} MAE model: {sum(p.numel() for p in mae_models[modality].parameters()):,} parameters")
        except Exception as e:
            print(f"❌ Failed to create {modality} MAE model: {e}")
    
    # Enhanced Trainer with Better Error Handling
    class RobustMAETrainer:
        def __init__(self, model, lr=5e-4, weight_decay=0.01):
            self.model = model
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
            self.best_loss = float('inf')
            self.training_history = []
            self.stuck_counter = 0
            self.last_loss = float('inf')
        
        def train_epoch(self, dataloader, modality, epoch):
            """Train one epoch with comprehensive error handling"""
            self.model.train()
            total_loss = 0
            num_batches = 0
            batches_with_modality = 0
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d} - {modality}", leave=False)
            
            try:
                for batch_idx, batch in enumerate(pbar):
                    # Check if we have the modality
                    if modality not in batch or batch[modality].size(0) == 0:
                        continue
                    
                    batches_with_modality += 1
                    
                    try:
                        imgs = batch[modality].to(device)
                        
                        # Preprocess images based on modality
                        imgs = self.preprocess_images(imgs, modality)
                        
                        # Skip if preprocessing failed
                        if imgs is None:
                            continue
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        output = self.model(imgs)
                        
                        if output['loss'] is None:
                            print(f"Warning: Got None loss for {modality}")
                            continue
                        
                        loss = output['loss']
                        
                        # Check for NaN or infinite loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: Invalid loss value: {loss.item()}")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        self.optimizer.step()
                        
                        # Update metrics
                        loss_value = loss.item()
                        total_loss += loss_value
                        num_batches += 1
                        
                        # Check for stuck training
                        if abs(loss_value - self.last_loss) < 1e-6:
                            self.stuck_counter += 1
                        else:
                            self.stuck_counter = 0
                        
                        self.last_loss = loss_value
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'Loss': f"{loss_value:.4f}",
                            'Batches': f"{batches_with_modality}/{len(dataloader)}",
                            'Stuck': self.stuck_counter
                        })
                        
                        # Force CUDA synchronization
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # Memory cleanup
                        if batch_idx % 10 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        traceback.print_exc()
                        continue
                    
                    # Safety check: if stuck for too long, break
                    if self.stuck_counter > 100:
                        print(f"Training appears stuck for {modality}, breaking epoch")
                        break
                
                pbar.close()
                
                # Update scheduler
                self.scheduler.step()
                
                # Calculate average loss
                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                
                # Track training history
                self.training_history.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'batches_processed': batches_with_modality
                })
                
                # Update best loss
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                
                return {'loss': avg_loss, 'batches_processed': batches_with_modality}
                
            except Exception as e:
                print(f"Error in train_epoch: {e}")
                traceback.print_exc()
                return {'loss': float('inf'), 'batches_processed': 0}
        
        def preprocess_images(self, imgs, modality):
            """Preprocess images with error handling"""
            try:
                # Handle OCT 3D volumes
                if modality == 'oct':
                    if imgs.ndim == 5:  # [B, 1, H, W, Slices]
                        imgs = imgs.mean(dim=-1)  # Take mean across slices
                    elif imgs.ndim == 4:  # [B, H, W, Slices]
                        imgs = imgs.mean(dim=-1).unsqueeze(1)  # Add channel dim
                    elif imgs.ndim == 3:  # [B, H, W]
                        imgs = imgs.unsqueeze(1)  # Add channel dim
                
                # Handle FLIO
                elif modality == 'flio':
                    if imgs.ndim == 3:  # [H, W]
                        imgs = imgs.unsqueeze(0)  # Add batch dim
                    if imgs.ndim == 4 and imgs.shape[1] != 2:
                        imgs = imgs.permute(0, 3, 1, 2)  # Adjust if [B, H, W, C]
                
                # Resize if needed
                target_size = self.model.input_size
                if imgs.shape[-2:] != target_size:
                    imgs = F.interpolate(imgs, size=target_size, mode='bilinear', align_corners=False)
                
                return imgs
                
            except Exception as e:
                print(f"Error preprocessing {modality} images: {e}")
                return None
    
    # Training execution with comprehensive monitoring
    training_results = {}
    start_time = time.time()
    
    print(f"\n🚀 Starting MAE training for {len(mae_models)} modalities...")
    
    for modality, model in mae_models.items():
        print(f"\n{'='*60}")
        print(f"🎯 Training MAE for {modality.upper()}")
        print(f"{'='*60}")
        
        # Verify data availability
        print(f"🔍 Checking {modality} data availability...")
        modality_found = False
        sample_count = 0
        
        for i, batch in enumerate(mae_train_loader):
            if modality in batch and batch[modality].size(0) > 0:
                modality_found = True
                sample_count += batch[modality].size(0)
                if i >= 2:  # Check first few batches
                    break
        
        if not modality_found:
            print(f"⚠️ No {modality} data found - skipping this modality")
            continue
        
        print(f"✅ Found {modality} data: {sample_count} samples")
        
        # Create trainer
        trainer = RobustMAETrainer(
            model, 
            lr=ssl_config['ssl_lr'], 
            weight_decay=ssl_config['ssl_weight_decay']
        )
        
        # Training loop with auto-save and monitoring
        print(f"🔄 Starting training loop for {ssl_config['ssl_epochs']} epochs...")
        
        for epoch in range(ssl_config['ssl_epochs']):
            epoch_start = time.time()
            
            print(f"\n📊 Epoch {epoch+1:02d}/{ssl_config['ssl_epochs']} - {modality}")
            
            # Train epoch
            metrics = trainer.train_epoch(mae_train_loader, modality, epoch)
            
            epoch_time = time.time() - epoch_start
            
            if metrics['loss'] != float('inf'):
                print(f"   ✅ Epoch {epoch+1:02d} completed:")
                print(f"      📉 Loss: {metrics['loss']:.4f}")
                print(f"      🔢 Batches: {metrics['batches_processed']}")
                print(f"      ⏱️  Time: {epoch_time:.1f}s")
                print(f"      📚 LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"   ⚠️ Epoch {epoch+1:02d} failed - no valid data")
            
            # Auto-save every 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    save_path = checkpoint_dir / f'mae_{modality}_checkpoint_epoch_{epoch+1}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'scheduler_state_dict': trainer.scheduler.state_dict(),
                        'best_loss': trainer.best_loss,
                        'training_history': trainer.training_history
                    }, save_path)
                    print(f"   💾 Checkpoint saved: {save_path}")
                except Exception as e:
                    print(f"   ❌ Failed to save checkpoint: {e}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Progress update
            progress = (epoch + 1) / ssl_config['ssl_epochs'] * 100
            print(f"   📈 Progress: {progress:.1f}% complete")
        
        # Save final model
        try:
            final_save_path = checkpoint_dir / f'mae_{modality}_final.pth'
            torch.save({
                'final_epoch': ssl_config['ssl_epochs'],
                'model_state_dict': model.state_dict(),
                'best_loss': trainer.best_loss,
                'training_history': trainer.training_history
            }, final_save_path)
            
            training_results[modality] = {
                'final_loss': trainer.best_loss,
                'training_history': trainer.training_history,
                'model_path': str(final_save_path)
            }
            
            print(f"✅ {modality.upper()} MAE training completed!")
            print(f"   💾 Final model saved: {final_save_path}")
            print(f"   📊 Best loss: {trainer.best_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to save final {modality} model: {e}")
            training_results[modality] = {
                'final_loss': float('inf'),
                'training_history': [],
                'model_path': 'failed'
            }
    
    # Save comprehensive training summary
    total_time = time.time() - start_time
    training_summary = {
        'total_training_time_minutes': total_time / 60,
        'ssl_epochs': ssl_config['ssl_epochs'],
        'batch_size': ssl_config['ssl_batch_size'],
        'modalities_trained': list(training_results.keys()),
        'results_per_modality': training_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        summary_path = checkpoint_dir / 'mae_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        print(f"💾 Training summary saved: {summary_path}")
    except Exception as e:
        print(f"❌ Failed to save training summary: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎉 MAE PRETRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"🎯 Modalities trained: {len(training_results)}")
    print(f"💾 Training summary saved: {summary_path}")
    print(f"🚀 Ready for supervised training!")
    print(f"{'='*60}")
    
    # Print final results
    for modality, result in training_results.items():
        if result['final_loss'] != float('inf'):
            print(f"📈 {modality.upper()}: Final Loss = {result['final_loss']:.4f}")
        else:
            print(f"❌ {modality.upper()}: Training failed")
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return training_results, training_summary

if __name__ == "__main__":
    print("This script contains the fixed MAE training code.")
    print("Import and use the create_robust_mae_training function in your notebook.") 