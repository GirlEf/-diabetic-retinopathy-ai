#!/usr/bin/env python3
"""
Debug script for MAE training issues
"""

import torch
import torch.nn as nn
import time
import gc
import traceback
from pathlib import Path

def debug_mae_training():
    """Debug MAE training step by step"""
    
    print("=== MAE TRAINING DEBUG ===")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    # Test simple tensor operations
    print("\n1. Testing basic tensor operations...")
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"✅ Tensor created: {x.shape}")
        
        # Test patch embedding
        patch_embed = nn.Conv2d(3, 512, kernel_size=16, stride=16).to(device)
        patches = patch_embed(x)
        print(f"✅ Patch embedding: {patches.shape}")
        
        # Test flattening
        patches_flat = patches.flatten(2).transpose(1, 2)
        print(f"✅ Flattened: {patches_flat.shape}")
        
        del x, patches, patches_flat, patch_embed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Basic tensor operations failed: {e}")
        traceback.print_exc()
        return False
    
    # Test transformer operations
    print("\n2. Testing transformer operations...")
    try:
        batch_size = 2
        seq_len = 196  # 14x14 patches
        d_model = 512
        
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        print(f"✅ Input tensor: {x.shape}")
        
        # Test transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        print("✅ Encoder layer created")
        
        # Test forward pass
        output = encoder_layer(x)
        print(f"✅ Encoder output: {output.shape}")
        
        del x, output, encoder_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Transformer operations failed: {e}")
        traceback.print_exc()
        return False
    
    # Test masking operations
    print("\n3. Testing masking operations...")
    try:
        N, L, D = 2, 196, 512
        x = torch.randn(N, L, D).to(device)
        mask_ratio = 0.75
        
        len_keep = max(1, int(L * (1 - mask_ratio)))
        print(f"✅ Will keep {len_keep} patches out of {L}")
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        print(f"✅ Masked tensor: {x_masked.shape}")
        
        del x, x_masked, noise, ids_shuffle, ids_restore, ids_keep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Masking operations failed: {e}")
        traceback.print_exc()
        return False
    
    # Test memory management
    print("\n4. Testing memory management...")
    try:
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            print(f"Initial memory: {initial_memory / 1e9:.3f} GB")
            
            # Create some tensors
            tensors = []
            for i in range(10):
                tensor = torch.randn(100, 100, 100).to(device)
                tensors.append(tensor)
            
            current_memory = torch.cuda.memory_allocated()
            print(f"After creating tensors: {current_memory / 1e9:.3f} GB")
            
            # Delete tensors
            del tensors
            torch.cuda.empty_cache()
            gc.collect()
            
            final_memory = torch.cuda.memory_allocated()
            print(f"After cleanup: {final_memory / 1e9:.3f} GB")
            
            if final_memory <= initial_memory * 1.1:  # Allow 10% tolerance
                print("✅ Memory management working correctly")
            else:
                print("⚠️ Memory cleanup may not be working properly")
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        traceback.print_exc()
    
    # Test DataLoader operations
    print("\n5. Testing DataLoader operations...")
    try:
        from torch.utils.data import Dataset, DataLoader
        
        # Create dummy dataset
        class DummyDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'fundus': torch.randn(3, 224, 224),
                    'oct': torch.randn(1, 256, 256),
                    'flio': torch.randn(2, 256, 256),
                    'participant_id': f'patient_{idx}'
                }
        
        dataset = DummyDataset(50)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        print("✅ DataLoader created")
        
        # Test iteration
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        print(f"✅ Successfully processed {batch_count} batches")
        
        del dataset, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n=== DEBUG COMPLETED ===")
    print("✅ All basic operations working correctly")
    print("If MAE training still hangs, the issue is likely in the training loop logic")
    
    return True

def test_mae_model_creation():
    """Test creating a simple MAE model"""
    
    print("\n=== TESTING MAE MODEL CREATION ===")
    
    try:
        # Simple MAE model
        class SimpleMAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, 512, kernel_size=16, stride=16)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
                    num_layers=2
                )
                self.decoder = nn.Linear(512, 3 * 16 * 16)
            
            def forward(self, x):
                x = self.patch_embed(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleMAE().to(device)
        
        print(f"✅ MAE model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        output = model(x)
        print(f"✅ Forward pass successful: {output.shape}")
        
        del model, x, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ MAE model creation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running MAE training debug...")
    
    # Run basic debug
    basic_ok = debug_mae_training()
    
    # Test MAE model creation
    mae_ok = test_mae_model_creation()
    
    if basic_ok and mae_ok:
        print("\n🎉 All tests passed! MAE training should work.")
        print("If you still experience hanging, check:")
        print("1. GPU memory usage during training")
        print("2. DataLoader batch processing")
        print("3. CUDA synchronization issues")
    else:
        print("\n❌ Some tests failed. Check the error messages above.") 