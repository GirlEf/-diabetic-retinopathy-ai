# ============================================================================
# /src/data/transforms.py
# Data Augmentation and Transforms for Medical Imaging
# ============================================================================
"""
This module provides data augmentation and transformation capabilities for multi-modal 
medical imaging data. It handles three types of medical imaging modalities:

1. Fundus images - 2D color retinal photographs
2. OCT volumes - 3D cross-sectional retinal scans  
3. FLIO data - 4-channel fluorescence lifetime imaging

The transforms include geometric augmentations (flips, rotations), intensity 
modifications (brightness/contrast), and noise additions to improve model 
generalization while preserving medical image characteristics.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from typing import Dict
import cv2

class MultiChannelHorizontalFlip(A.HorizontalFlip):
    """
    Custom horizontal flip transformation that properly handles multi-channel data.
    
    Standard Albumentations HorizontalFlip works well for 3-channel RGB images,
    but FLIO data has 4 channels and requires special handling to maintain
    channel relationships during flipping operations.
    """
    
    def apply(self, img, **params):
        """
        Apply horizontal flip to single or multi-channel images.
        
        Args:
            img: Input image array of shape (H, W) or (H, W, C)
            **params: Additional parameters (unused but required by Albumentations)
            
        Returns:
            np.ndarray: Horizontally flipped image
        """
        if len(img.shape) == 3:
            # For multi-channel images (like FLIO with 4 channels)
            # Flip along axis 1 (width dimension) while preserving channel relationships
            return np.flip(img, axis=1)
        else:
            # For single channel images, use OpenCV's optimized flip function
            # Flag 1 indicates horizontal flip
            return cv2.flip(img, 1)

class MultiChannelVerticalFlip(A.VerticalFlip):
    """
    Custom vertical flip transformation that properly handles multi-channel data.
    
    Similar to horizontal flip but flips along the height dimension.
    This is particularly important for FLIO data where spatial relationships
    between channels must be maintained.
    """
    
    def apply(self, img, **params):
        """
        Apply vertical flip to single or multi-channel images.
        
        Args:
            img: Input image array of shape (H, W) or (H, W, C)
            **params: Additional parameters (unused but required by Albumentations)
            
        Returns:
            np.ndarray: Vertically flipped image
        """
        if len(img.shape) == 3:
            # For multi-channel images, flip along axis 0 (height dimension)
            return np.flip(img, axis=0)
        else:
            # For single channel images, use OpenCV's flip function
            # Flag 0 indicates vertical flip
            return cv2.flip(img, 0)

class MultiModalTransforms:
    """
    Comprehensive data augmentation pipeline for multi-modal medical imaging.
    
    This class provides modality-specific augmentation strategies:
    - Fundus: Standard 2D image augmentations (rotation, flip, brightness, blur)
    - OCT: Conservative 3D volume augmentations (slice-wise processing)
    - FLIO: Multi-channel aware augmentations (geometric only, preserving lifetime values)
    
    Different augmentation intensities are used based on medical imaging best practices
    to avoid introducing artifacts that could mislead clinical interpretation.
    """
    
    def __init__(self, augmentation_prob: float = 0.5, mode: str = 'train'):
        """
        Initialize the multi-modal transforms.
        
        Args:
            augmentation_prob: Base probability for applying augmentations (0.0 to 1.0)
            mode: Either 'train' (with augmentations) or 'val'/'test' (minimal transforms)
        """
        self.augmentation_prob = augmentation_prob
        self.mode = mode
        
        # Initialize transform pipelines for each modality
        # Each modality has different augmentation strategies based on their characteristics
        self.fundus_transforms = self._get_fundus_transforms()
        self.oct_transforms = self._get_oct_transforms()
        self.flio_transforms = self._get_flio_transforms()
    
    def _get_fundus_transforms(self):
        """
        Create augmentation pipeline for fundus (retinal) images.
        
        Fundus images are 2D color photographs of the retina. They can handle
        more aggressive augmentations since they're photographic in nature.
        
        Augmentations included:
        - Geometric: Rotation, flipping (preserves anatomical relationships)
        - Intensity: Brightness/contrast (simulates different imaging conditions)
        - Noise: Gaussian blur, dropout (simulates real-world imaging artifacts)
        
        Returns:
            A.Compose: Albumentations composition for fundus images
        """
        if self.mode == 'train':
            return A.Compose([
                # Geometric augmentations (preserve anatomical structure)
                A.RandomRotate90(p=self.augmentation_prob),  # 90-degree rotations are safe
                A.HorizontalFlip(p=self.augmentation_prob),  # Horizontal flip is anatomically valid
                A.VerticalFlip(p=self.augmentation_prob/2),  # Vertical flip less common, lower probability
                
                # Intensity augmentations (simulate different imaging conditions)
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,  # Conservative limits to avoid unrealistic images
                    contrast_limit=0.1, 
                    p=self.augmentation_prob
                ),
                
                # Noise and blur (simulate real-world imaging artifacts)
                A.GaussianBlur(blur_limit=3, p=self.augmentation_prob/2),  # Mild blur
               A.CoarseDropout(
    max_holes=8, 
    max_height=16, 
    max_width=16, 
    p=self.augmentation_prob/2,
    fill_value=0
),
                
                # Convert to PyTorch tensor (required for model input)
                ToTensorV2()
            ])
        else:
            # Validation/test mode: only essential tensor conversion
            return A.Compose([ToTensorV2()])
    
    def _get_oct_transforms(self):
        """
        Create augmentation pipeline for OCT (Optical Coherence Tomography) volumes.
        
        OCT data is 3D volumetric data showing retinal layer structure. 
        Augmentations must be very conservative to preserve medical accuracy.
        
        Note: Geometric transforms are avoided for OCT because:
        - Layer structure is orientation-dependent
        - Flipping could change anatomical interpretation
        - Rotation might introduce interpolation artifacts
        
        Returns:
            A.Compose: Conservative augmentation pipeline for OCT
        """
        if self.mode == 'train':
            return A.Compose([
                # Commented out: Gaussian noise can interfere with layer detection
                # A.GaussianNoise(var_limit=(10, 50), p=self.augmentation_prob/2),
                
                # Very conservative intensity changes only
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,  # Much smaller limits than fundus
                    contrast_limit=0.05,    # Preserves layer contrast relationships
                    p=self.augmentation_prob
                )
            ])
        else:
            # No transforms for validation/test
            return A.Compose([])
    
    def _get_flio_transforms(self):
        """
        Create augmentation pipeline for FLIO (Fluorescence Lifetime Imaging) data.
        
        FLIO data contains 4 channels of lifetime and intensity information.
        Augmentations must preserve:
        - Channel relationships (lifetime and intensity are paired)
        - Quantitative values (lifetime measurements have physical meaning)
        
        Only geometric transforms are used since intensity changes could
        affect the physical interpretation of fluorescence lifetime values.
        
        Returns:
            A.Compose: Geometric-only augmentation pipeline for FLIO
        """
        if self.mode == 'train':
            return A.Compose([
                # Use custom multi-channel aware transforms
                MultiChannelHorizontalFlip(p=self.augmentation_prob),
                MultiChannelVerticalFlip(p=self.augmentation_prob),
                A.RandomRotate90(p=self.augmentation_prob),
                
                # Commented out: Noise could interfere with lifetime measurements
                # A.GaussianNoise(var_limit=(0.001, 0.01), p=self.augmentation_prob/2)
            ])
        else:
            # No transforms for validation/test
            return A.Compose([])
    
    def apply_fundus_transforms(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply fundus-specific transformations to a retinal image.
        
        Args:
            image: Input fundus image as numpy array (H, W, 3)
            
        Returns:
            torch.Tensor: Transformed image ready for model input
        """
        transformed = self.fundus_transforms(image=image)
        return transformed['image']
    
    def apply_oct_transforms(self, volume: np.ndarray) -> torch.Tensor:
        """
        Apply OCT-specific transformations to a 3D volume.
        
        OCT volumes are processed slice by slice to maintain 3D structure
        while applying 2D transforms to each cross-sectional slice.
        
        Args:
            volume: Input OCT volume as numpy array (H, W, D)
            
        Returns:
            torch.Tensor: Transformed volume with shape (1, H, W, D)
        """
        if self.mode == 'train':
            # Apply 2D transforms to each slice independently
            transformed_volume = np.zeros_like(volume)
            for i in range(volume.shape[2]):  # Iterate through depth slices
                slice_2d = volume[:, :, i]
                # Apply 2D transforms to this slice
                transformed = self.oct_transforms(image=slice_2d)
                transformed_volume[:, :, i] = transformed['image']
            volume = transformed_volume
        
        # Convert to tensor and add channel dimension for model compatibility
        return torch.tensor(volume).unsqueeze(0)  # Shape: (1, H, W, D)
    
    def apply_flio_transforms(self, flio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply FLIO-specific transformations to preprocessed 4-channel tensor.
        
        This method handles the tensor format conversion required by Albumentations
        and ensures proper multi-channel processing.
        
        Args:
            flio_tensor: Input FLIO tensor of shape [C, H, W] (4 channels)
            
        Returns:
            torch.Tensor: Transformed FLIO tensor of shape [C, H, W]
        """
        print(f"FLIO tensor shape before transforms: {flio_tensor.shape}")

        # Albumentations expects format [H, W, C], so we need to permute
        flio_np = flio_tensor.permute(1, 2, 0).numpy()

        if self.mode == 'train':
            try:
                # Apply multi-channel aware transforms
                transformed = self.flio_transforms(image=flio_np)
                flio_np = transformed['image']
            except Exception as e:
                print(f"FLIO transform failed, skipping: {e}")
                # Continue with original data if transforms fail

        # Convert back to PyTorch tensor format [C, H, W]
        flio_tensor = torch.tensor(flio_np).permute(2, 0, 1).float()
        print(f"FLIO tensor shape after transforms: {flio_tensor.shape}")
        
        return flio_tensor
    
    def process_flio_data(self, flio_data: Dict) -> torch.Tensor:
        """
        Process raw FLIO data dictionary into standardized tensor format.
        
        This method handles the complex task of converting raw FLIO data
        (which may come in various formats) into a consistent 4-channel tensor.
        
        FLIO data structure:
        - Channel 0: Short wavelength lifetime
        - Channel 1: Long wavelength lifetime  
        - Channel 2: Short wavelength intensity
        - Channel 3: Long wavelength intensity
        
        Args:
            flio_data: Dictionary containing raw FLIO channel data
            
        Returns:
            torch.Tensor: Processed FLIO tensor of shape [C, H, W]
        """
        # Expected channel names in the input dictionary
        channels = []
        for channel in ['lifetime_ch1', 'lifetime_ch2', 'intensity_ch1', 'intensity_ch2']:
            if channel in flio_data:
                channel_data = flio_data[channel]
                
                # Handle different input dimensionalities
                if len(channel_data.shape) > 2:
                    # Reduce higher-dimensional data to 2D
                    if len(channel_data.shape) == 3:
                        # (H, W, T) -> (H, W): Average over time dimension
                        channel_data = np.mean(channel_data, axis=-1)
                    elif len(channel_data.shape) == 4:
                        # (H, W, T, X) -> (H, W): Average over extra dimensions
                        channel_data = np.mean(channel_data, axis=(-1, -2))
                        
                channels.append(channel_data)
        
        # Handle missing channel data
        if not channels:
            # Create dummy data if no channels are found
            print("Warning: No FLIO channels found, creating dummy data")
            dummy_shape = (256, 256)  # Default size - should match config
            flio_stack = np.zeros((*dummy_shape, 4))
        else:
            # Ensure all channels have consistent dimensions
            target_shape = channels[0].shape
            normalized_channels = []
            
            for channel in channels:
                if channel.shape != target_shape:
                    print(f"Warning: Channel shape {channel.shape} != target {target_shape}")
                    # Resize inconsistent channels (this is a fallback)
                    channel = np.resize(channel, target_shape)
                normalized_channels.append(channel)
            
            # Stack channels into multi-channel image: (H, W, C)
            flio_stack = np.stack(normalized_channels, axis=-1)
        
        print(f"Final flio_stack shape before transforms: {flio_stack.shape}")
        
        # Apply augmentations if in training mode
        if self.mode == 'train':
            try:
                # Use custom multi-channel transforms
                transformed = self.flio_transforms(image=flio_stack)
                flio_stack = transformed['image']
            except Exception as e:
                print(f"Warning: FLIO transform failed: {e}")
                
                # Fallback: Apply transforms channel by channel
                # This is less efficient but more robust
                transformed_channels = []
                for i in range(flio_stack.shape[-1]):
                    channel = flio_stack[:, :, i]
                    
                    # Create simple single-channel transforms
                    simple_transforms = A.Compose([
                        A.HorizontalFlip(p=self.augmentation_prob),
                        A.VerticalFlip(p=self.augmentation_prob),
                        A.RandomRotate90(p=self.augmentation_prob),
                    ])
                    
                    transformed_channel = simple_transforms(image=channel)['image']
                    transformed_channels.append(transformed_channel)
                
                flio_stack = np.stack(transformed_channels, axis=-1)
        
        print(f"Final flio_stack shape after transforms: {flio_stack.shape}")
        
        # Convert to PyTorch tensor
        flio_tensor = torch.tensor(flio_stack, dtype=torch.float32)
        
        # Handle various input tensor shapes and convert to standard [C, H, W] format
        if len(flio_tensor.shape) == 4:
            # Handle 4D tensors - need to reduce to 3D
            if flio_tensor.shape[-1] <= 4:  
                # Format: (H, W, C, T) where C <= 4
                flio_tensor = torch.mean(flio_tensor, dim=-1)  # Average over time
                flio_tensor = flio_tensor.permute(2, 0, 1)     # (H, W, C) -> (C, H, W)
            elif flio_tensor.shape[0] <= 4:  
                # Format: (C, H, W, T) where C <= 4
                flio_tensor = torch.mean(flio_tensor, dim=-1)  # Average over time -> (C, H, W)
            else:
                # Fallback: reshape to 3D
                flio_tensor = flio_tensor.view(flio_tensor.shape[0], flio_tensor.shape[1], -1)
                flio_tensor = flio_tensor.permute(2, 0, 1)
                
        elif len(flio_tensor.shape) == 3:
            # Standard case: (H, W, C) -> (C, H, W)
            flio_tensor = flio_tensor.permute(2, 0, 1)
            
        elif len(flio_tensor.shape) == 2:
            # 2D case: (H, W) -> (1, H, W) - treat as single channel
            flio_tensor = flio_tensor.unsqueeze(0)
        
        print(f"Final output tensor shape: {flio_tensor.shape}")
        return flio_tensor