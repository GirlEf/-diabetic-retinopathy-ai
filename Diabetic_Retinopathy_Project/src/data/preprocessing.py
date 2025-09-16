# ============================================================================
# /src/data/preprocessing.py
# Medical Imaging Preprocessing Pipeline
# ============================================================================
"""
This module provides preprocessing capabilities for multi-modal medical imaging data:
- Fundus images (retinal photography)
- OCT (Optical Coherence Tomography) volumes
- FLIO (Fluorescence Lifetime Imaging Ophthalmoscopy) data

The preprocessing includes resizing, normalization, enhancement, and format standardization
to prepare the data for machine learning models.
"""

import sys
import os
# Add parent directory to path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import numpy as np
import pydicom  # For reading DICOM medical image files
import torch
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.config import Config
from typing import List, Union

class MultiModalPreprocessor:
    """
    A comprehensive preprocessing pipeline for multi-modal medical imaging data.
    
    This class handles three types of medical imaging data:
    1. Fundus images - 2D color photographs of the retina
    2. OCT volumes - 3D cross-sectional images of retinal layers
    3. FLIO data - 4-channel fluorescence lifetime imaging data
    
    Each modality requires specific preprocessing steps to ensure consistency
    and optimal performance in downstream machine learning tasks.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the preprocessor with configuration settings.
        
        Args:
            config (Config): Configuration object containing hyperparameters
                           including target image sizes for each modality
        """
        self.config = config
        
        # Extract target dimensions for each imaging modality from config
        # These dimensions ensure consistent input sizes for ML models
        self.fundus_size = tuple(config.hyperparams['data']['image_size'])  # e.g., (512, 512)
        self.oct_size = tuple(config.hyperparams['data']['oct_size'])        # e.g., (512, 512, 128)
        self.flio_size = tuple(config.hyperparams['data']['flio_size'])      # e.g., (256, 256)
        
        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast in fundus images, improving feature visibility
        # clipLimit=2.0 prevents over-amplification of noise
        # tileGridSize=(8,8) divides image into 8x8 grid for local enhancement
        
    
    def preprocess_fundus(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess a fundus (retinal) image for machine learning.
        """
        # Load image based on file format
        if str(image_path).endswith('.dcm'):
            # Load DICOM file (medical imaging standard format)
            img = self._load_dicom_file(image_path)
            # Convert grayscale to RGB if needed (fundus should be color)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # Load standard image formats (PNG, JPG, etc.)
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # OpenCV loads as BGR, convert to RGB for consistency
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to target dimensions for model input consistency
        img = cv2.resize(img, (224, 224))

        # Normalize pixel values from [0,255] to [0,1] range
        img = img.astype(np.float32) / 255.0

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast, making retinal features more visible
        # Process each color channel independently
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(3):
            channel = (img[:, :, i] * 255).astype(np.uint8)
            img[:, :, i] = clahe.apply(channel) / 255.0

        # Standardize the image (zero mean, unit variance)
        img = (img - img.mean()) / (img.std() + 1e-8)  # 1e-8 prevents division by zero

        return img

    def preprocess_oct(self, oct_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess OCT (Optical Coherence Tomography) volume data.
        
        OCT provides 3D cross-sectional images of retinal layers. This function:
        1. Loads the 3D volume (DICOM or .npy format)
        2. Resizes to target dimensions while preserving depth information
        3. Normalizes the volume to zero mean and unit variance
        
        Args:
            oct_path: Path to OCT volume file (.dcm or .npy)
            
        Returns:
            np.ndarray: Preprocessed OCT volume of shape (height, width, depth)
            
        Raises:
            ValueError: If volume is not 3D or cannot be loaded
        """
        # Load OCT data based on file format
        if str(oct_path).endswith('.dcm'):
            # Load DICOM volume
            volume = self._load_dicom_file(oct_path)
        else:
            # Load NumPy array (.npy format)
            volume = np.load(oct_path)
        
        # Ensure we have a 3D volume and resize appropriately
        if len(volume.shape) == 3:
            # Initialize resized volume with target dimensions
            volume_resized = np.zeros(self.oct_size)
            
            # Resize each slice (2D cross-section) individually
            # This preserves the depth information while standardizing x,y dimensions
            for i in range(min(volume.shape[2], self.oct_size[2])):
                slice_2d = volume[:, :, i]  # Extract 2D slice
                # Resize slice to target x,y dimensions
                slice_resized = cv2.resize(slice_2d, 
                                         (self.oct_size[1], self.oct_size[0]))
                volume_resized[:, :, i] = slice_resized
        else:
            raise ValueError(f"Expected 3D OCT volume, got shape: {volume.shape}")
        
        # Normalize the entire volume (zero mean, unit variance)
        # This is crucial for 3D medical imaging to handle intensity variations
        volume_normalized = (volume_resized - np.mean(volume_resized)) / \
                           (np.std(volume_resized) + 1e-8)
        
        return volume_normalized.astype(np.float32)

    def preprocess_flio(self, flio_files: List[Union[str, Path]]) -> torch.Tensor:
        """
        Preprocess FLIO (Fluorescence Lifetime Imaging Ophthalmoscopy) data.
        Keeps only two available channels: lifetime_ch1 and lifetime_ch2.
        
        Args:
            flio_files: List of paths to FLIO DICOM files

        Returns:
            torch.Tensor: 2-channel FLIO tensor of shape (2, height, width)
            
        Raises:
            ValueError: If input is not a non-empty list
        """
        # Validate input parameters
        if not isinstance(flio_files, list) or len(flio_files) == 0:
            raise ValueError("FLIO input must be a non-empty list of DICOM file paths")

        print("\n📂 Received FLIO files:")
        for path in flio_files:
            print("   -", path)

        # Define mapping from filename keywords to channel names
        # Focus only on lifetime channels since intensity channels are missing
        channel_keywords = {
            'short_wavelength_l': 'lifetime_ch1',    # Short wavelength lifetime
            'long_wavelength_l': 'lifetime_ch2',     # Long wavelength lifetime
        }

        # Define physiologically meaningful value ranges for lifetime channels
        clip_ranges = {
            'lifetime_ch1': (0.5, 3.0),      # Lifetime in nanoseconds
            'lifetime_ch2': (1.0, 5.0),      # Lifetime in nanoseconds
        }

        # Initialize storage for channel data
        channel_data = {}

        # Process each DICOM file and assign to appropriate channel
        for dcm_path in flio_files:
            dcm_path = str(dcm_path)
            # Check filename against each keyword to determine channel type
            for keyword, channel_name in channel_keywords.items():
                if keyword in dcm_path.lower():
                    print(f"✅ Matched {channel_name} using keyword: {keyword}")
                    if Path(dcm_path).exists():
                        # Load DICOM image
                        image = self._load_dicom_file(dcm_path)
                        
                        # Handle different image dimensions
                        if image.ndim == 3:
                            # If 3D, take middle slice (most representative)
                            image = image[:, :, image.shape[-1] // 2]
                        elif image.ndim != 2:
                            raise ValueError(f"Unexpected FLIO shape: {image.shape}")
                        
                        # Resize to target dimensions
                        image = cv2.resize(image, self.flio_size)
                        
                        # Clip values to physiologically meaningful ranges
                        # This removes outliers and ensures data quality
                        image = np.clip(image, *clip_ranges[channel_name]).astype(np.float32)
                        
                        # Store processed channel data
                        channel_data[channel_name] = image
                    else:
                        print(f"❌ File does not exist: {dcm_path}")
                    break  # Found matching keyword, move to next file

        # Only keep the two lifetime channels that are available
        channels = []
        for ch in ['lifetime_ch1', 'lifetime_ch2']:
            if ch in channel_data:
                channels.append(channel_data[ch])
            else:
                # Missing channel - fill with zeros and warn user
                print(f"⚠️ Missing FLIO channel: {ch} – filling with zeros.")
                channels.append(np.zeros(self.flio_size, dtype=np.float32))

        # Stack channels into a single tensor with shape [C, H, W]
        # This creates a 2-channel image for the available FLIO data
        flio_tensor = np.stack(channels, axis=0)

        # Min-max normalize each channel independently
        # This ensures consistent value ranges across channels
        for i in range(flio_tensor.shape[0]):
            ch_min, ch_max = flio_tensor[i].min(), flio_tensor[i].max()
            if ch_max - ch_min > 0:
                # Normalize to [0, 1] range
                flio_tensor[i] = (flio_tensor[i] - ch_min) / (ch_max - ch_min)

        print(f"✅ Final FLIO tensor shape (2 channels): {flio_tensor.shape}\n")

        # Convert to PyTorch tensor and return
        return torch.tensor(flio_tensor, dtype=torch.float32)

    def _load_dicom_file(self, dicom_path: Union[str, Path]) -> np.ndarray:
        """
        Load a single DICOM medical imaging file.
        
        DICOM (Digital Imaging and Communications in Medicine) is the standard
        format for medical images. This function handles the complexities of
        DICOM loading and extracts the pixel data.
        
        Args:
            dicom_path: Path to the DICOM file
            
        Returns:
            np.ndarray: Image data as numpy array (2D or 3D depending on modality)
            
        Raises:
            ValueError: If DICOM file cannot be loaded or has no pixel data
        """
        try:
            # Load DICOM dataset using pydicom
            ds = pydicom.dcmread(dicom_path)
            
            # Extract pixel data from DICOM dataset
            if hasattr(ds, 'pixel_array'):
                # Convert to float32 for consistent processing
                image = ds.pixel_array.astype(np.float32)
            else:
                raise ValueError(f"No pixel data in DICOM: {dicom_path}")
            
            # Handle different image dimensions based on modality
            if len(image.shape) == 2:
                # 2D image (fundus, FLIO channels)
                return image
            elif len(image.shape) == 3:
                # 3D volume (OCT)
                return image
            else:
                raise ValueError(f"Unexpected image dimensions: {image.shape}")
                
        except Exception as e:
            raise ValueError(f"Error loading DICOM {dicom_path}: {e}")

# Test the preprocessing module when run directly
if __name__ == "__main__":
    """
    Simple test to verify the preprocessing module loads correctly.
    This is useful for debugging and development.
    """
    print("Testing preprocessing module...")
    try:
        # Attempt to create a preprocessor instance
        config = Config()
        preprocessor = MultiModalPreprocessor(config)
        print("✅ Preprocessing module loaded successfully!")
        
        # Could add more comprehensive tests here:
        # - Test with sample images
        # - Verify output dimensions
        # - Check processing time
        
    except Exception as e:
        print(f"❌ Error loading preprocessing module: {e}")
        # In a production environment, you might want to log this error
        # or provide more detailed debugging information