# ============================================================================
# /src/data/dataset.py
# Multi-Modal Medical Imaging Dataset Loader
# ============================================================================
"""
This module provides a PyTorch Dataset class for loading multi-modal medical imaging data.
It handles three types of medical imaging modalities simultaneously:

1. Fundus images - 2D color photographs of the retina
2. OCT volumes - 3D cross-sectional images of retinal layers  
3. FLIO data - 4-channel fluorescence lifetime imaging

The dataset supports:
- Flexible modality selection (can use any combination of the three)
- Train/validation/test splits
- Missing modality handling (patients don't need all modalities)
- Automated data manifest creation from directory structures
- Class balancing for imbalanced medical datasets

This is designed for diabetic retinopathy (DR) classification tasks where
patients are graded on a severity scale (typically 0-4).
"""

import sys
import os
# Add parent directory to path for importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import re
from typing import Dict, Tuple, Optional, List
import logging
from src.utils.config import Config
from src.data.transforms import MultiModalTransforms
from src.data.preprocessing import MultiModalPreprocessor
import wandb
# Disable wandb logging to avoid prompts during dataset loading
wandb.init(mode="disabled")


class MultiModalRetinalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal retinal imaging data.
    
    This dataset class handles the complex task of loading and organizing
    multi-modal medical imaging data where patients may have different
    combinations of available imaging modalities.
    
    Key features:
    - Supports missing modalities (patient doesn't need all three types)
    - Handles different file formats (DICOM, standard images, numpy arrays)
    - Applies modality-specific preprocessing and augmentation
    - Manages train/validation/test splits
    - Provides class balancing for imbalanced medical datasets
    
    The dataset assumes a directory structure where patient data is organized
    by participant ID with separate folders/files for each modality.
    """
    
    def __init__(
        self,
        config: Config,
        split: str = 'train',
        modalities: List[str] = ['fundus', 'oct', 'flio'],
        transforms: Optional[MultiModalTransforms] = None
    ):
        """
        Initialize the multi-modal retinal dataset.
        
        Args:
            config: Configuration object containing paths and hyperparameters
            split: Data split to load ('train', 'val', 'test')
            modalities: List of modalities to include in the dataset
            transforms: Optional custom transforms; if None, creates default transforms
        """
        self.config = config
        self.split = split
        self.modalities = modalities
        
        # Initialize transforms for data augmentation
        # If no transforms provided, create default ones based on the split
        self.transforms = transforms or MultiModalTransforms(mode=split)
        
        # Initialize preprocessor for handling different file formats and normalization
        self.preprocessor = MultiModalPreprocessor(config)
        
        # Load the data manifest (mapping of participant IDs to file paths)
        self.data_df = self._load_data_manifest()
        
        # Load labels (diabetic retinopathy grades)
        self.labels = self._load_labels()
        
        # Filter dataset to only include patients with at least one available modality
        self.data_df = self._filter_complete_cases()
        
        logging.info(f"Loaded {len(self.data_df)} samples for {split} split")
    
    def _load_data_manifest(self) -> pd.DataFrame:
        """
        Load or create a data manifest mapping participant IDs to file paths.
        
        The manifest is a CSV file that contains:
        - participant_id: Unique identifier for each patient
        - fundus_path: Path to fundus image file
        - oct_path: Path to OCT volume file
        - flio_path: Path to FLIO data files
        - split: Train/validation/test assignment
        
        Returns:
            pd.DataFrame: Data manifest with file paths for each participant
        """
        # Look for pre-existing manifest file
        manifest_path = Path(self.config.paths['data']['raw']) / f"{self.split}_manifest.csv"
        
        if manifest_path.exists():
            # Load existing manifest
            return pd.read_csv(manifest_path)
        else:
            # Create manifest from directory structure if none exists
            return self._create_manifest_from_dirs()
    
    def _create_manifest_from_dirs(self) -> pd.DataFrame:
        """
        Create a data manifest by scanning the directory structure.
        
        This method is called when no pre-existing manifest file is found.
        It reads a labels file and constructs full file paths based on the
        configured directory structure.
        
        The method handles different directory structures for each modality:
        - Fundus: adds 'retinal_photography' subdirectory
        - OCT: handles 'retinal_oct' to 'retina_oct' naming conversion
        - FLIO: adds 'retinal_flio' subdirectory
        
        Returns:
            pd.DataFrame: Newly created manifest with full file paths
        """
        # Load the labels file which contains relative paths
        labels_path = Path(self.config.get_data_path('labels'))
        labels = pd.read_csv(labels_path)
        
        # Filter by the current split (train/val/test)
        labels = labels[labels['split'] == self.split]
        
        # Get root directories for each modality
        fundus_root = Path(self.config.get_data_path('fundus'))
        oct_root = Path(self.config.get_data_path('oct'))
        flio_root = Path(self.config.get_data_path('flio'))
        
        def join_path(root, rel):
            """
            Join root path with relative path, handling modality-specific subdirectories.
            
            Args:
                root: Root directory path for the modality
                rel: Relative path from the labels file
                
            Returns:
                str: Full absolute path to the file
            """
            # Remove leading slash from relative path
            rel = rel.lstrip('/')
            
            # Handle modality-specific directory structures
            if 'fundus' in str(root):
                # Fundus images are in a 'retinal_photography' subdirectory
                full_path = root / 'retinal_photography' / rel
            elif 'oct' in str(root):
                # OCT data has a naming convention difference
                rel = rel.replace('retinal_oct/', 'retina_oct/')
                full_path = root / rel
            elif 'flio' in str(root):
                # FLIO data is in a 'retinal_flio' subdirectory
                full_path = root / 'retinal_flio' / rel
            else:
                # Default case
                full_path = root / rel
            
            return str(full_path)
        
        # Apply path joining to create full paths for each modality
        labels['fundus_path'] = labels['fundus_path'].apply(lambda x: join_path(fundus_root, x))
        labels['oct_path'] = labels['oct_path'].apply(lambda x: join_path(oct_root, x))
        labels['flio_path'] = labels['flio_path'].apply(lambda x: join_path(flio_root, x))
        
        # Rename label column to dr_grade for consistency
        # DR grade typically ranges from 0 (no DR) to 4 (proliferative DR)
        labels = labels.rename(columns={'label': 'dr_grade'})
        
        # Print diagnostic information
        print(f"Created manifest with {len(labels)} samples")
        print(f"Sample fundus path: {labels['fundus_path'].iloc[0] if len(labels) > 0 else 'No samples'}")
        print(f"Sample oct path: {labels['oct_path'].iloc[0] if len(labels) > 0 else 'No samples'}")
        print(f"Sample flio path: {labels['flio_path'].iloc[0] if len(labels) > 0 else 'No samples'}")
        
        return labels
    
    def _load_labels(self) -> pd.DataFrame:
        """
        Load the labels CSV file containing diabetic retinopathy grades.
        
        The labels file contains:
        - participant_id: Patient identifier
        - dr_grade: Diabetic retinopathy severity grade (0-4)
        - split: Train/validation/test assignment
        
        Returns:
            pd.DataFrame: Labels dataframe filtered for current split
        """
        labels_path = Path(self.config.paths['dataset']['labels'])
        labels = pd.read_csv(labels_path)
        
        # Filter for the current split only
        labels = labels[labels['split'] == self.split]
        
        # Rename label column to dr_grade for consistency
        labels = labels.rename(columns={'label': 'dr_grade'})
        
        return labels
    
    def _filter_complete_cases(self) -> pd.DataFrame:
        """
        Filter the dataset to include only patients with at least one available modality.
        
        In medical imaging datasets, not all patients have all modalities available.
        This method ensures we only include patients who have at least one of the
        requested modalities, preventing empty samples during training.
        
        Returns:
            pd.DataFrame: Filtered dataframe containing only complete cases
        """
        available_cases = self.data_df.copy()
        
        print(f"Filtering {len(available_cases)} samples for available modalities...")
        
        # Check each patient to see if they have at least one modality available
        valid_patients = []
        for _, row in available_cases.iterrows():
            has_any_modality = False
            
            # Check each requested modality
            for modality in self.modalities:
                path_col = f"{modality}_path"
                path = Path(row[path_col])
                
                # If file exists, this patient has at least one modality
                if path.exists():
                    has_any_modality = True
                    break
            
            # Add to valid patients if they have at least one modality
            if has_any_modality:
                valid_patients.append(row['participant_id'])
        
        # Filter to only include patients with at least one modality
        available_cases = available_cases[
            available_cases['participant_id'].isin(valid_patients)
        ]
        
        print(f"After filtering: {len(available_cases)} samples")
        return available_cases.reset_index(drop=True)
    
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Required method for PyTorch Dataset interface.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single multi-modal sample from the dataset.
        Handles missing modalities gracefully.
        """
        row = self.data_df.iloc[idx]
        participant_id = row['participant_id']
        
        label_row = self.labels[self.labels['participant_id'] == participant_id]
        if len(label_row) == 0:
            raise ValueError(f"No label found for patient {participant_id}")
        
        if 'dr_grade' not in label_row.columns:
            print(f"Available columns in label_row: {label_row.columns.tolist()}")
            raise ValueError(f"Column 'dr_grade' not found in label_row")
        
        dr_grade = label_row['dr_grade'].iloc[0]
        
        sample = {
            'participant_id': participant_id,
            'label': torch.tensor(dr_grade, dtype=torch.long)
        }

        # Fundus
        if 'fundus' in self.modalities and Path(row['fundus_path']).exists():
            fundus_img = self.preprocessor.preprocess_fundus(row['fundus_path'])
            sample['fundus'] = self.transforms.apply_fundus_transforms(fundus_img)

        # OCT
        if 'oct' in self.modalities and Path(row['oct_path']).exists():
            oct_volume = self.preprocessor.preprocess_oct(row['oct_path'])
            sample['oct'] = self.transforms.apply_oct_transforms(oct_volume)

        # FLIO
        if 'flio' in self.modalities and Path(row['flio_path']).exists():
            flio_path = Path(row['flio_path'])
            flio_dir = flio_path.parent
            pid = flio_dir.name

            # 🔍 Grab all FLIO DICOMs for the patient
            flio_files = list(flio_dir.glob(f"{pid}_flio_*.dcm"))

            if flio_files:
                flio_tensor = self.preprocessor.preprocess_flio(flio_files)
                sample['flio'] = self.transforms.apply_flio_transforms(flio_tensor)

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Medical datasets are often imbalanced - there are typically many more
        patients with mild or no diabetic retinopathy than severe cases.
        Class weights help the model pay more attention to rare classes.
        
        The weight for each class is calculated as:
        weight = total_samples / (num_classes * samples_in_class)
        
        Returns:
            torch.Tensor: Weight for each class, ordered by class index
        """
        # Count samples in each DR grade class
        labels_counts = self.labels['dr_grade'].value_counts().sort_index()
        total_samples = len(self.labels)
        
        # Calculate inverse frequency weights
        # More rare classes get higher weights
        weights = total_samples / (len(labels_counts) * labels_counts.values)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _map_metadata_path_to_local(self, metadata_path: str, modality: str) -> str:
        """
        Map a metadata path to the corresponding local file path.
        
        This utility method converts relative paths from metadata files
        to absolute local paths based on the configuration.
        
        Args:
            metadata_path: Path from metadata file
            modality: Type of modality ('fundus', 'oct', 'flio')
            
        Returns:
            str: Local file path
        """
        # Extract just the filename from the metadata path
        filename = Path(metadata_path).name
        
        # Map to local directory based on modality
        local_path = Path(self.config.get_data_path(modality)) / filename
        
        return str(local_path)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the MultiModalRetinalDataset.
    This section demonstrates how to instantiate and use the dataset.
    """
    from src.utils.config import Config
    
    # Initialize configuration
    config = Config()
    
    # Create dataset for training
    train_dataset = MultiModalRetinalDataset(
        config=config,
        split='train',
        modalities=['fundus', 'oct', 'flio']
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    # Check class distribution
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Example of creating a DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    print("Dataset loaded successfully!")