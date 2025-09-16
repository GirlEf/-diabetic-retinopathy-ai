#!/usr/bin/env python3


import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath('.'))

from src.utils.config import Config

def debug_dataset():
    print("=== DATASET DEBUGGING ===")
    
    # 1. Load config
    config = Config()
    print(f"Config loaded successfully")
    
    # 2. Check metadata file
    labels_path = Path(config.get_data_path('labels'))
    print(f"\n1. Metadata file: {labels_path}")
    print(f"   Exists: {labels_path.exists()}")
    
    if labels_path.exists():
        metadata = pd.read_csv(labels_path)
        print(f"   Total rows: {len(metadata)}")
        print(f"   Columns: {metadata.columns.tolist()}")
        print(f"   Split distribution: {metadata['split'].value_counts().to_dict()}")
        
        # Check train split
        train_data = metadata[metadata['split'] == 'train']
        print(f"   Train samples: {len(train_data)}")
        
        if len(train_data) > 0:
            print(f"   Sample train row:")
            print(f"     participant_id: {train_data.iloc[0]['participant_id']}")
            print(f"     fundus_path: {train_data.iloc[0]['fundus_path']}")
            print(f"     oct_path: {train_data.iloc[0]['oct_path']}")
            print(f"     flio_path: {train_data.iloc[0]['flio_path']}")
    
    # 3. Check data directories
    print(f"\n2. Data directories:")
    for modality in ['fundus', 'oct', 'flio']:
        mod_path = Path(config.get_data_path(modality))
        print(f"   {modality}: {mod_path}")
        print(f"     Exists: {mod_path.exists()}")
        if mod_path.exists():
            # Count files
            if modality == 'flio':
                files = list(mod_path.rglob('*'))
            else:
                files = list(mod_path.rglob('*.dcm'))
            print(f"     Files found: {len(files)}")
            if len(files) > 0:
                print(f"     Sample file: {files[0]}")
    
    # 4. Test path construction
    print(f"\n3. Path construction test:")
    if labels_path.exists() and len(train_data) > 0:
        sample_row = train_data.iloc[0]
        
        fundus_root = Path(config.get_data_path('fundus'))
        oct_root = Path(config.get_data_path('oct'))
        flio_root = Path(config.get_data_path('flio'))
        
        def test_path(root, rel_path):
            rel_path = rel_path.lstrip('/')
            full_path = root / rel_path
            exists = full_path.exists()
            return str(full_path), exists
        
        fundus_full, fundus_exists = test_path(fundus_root, sample_row['fundus_path'])
        oct_full, oct_exists = test_path(oct_root, sample_row['oct_path'])
        flio_full, flio_exists = test_path(flio_root, sample_row['flio_path'])
        
        print(f"   Fundus: {fundus_full}")
        print(f"     Exists: {fundus_exists}")
        print(f"   OCT: {oct_full}")
        print(f"     Exists: {oct_exists}")
        print(f"   FLIO: {flio_full}")
        print(f"     Exists: {flio_exists}")
    
    # 5. Check if any files exist with similar patterns
    print(f"\n4. File pattern matching:")
    if labels_path.exists() and len(train_data) > 0:
        sample_id = str(train_data.iloc[0]['participant_id'])
        print(f"   Looking for files with participant_id: {sample_id}")
        
        for modality in ['fundus', 'oct', 'flio']:
            mod_path = Path(config.get_data_path(modality))
            if mod_path.exists():
                if modality == 'flio':
                    matching_files = list(mod_path.rglob(f'*{sample_id}*'))
                else:
                    matching_files = list(mod_path.rglob(f'*{sample_id}*.dcm'))
                print(f"   {modality} files with {sample_id}: {len(matching_files)}")
                if len(matching_files) > 0:
                    print(f"     Sample: {matching_files[0]}")

if __name__ == "__main__":
    debug_dataset() 