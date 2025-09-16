#!/usr/bin/env python3
"""
Fix script for the preprocessing notebook path issues
"""

import os
import sys
from pathlib import Path

def fix_notebook_paths():
    """Fix the path setup for running notebooks from the notebooks directory"""
    
    # Get the project root (parent of notebooks directory)
    project_root = Path.cwd().parent
    print(f"Project root: {project_root}")
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Change working directory to project root
    os.chdir(project_root)
    print(f"Changed working directory to: {os.getcwd()}")
    
    # Test imports
    try:
        from src.utils.config import Config
        from src.data.dataset import MultiModalRetinalDataset
        from src.data.preprocessing import MultiModalPreprocessor
        from src.data.transforms import MultiModalTransforms
        
        print("✅ All imports successful!")
        
        # Test config loading
        config = Config()
        print(f"✅ Config loaded successfully!")
        print(f"   Data paths: {config.paths}")
        print(f"   Hyperparams: {config.hyperparams}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=== FIXING PREPROCESSING NOTEBOOK PATHS ===\n")
    
    if fix_notebook_paths():
        print("\n✅ Path setup successful!")
        print("\nTo run the preprocessing notebook:")
        print("1. Open the notebook in Jupyter")
        print("2. Add this cell at the beginning:")
        print("   ```python")
        print("   import sys")
        print("   import os")
        print("   from pathlib import Path")
        print("   ")
        print("   # Fix paths for notebook")
        print("   project_root = Path.cwd().parent")
        print("   sys.path.insert(0, str(project_root))")
        print("   os.chdir(project_root)")
        print("   ```")
        print("3. Run all cells")
    else:
        print("\n❌ Path setup failed. Please check your project structure.") 