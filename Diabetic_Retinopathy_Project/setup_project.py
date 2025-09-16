import os
from pathlib import Path

def create_project_structure():
    # Directory structure
    directories = [
        "src/data", "src/models", "src/utils", "configs", "notebooks",
        "data/raw/fundus", "data/raw/oct", "data/raw/flio", 
        "data/processed", "data/interim",
        "models/checkpoints", "models/pretrained", "models/outputs",
        "results/logs", "results/figures", "results/metrics"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Files to create
    files = [
        "src/__init__.py", "src/data/__init__.py", "src/data/dataset.py", 
        "src/data/preprocessing.py", "src/data/transforms.py",
        "src/models/__init__.py", "src/models/ssl.py", "src/models/fusion.py", 
        "src/models/foundation_models.py", "src/utils/__init__.py", 
        "src/utils/config.py", "src/utils/training_utils.py",
        "src/utils/visualize.py", "src/utils/metrics.py",
        "configs/paths.yaml", "configs/hyperparams.yaml", 
        "requirements.txt", "README.md", ".gitignore",
        "notebooks/01_Data_Exploration.ipynb", "notebooks/02_Preprocessing.ipynb", 
        "notebooks/03_SSL_Pretraining.ipynb", "notebooks/04_Supervised_Training.ipynb"
    ]
    
    # Create all files
    for file_path in files:
        Path(file_path).touch()
    
    print("✅ Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()