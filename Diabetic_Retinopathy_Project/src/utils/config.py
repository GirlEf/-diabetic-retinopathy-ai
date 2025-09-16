# ============================================================================
# /src/utils/config.py
# ============================================================================

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Configuration class for the DR project"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Automatically find the absolute path to 'configs' folder
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # /src/utils/ → project root
            self.config_dir = project_root / "configs"
        else:
            self.config_dir = Path(config_dir).resolve()

        self.paths = self._load_yaml("paths.yaml")
        self.hyperparams = self._load_yaml("hyperparams.yaml")

        # Create necessary directories
        self._create_directories()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(self.config_dir / filename, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def _create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.paths['data']['processed'],
            self.paths['data']['interim'],
            self.paths['models']['checkpoints'],
            self.paths['models']['outputs'],
            self.paths['results']['logs'],
            self.paths['results']['figures'],
            self.paths['results']['metrics']
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_data_path(self, modality: str) -> str:
        """Get data path for a specific modality"""
        return self.paths['dataset'][modality]

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.hyperparams['model']

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.hyperparams['training']
    
    def get_manifest_path(self) -> Path:
        """Return the path to the manifest.csv file"""
        return Path(self.paths['data']['manifest'])
