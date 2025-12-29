# config_loader.py
import yaml
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import shutil

class Config:
    """Configuration loader and saver for experiments"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """
        Get nested config value using dot notation
        Example: config.get('training.learning_rate')
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    def save_to_output(self, output_dir: str):
        """
        Save config copy to output directory with timestamp
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save YAML copy
        config_save_path = output_path / f"config_{self.run_id}.yaml"
        shutil.copy(self.config_path, config_save_path)
        
        # Also save as JSON for easy programmatic access
        json_path = output_path / f"config_{self.run_id}.json"
        with open(json_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save run metadata
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config_file": str(config_save_path),
            "experiment_name": self.get('experiment.name'),
            "notes": self.get('experiment.notes'),
            "tags": self.get('experiment.tags', [])
        }
        metadata_path = output_path / f"metadata_{self.run_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved config to {config_save_path}")
        print(f"✓ Saved metadata to {metadata_path}")
        
        return config_save_path
    
    def __getitem__(self, key):
        """Allow dict-like access: config['training']"""
        return self.config[key]
    
    def __repr__(self):
        return f"Config(run_id={self.run_id}, keys={list(self.config.keys())})"
