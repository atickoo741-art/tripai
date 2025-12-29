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
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = self._generate_run_id()
        

    def _generate_run_id(self) -> str:
        """
        Generate smart run ID: {exp_name}_{config_hash}_{counter}
        Example: baseline_a3f2_001
        """
        import hashlib
        
        # Get experiment name (sanitized)
        exp_name = self.get('experiment.name', 'qat_run')
        exp_name = exp_name.lower().replace(' ', '_').replace('-', '_')[:20]
        
        # Generate short config hash (first 4 chars of SHA256)
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:4]
        
        # Get run counter (auto-increment)
        counter = self._get_next_counter(exp_name, config_hash)
        
        return f"{exp_name}_{config_hash}_{counter:03d}"
    
    def _get_next_counter(self, exp_name: str, config_hash: str) -> int:
        """Auto-increment counter for same experiment + config"""
        output_base = self.get('training.output_dir', './output_qat_lora')
        prefix = f"{exp_name}_{config_hash}_"
        
        # Find existing runs with same prefix
        if os.path.exists(os.path.dirname(output_base)):
            existing = [d for d in os.listdir(os.path.dirname(output_base)) 
                       if d.startswith(prefix)]
            if existing:
                counters = [int(d.split('_')[-1]) for d in existing 
                           if d.split('_')[-1].isdigit()]
                return max(counters) + 1 if counters else 1
        return 1
    

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
            "experiment_name": self.get('experiment.name'),
            "config_hash": self.run_id.split('_')[1],  # Extract hash
            "run_number": int(self.run_id.split('_')[-1]),
            "timestamp": datetime.now().isoformat(),
            "notes": self.get('experiment.notes'),
            "tags": self.get('experiment.tags', []),
            # "git_commit": self._get_git_commit(),  # Optional: track code version
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
