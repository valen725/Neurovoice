
import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "project": {
        "name": "NeuroVoice",
        "version": "1.0.0",
        "description": "Parkinson detection through voice analysis using CNN",
        "author": "NeuroVoice Team"
    },
    "data": {
        "raw_path": "data/raw",
        "processed_path": "data/processed", 
        "features_path": "data/features_final",
        "final_path": "data/final_for_training",
        "metadata_path": "data/metadata",
        "target_sample_rate": 16000,
        "target_duration": 3.0,
        "feature_types": ["mfcc", "mel"]
    },
    "model": {
        "architecture": "hybrid",
        "num_classes": 2,
        "dropout_rate": 0.3,
        "checkpoints_path": "model/checkpoints",
        "results_path": "model/results",
        "logs_path": "model/logs"
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 15,
        "validation_split": 0.2,
        "test_split": 0.2,
        "use_class_weights": True,
        "augment_data": True
    },
    "features": {
        "mfcc": {
            "n_mfcc": 20,
            "n_fft": 1024,
            "hop_length": 512
        },
        "mel": {
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512
        }
    },
    "evaluation": {
        "metrics": ["accuracy", "auc", "precision", "recall", "f1"],
        "generate_plots": True,
        "save_predictions": True
    },
    "system": {
        "verbose": False,
        "log_level": "WARNING"
    }
}


class NeuroVoiceConfig:
    """Configuration manager for NeuroVoice project."""
    
    def __init__(self, config_file: Optional[str] = None, verbose: bool = False):
        self.config = DEFAULT_CONFIG.copy()
        self.config_file = config_file
        self.verbose = verbose
        self._setup_logging()
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config["system"]["log_level"], logging.WARNING)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str) -> bool:
        """Load configuration from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Convert nested config to flat structure
            flat_config = self._flatten_config(file_config)
            
            # Update with loaded config
            self.config.update(flat_config)
            
            if self.verbose:
                print(f"Configuration loaded from: {config_file}")
            
            return True
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load config from {config_file}: {e}")
            return False
    
    def _flatten_config(self, nested_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested configuration to flat structure expected by training code."""
        flat_config = {}
        
        # Handle nested structure - extract values to flat keys
        if "training" in nested_config:
            training = nested_config["training"]
            flat_config.update({
                "batch_size": training.get("batch_size", 32),
                "num_epochs": training.get("num_epochs", 100),
                "learning_rate": training.get("learning_rate", 0.001),
                "optimizer": training.get("optimizer", "adamw"),
                "scheduler": training.get("scheduler", "reduce_on_plateau"),
                "early_stopping_patience": training.get("early_stopping_patience", 15),
                "test_size": training.get("test_split", 0.2),
                "validation_size": training.get("validation_split", 0.2),
                "use_class_weights": training.get("use_class_weights", True),
                "augment_train": training.get("augment_data", True),
                "weight_decay": training.get("weight_decay", 1e-4),
                "gradient_clipping": training.get("gradient_clipping", 1.0)
            })
        
        if "model" in nested_config:
            model = nested_config["model"]
            flat_config.update({
                "model_type": model.get("architecture", "hybrid"),
                "num_classes": model.get("num_classes", 2),
                "dropout_rate": model.get("dropout_rate", 0.3)
            })
        
        if "data" in nested_config:
            data = nested_config["data"]
            # Try to find the metadata file
            metadata_candidates = [
                "data/metadata/metadata_final_for_training.csv",
                "data/metadata/metadata_features_final.csv",
                "data/metadata/metadata_features.csv"
            ]
            
            metadata_file = None
            for candidate in metadata_candidates:
                if Path(candidate).exists():
                    metadata_file = candidate
                    break
            
            flat_config.update({
                "metadata_file": metadata_file or metadata_candidates[0],
                "feature_type": "both"  # Based on feature_types in data config
            })
        
        # Add any missing defaults
        defaults = {
            "random_state": 42,
            "num_workers": 4,
            "early_stopping_min_delta": 0.001
        }
        
        for key, value in defaults.items():
            if key not in flat_config:
                flat_config[key] = value
        
        # If it's already a flat config, just use it
        if "batch_size" in nested_config:
            flat_config.update(nested_config)
        
        return flat_config
    
    def save_config(self, config_file: str) -> bool:
        """Save current configuration to JSON file."""
        try:
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            if self.verbose:
                print(f"Configuration saved to {config_file}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error saving configuration: {e}")
            self.logger.error(f"Failed to save config to {config_file}: {e}")
            return False
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get value using dot notation.
        Example: config.get('model.num_classes')
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set value using dot notation.
        Example: config.set('training.batch_size', 64)
        """
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def create_directories(self) -> bool:
        """Create necessary project directories."""
        directories = [
            self.get('data.raw_path'),
            self.get('data.processed_path'), 
            self.get('data.features_path'),
            self.get('data.final_path'),
            self.get('data.metadata_path'),
            self.get('model.checkpoints_path'),
            self.get('model.results_path'),
            self.get('model.logs_path'),
            'model/evaluation_results',
            'model/plots'
        ]
        
        try:
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                print(f"Project directories created: {len(directories)} folders")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error creating directories: {e}")
            self.logger.error(f"Failed to create directories: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        errors = []
        
        # Validate critical paths
        if not self.get('data.metadata_path'):
            errors.append("Metadata path not specified")
        
        # Validate model parameters
        if self.get('model.num_classes') < 2:
            errors.append("num_classes must be >= 2")
        
        # Validate training parameters
        if self.get('training.batch_size') < 1:
            errors.append("batch_size must be >= 1")
        
        if self.get('training.learning_rate') <= 0:
            errors.append("learning_rate must be > 0")
        
        # Validate splits
        val_split = self.get('training.validation_split')
        test_split = self.get('training.test_split')
        if val_split + test_split >= 1.0:
            errors.append("validation_split + test_split must be < 1.0")
        
        if errors:
            if self.verbose:
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  - {error}")
            
            for error in errors:
                self.logger.error(f"Config validation: {error}")
            
            return False
        
        if self.verbose:
            print("Configuration is valid")
        
        return True
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            'metadata_file': f"{self.get('data.metadata_path')}/metadata_features_final.csv",
            'batch_size': self.get('training.batch_size'),
            'test_size': self.get('training.test_split'),
            'validation_size': self.get('training.validation_split'),
            'feature_type': 'both',
            'model_type': self.get('model.architecture'),
            'num_classes': self.get('model.num_classes'),
            'dropout_rate': self.get('model.dropout_rate'),
            'num_epochs': self.get('training.num_epochs'),
            'learning_rate': self.get('training.learning_rate'),
            'optimizer': self.get('training.optimizer'),
            'scheduler': self.get('training.scheduler'),
            'weight_decay': 0.01,
            'gradient_clipping': 1.0,
            'use_class_weights': self.get('training.use_class_weights'),
            'early_stopping_patience': self.get('training.early_stopping_patience'),
            'early_stopping_min_delta': 0.001,
            'augment_train': self.get('training.augment_data'),
            'num_workers': 4,
            'random_state': 42
        }
    
    def print_config(self):
        """Print current configuration in readable format."""
        print("\nNeuroVoice Configuration")
        print("-" * 40)
        
        def print_section(section_data, indent=0):
            for key, value in section_data.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    print(f"{prefix}{key}:")
                    print_section(value, indent + 1)
                else:
                    print(f"{prefix}{key}: {value}")
        
        print_section(self.config)
        print("-" * 40)


def setup_environment(verbose: bool = False) -> bool:
    """Setup NeuroVoice environment."""
    if verbose:
        print("Setting up NeuroVoice environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        if verbose:
            print("Error: Python 3.8+ required")
        return False
    
    # Create configuration
    config = NeuroVoiceConfig(verbose=verbose)
    
    # Create directories
    if not config.create_directories():
        return False
    
    # Validate configuration
    if not config.validate_config():
        return False
    
    # Save default configuration
    config.save_config('model/config.json')
    
    if verbose:
        print("Environment setup completed successfully")
    
    return True


def get_project_info() -> Dict[str, str]:
    """Get project information."""
    config = NeuroVoiceConfig()
    
    return {
        'name': config.get('project.name'),
        'version': config.get('project.version'),
        'description': config.get('project.description'),
        'author': config.get('project.author'),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'project_root': str(Path.cwd()),
        'config_file': 'model/config.json'
    }


def print_welcome():
    """Print welcome message with usage instructions."""
    info = get_project_info()
    
    print("=" * 60)
    print(f"{info['name']} v{info['version']}")
    print("=" * 60)
    print(f"Description: {info['description']}")
    print(f"Author: {info['author']}")
    print(f"Python: {info['python_version']}")
    print(f"Location: {info['project_root']}")
    print("=" * 60)
    print()
    print("Available commands:")
    print("  Training:")
    print("    python model/train.py --epochs 100 --batch_size 32")
    print("  Evaluation:")
    print("    python model/evaluate.py --checkpoint model/checkpoints/best_model.pth")
    print("  Prediction:")
    print("    python model/predict.py --checkpoint best_model.pth --audio audio.wav")
    print("=" * 60)


if __name__ == "__main__":
    # Setup environment when run directly
    print_welcome()
    
    if setup_environment(verbose=True):
        print("\nNeuroVoice is ready to use!")
        
    else:
        print("\nError setting up environment")
        sys.exit(1)