import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_default_config() -> Dict[str, Any]:
    return {
        'model': 'yolo_world_l',
        'classes': ['person', 'vehicle', 'animal', 'electronic device', 'furniture'],
        'conf_threshold': 0.25,
        'iou_threshold': 0.7,
        'source': '0',
        'output': None,
        'device': 'auto'
    }

class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            return load_config(self.config_path)
        else:
            config = create_default_config()
            save_config(config, self.config_path)
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value
        save_config(self.config, self.config_path)
    
    def update_classes(self, new_classes: list):
        self.config['classes'] = new_classes
        save_config(self.config, self.config_path)