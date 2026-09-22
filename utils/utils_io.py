import os
import yaml
from box import ConfigBox
import cv2
import numpy as np
import json
from typing import Optional, Any
import collections.abc

def load_cfg(cfg_path: str, default_cfg = {}) -> ConfigBox:
    final_cfg = default_cfg.copy()
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f)
            if user_cfg:
                final_cfg.update(user_cfg)
    return ConfigBox(final_cfg, box_it_up=True)


def load_cfg_recursive(cfg_path: str) -> ConfigBox:
    """
    Recursively loads YAML configuration files.
    
    If a key ends with '_path' and points to a valid file, the function 
    automatically loads that file and attaches it to the parent config 
    using a key without the '_path' suffix.
    """
    # 1. Validate file existence
    if not cfg_path or not os.path.exists(cfg_path):
        print(f"[Warning] Config file not found: {cfg_path}")
        return ConfigBox({})

    # 2. Load the primary YAML file
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f) or {}
    
    print(f"[Info] Loading config: {cfg_path}")
    
    # 3. Process the dictionary and look for sub-configs
    processed_data = raw_data.copy()
    
    for key, value in raw_data.items():
        # Check if the key indicates a path and the value is a string
        if isinstance(value, str) and key.endswith("_path"):
            # Generate the target object key (e.g., 'aria_cam_cfg_path' -> 'aria_cam_cfg')
            target_key = key.replace("_path", "")
            
            # Resolve the sub-config path (handling relative paths)
            # If the sub-path is relative, it is relative to the current working directory
            # or you can join it with the directory of the parent config if needed.
            sub_cfg_path = value
            
            if os.path.exists(sub_cfg_path):
                # Recursive call to load the child config
                processed_data[target_key] = load_cfg_recursive(sub_cfg_path)
            else:
                print(f"[Warning] Sub-config defined at '{key}' but file not found: {sub_cfg_path}")

    # 4. Return as a Box object for dot-notation access
    return ConfigBox(processed_data, box_it_up=True)


def deep_update(source_dict: dict, override_dict: dict) -> dict:
    """Recursively updates a nested dictionary (Deep Merge)."""
    for key, value in override_dict.items():
        if isinstance(value, collections.abc.Mapping):
            existing = source_dict.get(key, {})
            if isinstance(existing, collections.abc.Mapping):
                source_dict[key] = deep_update(existing, value)
            else:
                # Source is scalar but override is a dict → replace entirely
                source_dict[key] = dict(value)
        else:
            source_dict[key] = value
    return source_dict

def load_cfg_dynamic_task(base_cfg_path: str, mps_path: str, task_name: str = None) -> ConfigBox:
    """
    Dynamic Configuration Compiler.
    Saves merged configs to aria/cfg/ using the original base names.
    """
    
    def _load_recursive(cfg_path: str):
        if not cfg_path or not os.path.exists(cfg_path):
            print(f"║ [Warning] Config file not found: {cfg_path}")
            return {}
        
        with open(cfg_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f) or {}
        
        processed_data = raw_data.copy()
        for key, value in raw_data.items():
            if isinstance(value, str) and key.endswith("_path"):
                # target_key will be "AriaHands" if key is "AriaHands_path"
                target_key = key.replace("_path", "")
                
                # Path resolution logic: try absolute/project-root first, then relative to current file
                sub_cfg_path = value
                if not os.path.exists(sub_cfg_path):
                    sub_cfg_path = os.path.normpath(os.path.join(os.path.dirname(cfg_path), value))
                
                if os.path.exists(sub_cfg_path):
                    processed_data[target_key] = _load_recursive(sub_cfg_path)
                else:
                    print(f"║ [Warning] Sub-config missing: {value}")
        return processed_data

    # 1. Load Base
    print(f"║ [Config] Compiling Base Config: {base_cfg_path}")
    master_dict = _load_recursive(base_cfg_path)

    # 2. Merge Task Overrides
    if task_name:
        base_dir = os.path.dirname(base_cfg_path)
        # Search in tasks/ folder relative to base/
        task_yaml = os.path.normpath(os.path.join(base_dir, "..", "tasks", f"{task_name}.yaml"))
        
        if os.path.exists(task_yaml):
            with open(task_yaml, 'r', encoding='utf-8') as f:
                task_overrides = yaml.safe_load(f) or {}
            master_dict = deep_update(master_dict, task_overrides)
            print(f"║ [Config] Applied task overrides from: {task_name}.yaml")

    # 3. Export to aria/cfg/ and Redirect Paths
    local_cfg_dir = os.path.join(mps_path, "preprocess", "cfg")
    os.makedirs(local_cfg_dir, exist_ok=True)
    
    for key in list(master_dict.keys()):
        if key.endswith("_path") and isinstance(master_dict[key], str):
            target_key = key.replace("_path", "")
            
            if target_key in master_dict:
                sub_cfg_dict = master_dict[target_key]
                
                # Filename will now be exactly like "AriaHands.yaml"
                local_yaml_name = f"{target_key}.yaml"
                local_yaml_path = os.path.abspath(os.path.join(local_cfg_dir, local_yaml_name))
                
                with open(local_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(sub_cfg_dict, f, default_flow_style=False, sort_keys=False)
                
                # Update the _path in ConfigBox to point to the new absolute path
                master_dict[key] = local_yaml_path
                
    # Save the Master Snapshot
    with open(os.path.join(local_cfg_dir, "Preprocess.yaml"), 'w', encoding='utf-8') as f:
        yaml.safe_dump(master_dict, f, default_flow_style=False, sort_keys=False)

    print(f"║ [Config] YAML configs exported to: {local_cfg_dir}")
    return ConfigBox(master_dict, box_it_up=True)


def read_json(path: str) -> Optional[dict]:
    if not path or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_imread_gray(path: str, h: int, w: int) -> np.ndarray:
    """Read grayscale image safely and resize."""
    if path and os.path.exists(path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape[0] != h or im.shape[1] != w:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
            return im
    return np.zeros((h, w), dtype=np.uint8)


def safe_imread_rgb(path: str, h: int, w: int) -> np.ndarray:
    """Read RGB image safely and resize."""
    if path and os.path.exists(path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            if im.shape[0] != h or im.shape[1] != w:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            return im
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_json_serializable(obj: Any) -> Any:
    """Recursive helper to convert types/numpy to serializable JSON."""
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (list, tuple)): return [make_json_serializable(i) for i in obj]
    if isinstance(obj, dict): return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, type): return str(obj)
    if hasattr(obj, '__dict__'): return make_json_serializable(vars(obj))
    return str(obj)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)