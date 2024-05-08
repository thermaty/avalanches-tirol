from pathlib import Path

import yaml


def load_config(config_path=None, key: str = None):
    if config_path is None:
        config_path = 'config.yml'

    if not Path(config_path).exists():
        raise FileNotFoundError(f'Config file not found at {config_path}')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        if key is not None:
            if key not in config:
                raise ValueError('Key not found in the configuration file.')
            return config[key]
        return config
