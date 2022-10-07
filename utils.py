import json
import os
from datetime import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
        # globals().update(config)
    print(config)
    return config


def update_config(path, config):
    with open(path, 'w') as json_file:
        json.dump(config, json_file, indent=4)

def current_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")