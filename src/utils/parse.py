import os
import shutil
import subprocess

from absl import logging
import yaml

from .misc import uniquify_dir


def get_config(config_path, root):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(root, config["output_dir"])
    output_dir = uniquify_dir(output_dir)
    logging.info(f"Log dir: {output_dir}")
    config["output_dir"] = output_dir
    os.makedirs(config["output_dir"])

    save_config(output_dir, config_path)

    return config


def save_config(save_dir: str, config_file: str):
    # Copy config file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dest_file = os.path.join(save_dir, "config.yaml")
    shutil.copy(config_file, dest_file)

    # Save runtime information
    runtime_config = fetch_runtime_information()
    with open(os.path.join(save_dir, "runtime.yaml"), "w") as stream:
        yaml.dump(runtime_config, stream)


def fetch_runtime_information() -> dict:
    return {"commit": fetch_commit_id()}


def fetch_commit_id() -> str:
    try:
        label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return label.decode("utf-8")
    except:
        return "unknown"
