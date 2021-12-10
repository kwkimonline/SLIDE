import argparse
import numpy as np

from config import yaml_config_hook
from uif import uif, sensr_proj

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="UIF")
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()


    proj_compl = sensr_proj(args)
    uif(args, proj_compl)
    