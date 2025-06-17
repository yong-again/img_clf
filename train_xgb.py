# train_xgb.py
import argparse
from parse_config import ConfigParser
from trainer import run_train_pipeline

def main(config):
    run_train_pipeline(config)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="ConvNeXt + XGBoost")
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Checkpoint path')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU device ids')

    from collections import namedtuple
    CustomArgs = namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bs', '--batch_size'], int, 'data_loader;args;batch_size'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
