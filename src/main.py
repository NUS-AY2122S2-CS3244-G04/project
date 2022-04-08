import argparse
import os
import sys

DIR = os.path.dirname(__file__)
sys.path.append(DIR)
from model_wrappers import wrappers
from data import datasets

def main(args):
    model = wrappers[args.model]()
    dataset = datasets[args.dataset]()
    model.set_data(dataset)
    model.set_load_filepath(args.load_filepath)
    if args.mode in {'train', 'full'}:
        model.train()
    if args.mode in {'test', 'full'}:
        model.evaluate()
        model.display_metrics()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='full', type=str, choices=['full', 'train', 'test'])
    parser.add_argument('--model_wrapper', type=str, required=True, choices=wrappers.keys())
    parser.add_argument('--dataset', type=str, required=True, choices=datasets.keys())
    parser.add_argument('--save_filepath', type=str)
    parser.add_argument('--load_filepath', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
