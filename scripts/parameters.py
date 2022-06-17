import os 
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))


from configs.configuration import get_configs
from core.interfaces.model import load_model
from core.utils.dp_scripts import calculate_parameters_of_model

def get_parameters(args):
    cfg = get_configs(
        dataset=args.dataset,
        nn_type=args.nn_type,
        batch_size=100,
        pruning=False,
        pruned=args.pruned,
        quantize=False,
        resume=args.resume,
        out_dir='assets'
    )

    cfg.prune_mode = 'mask'

    model = load_model(cfg)

    p = calculate_parameters_of_model(model)

    file_size = os.path.getsize(os.path.join(cfg.resume, 'best.pth')) / 1e6

    return p, file_size


if __name__ == '__main__':

    argss = [
        argparse.Namespace(dataset='imagenet32', nn_type='densenet', quantize=False, pruned=False, resume='base'),
        argparse.Namespace(dataset='imagenet32', nn_type='resnet', quantize=False, pruned=False, resume='base'),
        argparse.Namespace(dataset='imagenet32', nn_type='resnet', quantize=False, pruned=True, resume='prune1'),
        argparse.Namespace(dataset='imagenet32', nn_type='resnet', quantize=False, pruned=True, resume='prune2'),
        argparse.Namespace(dataset='imagenet32', nn_type='resnet', quantize=False, pruned=True, resume='prune3'),
        argparse.Namespace(dataset='imagenet32', nn_type='resnet', quantize=False, pruned=True, resume='prune4'),
        argparse.Namespace(dataset='imagenet64', nn_type='densenet', quantize=False, pruned=False, resume='base'),
        argparse.Namespace(dataset='imagenet64', nn_type='resnet', quantize=False, pruned=False, resume='base'),
        argparse.Namespace(dataset='imagenet64', nn_type='resnet', quantize=False, pruned=True, resume='prune1'),
        argparse.Namespace(dataset='imagenet64', nn_type='resnet', quantize=False, pruned=True, resume='prune2'),
        argparse.Namespace(dataset='imagenet64', nn_type='resnet', quantize=False, pruned=True, resume='prune3'),
        # argparse.Namespace(dataset='imagenet64', nn_type='resnet', quantize=False, pruned=True, resume='prune4'),
    ]

    for args in argss:

        f = open(f'assets/{args.dataset}/parameters.txt', 'a+')

        p, file_size = get_parameters(args)

        f.write(f'{args.nn_type:<8s}-{args.resume:<7s}: parameters {p:.1f} M, checkpoint size {file_size:.1f}MB\n')