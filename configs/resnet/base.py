from argparse import Namespace

base = Namespace(
    model_name='IODF',

    variable_type='discrete',
    distribution_type='logistic',
    n_flows=8,
    n_levels=3,
    n_bits=8,
    inverse_bin_width=256,
    splitfactor=0,
    split_quarter=True,
    n_mixtures=5,

    quantize=False,
    a_bits=8,
    w_bits=32,
    wq_level='C',

    pruning=False,
    pruned=False,
    prune_mode='conv1',

    coupling_type='resnet',
    splitprior_type='resnet',
    nn_depth=8,
    n_channels_list=[128,128,128],

    out_dir='workspace',

    round_approx='smooth',
    hard_round=True,
    temperature=1.,

    learning_rate=0.0003,
    lr_decay=0.99,
    warmup=5,
    manual_seed=None,
    resume=None
)