from collections import namedtuple

IODFConfig = namedtuple('IODFConfig', ['model_name',
                                    'variable_type', 
                                    'distribution_type', 
                                    'n_flows', 
                                    'n_levels',
                                    'n_bits',
                                    'inverse_bin_width',
                                    'splitfactor',
                                    'split_quarter',
                                    'n_mixtures',
                                    
                                    'quantize',
                                    'wq_level',

                                    'pruned',
                                    'prune_mode',

                                    'coupling_type',
                                    'splitprior_type', 
                                    'nn_depth',
                                    'n_channels_list',

                                    'out_dir',

                                    'round_approx',
                                    'hard_round',
                                    'temperature',

                                    'learning_rate',
                                    'lr_decay',
                                    'warmup',
                                    'manual_seed',
                                    'resume', 

                                    'dataset',
                                    'n_train_samples',
                                    'data_path',
                                    'input_size',
                                    'num_workers',

                                    'n_epochs',
                                    'evaluate_interval_epochs',
                                    'log_interval',

                                    'batch_size', 
                                    'device',
                                    'snap_dir'
])