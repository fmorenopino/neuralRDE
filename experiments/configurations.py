"""
configurations.py
===========================
Contains the hyper-parameter and final run configurations for each dataset.
"""

default = {
    'test': {
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [8],
        'hidden_dim': [1],
        'hidden_hidden_multiplier': [3],
        'num_layers': [4],
        'seed': [0],
    },
    'hyperopt': {
        'model_type': ['nrde'],
        'depth': [3],
        'step': [32],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2],
        'num_layers': [3],
        'seed': [1234, 4321, 2222],
    },
    'hyperopt-neural-cde': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'hyperopt-main': {
        'model_type': ['nrde'],
        'depth': [3],
        'step': [8],
        'hidden_dim': [64],
        'hidden_hidden_multiplier': [2],
        'num_layers': [2],
        'seed': [1234, 4321, 2222],
    },
    'hyperopt-test': {
        'model_type': ['nrde'],
        'depth': [3],
        'step': [4],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2],
        'num_layers': [3],
        'seed': [1234, 4321, 2222],
    },
    'hyperopt-original': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3, 4],
        'num_layers': [1, 2, 3],
        'seed': [1234, 4321, 2222],
    },
    'rnn': {
        'model_type': ['rnn'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'main-rnn': {
        'model_type': ['rnn'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [3],
        'seed': [111, 222, 333],
    },
    'main-gru': {
        'model_type': ['rnn'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [3],
        'seed': [111, 222, 333],
    },
    'lstm': {
        'model_type': ['lstm'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'gru': {
        'model_type': ['gru'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'hyperopt-odernn': {
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [128],
        'hidden_dim': [8],
        'hidden_hidden_multiplier': [3],
        'num_layers': [1],
        'seed': [111, 222, 333, 1234, 4321, 2222],
    },
    'main-odernn': {
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [128],
        'hidden_dim': [8],
        'hidden_hidden_multiplier': [3],
        'num_layers': [1],
        'seed': [111, 222, 333],
    },
    'hyperopt-logsig-rnn': {
        'model_type': ['logsig-rnn'],
        'depth': [1, 2, 3],
        'step': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        'hidden_dim': [8, 16, 32, 64, 128, 256, 388],
        'hidden_hidden_multiplier': [3],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'main': {
        # Data
        'data__batch_size': [128],
        # Main
        'model_type': ['nrde'],
        'depth': [3],
        'step': [8],
        'hidden_dim': [64],
        'hidden_hidden_multiplier': [2],
        'num_layers': [2],
        'hyperopt_metric': ['acc'],
        'seed': [111, 222, 333],
    },
    'bidmcmain': {
        'model_type': ['nrde'],
        'data__batch_size': [512],
        'hyperopt_metric': ['loss'],
        'depth': [1, 2, 3],
        'step': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        'seed': [111, 222, 333],
    },
    'bidmcmain-odernn': {
        'model_type': ['odernn_folded'],
        'data__batch_size': [512],
        'hyperopt_metric': ['loss'],
        'depth': [1],
        'step': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        'seed': [111, 222, 333],
    },
}


configs = {
    # UEA
    'UEA': {
        'EigenWorms': {
            'test': {
                **default['test']
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512]
                #'step': [4, 32, 128]    # Total steps [500]
            },
            'hyperopt-neural-cde': {
                **default['hyperopt-neural-cde'],
                'data__batch_size': [512]
                #'step': [4, 32, 128]    # Total steps [500]
            },
            'hyperopt-test': {
                **default['hyperopt-test'],
                'data__batch_size': [1024]
            },
            'rnn': {
                **default['rnn'],
                'data__batch_size': [16]
            },
            'main-rnn': {
                **default['main-rnn'],
                'data__batch_size': [512]
            },
            'main-gru': {
                **default['main-gru'],
                'data__batch_size': [512]
            },
            'lstm': {
                **default['lstm'],
                'data__batch_size': [16]
            },
            'gru': {
                **default['gru'],
                'data__batch_size': [16]
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
            },
            'main-odernn': {
                **default['main-odernn'],
                'data__batch_size': [512],
            },
            'hyperopt-logsig-rnn': {
                **default['hyperopt-logsig-rnn'],
                'data__batch_size': [1024],
            },
            'main': {
                **default['main'],
                'data__batch_size': [1024],

            },
        },
    },

    # TSR
    'TSR': {
        'BIDMC32SpO2': {
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
                'step': [8],
            },
            'hyperopt-main': {
                **default['hyperopt'],
                'data__batch_size': [512],
                'step': [8],
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
            },
            'main': {
                **default['bidmcmain'],
                'data__adjoint': [True]
            },
            'main-odernn': {
                **default['bidmcmain-odernn'],
                'data__adjoint': [True],
            },
        },

        'BIDMC32RR': {
            'test_adjoint': {
                'model_type': ['nrde'],
                'depth': [3],
                'step': [300],
                'hidden_dim': [32],
                'hidden_hidden_multiplier': [2],
                'num_layers': [2],
                'data__batch_size': [512],
                'data__adjoint': [True],
                'seed': [1234],
            },
            'test': {
                'model_type': ['nrde'],
                'depth': [3],
                'step': [300],
                'hidden_dim': [32],
                'hidden_hidden_multiplier': [2],
                'num_layers': [2],
                'data__batch_size': [512],
                'seed': [1234],
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
            },
            'main': {
                **default['bidmcmain'],
                'data__adjoint': [True]
            },
            'main-odernn': {
                **default['bidmcmain-odernn'],
                'data__adjoint': [True],
            },
        },

        'BIDMC32HR': {
            'test': {
                **default['test']
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
                #'step': [8],
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
            },
            'main': {
                **default['bidmcmain'],
                'data__adjoint': [True]
            },
            'main-odernn': {
                **default['bidmcmain-odernn'],
                'data__adjoint': [True],
            },
            'main_params': {
                **default['test'],
                'data__adjoint': [True]
            },
        },
    },

}

