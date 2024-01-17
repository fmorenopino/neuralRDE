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
        'depth': [2],
        'step': [2],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2],
        'num_layers': [3],
        'seed': [1234],
    },
    'hyperopt-test': {
        'model_type': ['nrde'],
        'depth': [3],
        'step': [4],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2],
        'num_layers': [3],
        'seed': [111, 222, 333],
    },
    'hyperopt-original': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3, 4],
        'num_layers': [1, 2, 3],
        'seed': [1234],
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
        'step': [1],
        'hidden_dim': [8],
        'hidden_hidden_multiplier': [3],
        'num_layers': [1],
        'seed': [0],
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
        'depth': [1, 2, 3],
        'step': [1, 2, 3, 5, 10, 20, 50],
        'hyperopt_metric': ['acc'],
        'seed': [111, 222, 333],
    },
    'main-odernn': {
        # Data
        'data__batch_size': [128],
        # Main
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
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
                'data__batch_size': [2]
                #'step': [4, 32, 128]    # Total steps [500]
            },
            'hyperopt-test': {
                **default['hyperopt-test'],
                'data__batch_size': [1024]
            },
            'rnn': {
                **default['rnn'],
                'data__batch_size': [1024]
            },
            'lstm': {
                **default['lstm'],
                'data__batch_size': [1024]
            },
            'gru': {
                **default['gru'],
                'data__batch_size': [1024]
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [2],
            },
            'hyperopt-logsig-rnn': {
                **default['hyperopt-logsig-rnn'],
                'data__batch_size': [1024],
            },
            'main': {
                **default['main'],
                'data__batch_size': [1024],
                'data__adjoint': [True],
                'hyperopt_metric': ['acc'],
                'depth': [1, 2, 3],
                'step': [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            },
            'main-odernn': {
                **default['main'],
                'model_type': ['odernn_folded'],
                'data__batch_size': [1024],
                'data__adjoint': [True],
                'hyperopt_metric': ['acc'],
                'depth': [1],
                'step': [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
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
            'main_params': {
                **default['test'],
                'data__adjoint': [True]
            },
        },
    },

}

