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
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3, 4],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidal-rnn': {
        'model_type': ['rnn'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidal-gru': {
        'model_type': ['gru'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidalLong-rnn': {
        'model_type': ['rnn'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidalLong-gru': {
        'model_type': ['gru'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [2, 3],
        'num_layers': [2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidal': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [16, 32, 64],
        'hidden_hidden_multiplier': [1, 2, 3],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidal-odernn': {
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [1,4,8,32,128],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [1],
        'num_layers': [1],
        'seed': [1234],
    },
    'hyperopt-sinusoidalLong-odernn': {
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [1,4,8,32,128],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [1],
        'num_layers': [1],
        'seed': [1234],
    },
    'hyperopt-sinusoidal-ncde': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [1, 2, 3],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidal-nrde': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [1, 2, 3],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidalLong-ncde': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [1, 2, 3],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'hyperopt-sinusoidalLong-nrde': {
        'model_type': ['nrde'],
        'depth': [1],
        'step': [1],
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'hidden_hidden_multiplier': [1, 2, 3],
        'num_layers': [1, 2, 3],
        'seed': [1234],
    },
    'hyperopt-hr': {
        'model_type': ['nrde'],
        'depth': [3],
        'step': [8],
        'hidden_dim': [64],
        'hidden_hidden_multiplier': [2],
        'num_layers': [2],
        'seed': [111, 222, 333],
    },
    'hyperopt-lob': {
        'model_type': ['nrde'],
        'depth': [3],
        'step': [8],
        'hidden_dim': [32],
        'hidden_hidden_multiplier': [2],
        'num_layers': [2],
        'seed': [111, 222, 333],
    },
    'hyperopt-odernn': {
        'model_type': ['odernn_folded'],
        'depth': [1],
        'step': [8],
        'hidden_dim': [8, 16, 32, 64, 128, 256, 388],
        'hidden_hidden_multiplier': [3],
        'num_layers': [1],
        'seed': [0],
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
    'main-hr': {
        # Data
        'data__batch_size': [512],
        # Main
        'model_type': ['nrde'],
        'depth': [3],
        'step': [8],
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
                'data__batch_size': [1024],
                'step': [36]    # Total steps [500]
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
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
    # Other
    'Other': {
        'Sinusoidal': {
            'test': {
                **default['test']
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [1024],
                'step': [36]    # Total steps [500]
            },
            'hyperopt-sinusoidal-ncde': {
                **default['hyperopt-sinusoidal-ncde'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidal-rnn': {
                **default['hyperopt-sinusoidal-rnn'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidal-gru': {
                **default['hyperopt-sinusoidal-gru'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidal-nrde': {
                **default['hyperopt-sinusoidal-nrde'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidal': {
                **default['hyperopt-sinusoidal'],
                'data__batch_size': [1024],
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
            },
            'hyperopt-sinusoidal-odernn': {
                **default['hyperopt-sinusoidal-odernn'],
                'data__batch_size': [512],
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
        'SinusoidalLong': {
            'test': {
                **default['test']
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [1024],
                'step': [36]    # Total steps [500]
            },
            'hyperopt-sinusoidalLong-rnn': {
                **default['hyperopt-sinusoidalLong-rnn'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidalLong-gru': {
                **default['hyperopt-sinusoidalLong-gru'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidal': {
                **default['hyperopt-sinusoidal'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidal-ncde': {
                **default['hyperopt-sinusoidal-ncde'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidalLong-nrde': {
                **default['hyperopt-sinusoidalLong-nrde'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidalLong-ncde': {
                **default['hyperopt-sinusoidal-ncde'],
                'data__batch_size': [1024],
            },
            'hyperopt-sinusoidalLong-odernn': {
                **default['hyperopt-sinusoidalLong-odernn'],
                'data__batch_size': [512],
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
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
        'LOB': {
            'test': {
                **default['test']
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [1024],
                'step': [36]    # Total steps [500]
            },
            'hyperopt-lob': {
                **default['hyperopt-lob'],
                'data__batch_size': [2048],
            },
            'hyperopt-odernn': {
                **default['hyperopt-odernn'],
                'data__batch_size': [512],
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
            'main-hr': {
                **default['main-hr'],
                'data__batch_size': [512],
            },
            'hyperopt': {
                **default['hyperopt'],
                'data__batch_size': [512],
                'step': [8],
            },
            'hyperopt-hr': {
                **default['hyperopt-hr'],
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
            'main_params': {
                **default['test'],
                'data__adjoint': [True]
            },
        },
    },

}
