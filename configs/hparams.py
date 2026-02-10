def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG_EDF_ORI():
    def __init__(self):
        super(EEG_EDF_ORI, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'TERSE': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8621, 'im': 0.8461, 'tov_wt': 0.7, 'graph_recover_wt': 0.4, 'gmask_ratio': 0.5}
        }

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'TERSE': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'tov_wt': 0.9, 'ent_loss_wt': 0.4085, 'im': 0.8837, 'graph_recover_wt': 0.5, 'gmask_ratio': 0.5}
        }

class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'TERSE': {'pre_learning_rate': 0.003, 'learning_rate': 0.0003, 'tov_wt': 0.7, 'ent_loss_wt': 0.8528, 'im': 0.589, 'graph_recover_wt': 0.6, 'gmask_ratio': 0.5}
        }

class SENSE_42():
    def __init__(self):
        super(SENSE_42, self).__init__()
        self.train_params = {
            'num_epochs': 60,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'TERSE': {
                'pre_learning_rate': 0.001,
                'learning_rate': 0.0001,
                'tov_wt': 0.8,
                'ent_loss_wt': 0.5,
                'im': 0.8,
                'graph_recover_wt': 0.5,
                'gmask_ratio': 0.5
            }
        }