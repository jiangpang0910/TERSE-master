def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG_EDF_ORI():
    def __init__(self):
        super(EEG_EDF_ORI, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.shuffle = True
        self.drop_last = False
        self.normalize = True
        self.adj_norm = True

        # model configs
        self.input_channels = 6
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 16
        self.features_len = 65
        self.final_out_channels = 8
        self.AR_hid_dim = 6
        self.AR_hid_dim_raw = 8

        # AR Discriminator
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1

        #  new added.
        self.final_channels = 8
        self.gnn_input_dim = 520
        self.gnn_output_dim = 256
        self.dropout_spatial_gnn = 0.2
        self.dropout_graph_recover = 0.2
        self.num_splits = 8

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),  ]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = False
        self.normalize = True
        self.adj_norm = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        self.AR_hid_dim = 9
        self.AR_hid_dim_raw = 128

        #  new added.
        self.final_channels = 128
        self.gnn_input_dim = 128
        self.gnn_output_dim = 256
        self.features_len = 1
        self.dropout_spatial_gnn = 0.2
        self.dropout_graph_recover = 0.2
        self.num_splits = 8

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        self.scenarios = [("6", "19"), ("2", "11"), ("33", "12"), ("5", "26"), ("28", "4")]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True
        self.adj_norm = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.AR_hid_dim = 3
        self.AR_hid_dim_raw = 128

        #  new added.
        self.final_channels = 128
        self.gnn_input_dim = 128
        self.gnn_output_dim = 256
        self.features_len = 1
        self.dropout_spatial_gnn = 0.2
        self.dropout_graph_recover = 0.2
        self.num_splits = 8


class SENSE_42():
    """
    SENSE-42 EEG Workload dataset.
    32-channel EEG at 1024 Hz, NASA-TLX workload labels discretized into 3 classes.
    Domain adaptation is performed across participants.
    """
    def __init__(self):
        super(SENSE_42, self).__init__()
        # Domain adaptation scenarios: (source_participant, target_participant)
        # Source: persons 1-16 (pretrain), Target: persons 17-24 (adapt & evaluate)
        self.scenarios = [
            ("1", "17"), ("3", "18"), ("5", "19"), ("7", "20"),
            ("9", "21"), ("11", "22"), ("13", "23"), ("15", "24"),
        ]

        self.class_names = ['low', 'medium', 'high']
        self.sequence_len = 512   # ~0.5s windows at 1024 Hz (good for EEG workload)
        self.shuffle = True
        self.drop_last = False
        self.normalize = True
        self.adj_norm = True

        # model configs
        self.input_channels = 32  # 32 EEG channels
        self.kernel_size = 25     # ~24ms at 1024 Hz (captures EEG waveform features)
        self.stride = 6
        self.dropout = 0.3
        self.num_classes = 3      # low / medium / high workload

        self.task = "classification"
        self.label_thresholds = [30, 50]

        # CNN features
        self.mid_channels = 16
        self.final_out_channels = 32
        self.features_len = 1
        self.AR_hid_dim = 32
        self.AR_hid_dim_raw = 32

        # new added (GNN / spatial)
        self.final_channels = 32
        self.gnn_input_dim = 32
        self.gnn_output_dim = 64
        self.dropout_spatial_gnn = 0.3
        self.dropout_graph_recover = 0.3
        self.num_splits = 8

        # lstm features
        self.lstm_hid = 64
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 256


class SENSE_42_REG(SENSE_42):
    """
    SENSE-42 EEG Workload dataset -- REGRESSION variant.
    Predicts continuous NASA-TLX score (1-100) instead of discrete classes.
    Inherits all architecture settings from SENSE_42.
    """
    def __init__(self):
        super(SENSE_42_REG, self).__init__()
        self.task = "regression"
        self.output_dim = 1
        self.num_classes = 1  # single scalar output (for compatibility)
        self.class_names = ['nasa_tlx_score']
        self.label_thresholds = [30, 50]  # used by classification variant

