class CSCMCalibrationPipeline:

    def __init__(self, data_directory, data_name_prefix, optional_arg_dict=None):
        # Initialise with the parameters and ranges to calibrate on
        # Pass the constraints to fit to
        # Possibly a pruning timeseries of data
        # Also get the path to the correct version of cscm-code to use
        # should possibly include some setup and cloning from tag and include environment or at least 
        # version check written to metadata...?
        pass

    def setup_steps_to_run(self):
        # Set to data or run
        self.prior_ensemble_run = None
        self.pruning_step = None
        self.neural_network_train = None
        self.calibration_step = None

    def run_full_calibration_pipeline(self):
        pass