"""
Main calibration pipeline class for CSCM calibration.
"""

import json
import os
import sys
import warnings
from datetime import date

import numpy as np

from .prune_distribution_to_timeseries import prune_all_chunks
from .run_prior_ensemble import run_prior_ensemble
from .set_up_calibration_configs_and_run import define_scendata_for_scm
from .shared_functions import make_constraints_config_from_RCMIP_csv, make_dataframe_of_zeros
from .weigth_ensemble_from_constraints_and_draw import weight_ensemble_and_draw

try:
    from pandas.core.common import SettingWithCopyWarning
except:  # noqa: E722
    from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", message=".*Parameter.*")

# Get path to ciceroscm - one level up from project root
cscm_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ciceroscm")
)

sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.parallel._configdistro import (  # noqa: E402
    _ConfigDistro,
)


class CSCMCalibrationPipeline:
    """
    CSCMCalibrationPipeline

    A pipeline class for running the full calibration process of the CSCM
    (Climate System Calibration Model).
    This class handles configuration loading, prior ensemble generation,
    distribution pruning, ensemble weighting,
    and orchestrates the full calibration workflow.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file containing all necessary
        calibration parameters and settings.

    Attributes
    ----------
    configs : dict
        Dictionary containing all loaded configuration parameters.
    datestr : str
        String representing the current date, used for file naming and versioning.

    Methods
    -------
    read_in_configs(config_file)
        Reads and loads the configuration file into the class instance.

    _run_prior_ensemble()
        Generates the prior ensemble based on the configuration and scenario data.

    prune_distribution(file_endstring=None)
        Prunes the generated distribution according to
        specified constraints and configuration.

    weight_ensemble_and_draw_write_config(file_endstring=None)
        Weights the ensemble, draws samples,
        and writes the resulting configuration.

    run_full_calibration_pipeline()
        Runs the complete calibration pipeline: prior ensemble, pruning,
        and weighting/drawing.
    """

    def __init__(self, config_file, constraints_to_read_separately=None):
        """
        Initialize the calibration class with configuration parameters.

        Reads in configuration settings from the specified file,
        sets up calibration parameters, constraints, and other necessary setup
        for the calibration process. Also generates a date
        string for metadata or output file naming.

        Parameters
        ----------
        config_file : str
            Path to the configuration file containing
            calibration parameters and settings.

        Attributes
        ----------
        datestr : str
            String representing the current date in the format '_YYYYMMDD',
            used for metadata or output files.

        Notes
        -----
        Additional setup such as environment checks, version control,
        or cloning from a specific tag
        may be performed within this initializer.
        """
        # Initialise with the parameters and ranges to calibrate on
        # Pass the constraints to fit to
        # Possibly a pruning timeseries of data
        # Also get the path to the correct version of cscm-code to use
        # should possibly include some setup and cloning from tag
        # and include environment or at least
        # version check written to metadata...?
        self.read_in_configs(
            config_file=config_file,
            constraints_to_read_separately=constraints_to_read_separately,
        )
        self.datestr = f"_{date.today().strftime('%Y%m%d')}"

    def read_in_configs(self, config_file, constraints_to_read_separately=None):
        """
        Read configuration settings from a JSON file and store them in the instance.

        Parameters
        ----------
        config_file : str
            Path to the JSON configuration file.

        Returns
        -------
        None

        Notes
        -----
        The loaded configuration is stored in the `self.configs` attribute.
        """
        config_file = os.path.abspath(config_file)
        config_dir = os.path.dirname(config_file)

        with open(config_file) as json_config:
            configs_raw = json.load(json_config)
        print(configs_raw)

        # Resolve relative paths in prior_configs relative to config file location
        if (
            "prior_configs" in configs_raw
            and "input_dir" in configs_raw["prior_configs"]
        ):
            input_dir = configs_raw["prior_configs"]["input_dir"]
            if not os.path.isabs(input_dir):
                # Convert relative path to absolute, relative to config file location
                configs_raw["prior_configs"]["input_dir"] = os.path.abspath(
                    os.path.join(config_dir, input_dir)
                )

        if constraints_to_read_separately is not None:
            configs_raw["constraint_configs"] = make_constraints_config_from_RCMIP_csv(
                constraints_from_RCMIP=constraints_to_read_separately
            )
            # configs_raw["constraing_configs"] = constraints_raw
        self.configs = configs_raw

    def _run_prior_ensemble(self, continue_from_existing=False, plot=False):
        """
        Run the prior ensemble simulation using configuration parameters.

        This method initializes the prior configuration distribution,
        prepares scenario data, and executes the prior ensemble run
        with the specified calibration and pruning configurations.
        It also supports optional parameters for the number of distributions
        and chunk size.

        Parameters
        ----------
        self : object
            The instance of the class containing configuration attributes.

        Returns
        -------
        None
            This method does not return a value.
            It performs the prior ensemble run and dumps output as files.

        Notes
        -----
        - Requires the following keys in `self.configs`: "prior_configs",
          "constraing_configs", "prune_configs".
        - Optional keys: "distnums", "chunk_size".
        - Relies on external functions: `_ConfigDistro`, `define_scendata_for_scm`,
          and `run_prior_ensemble`.
        """
        prior_cfgs = self.configs["prior_configs"]
        testconfig = _ConfigDistro(
            distro_dict=prior_cfgs["prior_distro_dict"],
            setvalues=prior_cfgs["set_values"],
        )
        scenariodata = define_scendata_for_scm(
            test_data_dir=prior_cfgs["input_dir"],
            gaspam=prior_cfgs["gases"],
            df_nat_ch4=prior_cfgs["nat_ch4"],
            df_nat_n2o=prior_cfgs["nat_n2o"],
            df_conc=prior_cfgs["conc"],
            df_emis=prior_cfgs["emis"],
            nystart=prior_cfgs["nystart"],
            emstart=prior_cfgs["emstart"],
            nyend=prior_cfgs["nyend"],
            sunvolc=prior_cfgs.get("sunvolc", 0),
            rf_volc_file=prior_cfgs.get("rf_volc_file", None),
            rf_solar_file=prior_cfgs.get("rf_solar_file", None),
            rf_luc_file=prior_cfgs.get("rf_luc_file", None),
        )
        print(type(scenariodata))
        scenariodata_ideadlised_experiments = None
        calibdata_idealised_experiments = None
        if "constraint_configs_idealised" in self.configs:
            scenariodata_idealised_experiments = []
            print(self.configs["constraint_configs_idealised"])
            for i, experiment in enumerate(self.configs["constraint_configs_idealised"]["Experiments"]):
                #nyend = np.max((self.configs["constraint_configs_idealised"]["Yearend_change"][i],2010))
                nyend = self.configs["constraint_configs_idealised"]["Yearend_change"][i]
                ref_yr = int(np.min((nyend-1, 2010)))
                if experiment.startswith("esm"):
                    df_emis_str = prior_cfgs["emis"].replace(prior_cfgs["emis"].split("_")[0], experiment)
                    df_conc_str = prior_cfgs["conc"].replace(prior_cfgs["conc"].split("_")[0], "piControl")
                    conc_run = False
                else:
                    df_emis_str = prior_cfgs["emis"].replace(prior_cfgs["emis"].split("_")[0], "esm-piControl")
                    df_conc_str = prior_cfgs["conc"].replace(prior_cfgs["conc"].split("_")[0], experiment)
                    conc_run = True
                scenariodata_i = define_scendata_for_scm(
                    test_data_dir=prior_cfgs["input_dir"],
                    gaspam=prior_cfgs["gases"],
                    df_nat_ch4=make_dataframe_of_zeros("CH4", 1750, nyend+1),
                    df_nat_n2o=make_dataframe_of_zeros("N2O", 1750, nyend+1),
                    df_conc=df_conc_str,
                    df_emis=df_emis_str,
                    rf_luc_file ="LUCalbedo_RCMIP_constant_zero_RCMIP3.txt",
                    nystart=1750,
                    emstart=int(nyend) + 1,
                    nyend=int(nyend),
                    sunvolc=0,
                )
                scenariodata_i[0]["conc_run"] = conc_run
                scenariodata_i[0]["scenname"] = experiment
                scenariodata_i[0]["ref_yr"] = ref_yr
                scenariodata_idealised_experiments.append(scenariodata_i[0])
                print(f"Finished loading idealised experiment scenario data for experiment: {experiment} with end year {nyend} and reference year {ref_yr}")
            #sys.exit(4)
        print("Now running prior ensemble...")
        run_prior_ensemble(
            testconfig=testconfig,
            scenariodata=scenariodata,
            calibdata=self.configs["constraint_configs"],
            prunecfgs=self.configs["prune_configs"],
            distnums=prior_cfgs.get("distnums", 6000000),
            chunk_size=prior_cfgs.get("chunk_size", 10000),
            startdate=self.datestr,
            max_workers=prior_cfgs.get("max_workers", 200),
            continue_from_existing=continue_from_existing,
            plot=plot,
            scenariodata_idealised_experiments=scenariodata_idealised_experiments,
            calibdata_idealised_experiments=self.configs.get("constraint_configs_idealised", None)
        )

    def prune_distribution(self, file_endstring=None):
        """
        Prune a distribution by processing and filtering samples

        Prune a distribution by processing and filtering samples
        To avoid memory issues when running the prior
        the output is assumed to be in chunks according
        to the configuration.

        Parameters
        ----------
        file_endstring : str, optional
            Suffix to append to output files. If None, defaults to `self.datestr`.

        Returns
        -------
        None

        Notes
        -----
        This method prepares a list of variables and their associated
        pruning information from the configuration, then calls `prune_all_chunks`
        to process the distribution in manageable chunks.
        The number of samples and chunk size are determined by the
        configuration dictionary.
        """
        if file_endstring is None:
            file_endstring = ""  # self.datestr
        prior_cfgs = self.configs["prior_configs"]
        tot_samples = prior_cfgs.get("distnums", 6000000)
        num_chunks = int(np.ceil(tot_samples / prior_cfgs.get("chunk_size", 10000)))
        prune_lists = []
        for varname, varinfo in self.configs["prune_configs"].items():
            prune_lists.append(
                [
                    varname,
                    os.path.join(
                        self.configs["prior_configs"]["input_dir"], varinfo[1]
                    ),
                    varinfo[2],
                ]
            )
        prune_all_chunks(
            total_samples=tot_samples,
            prune_lists=prune_lists,
            num_chunks=num_chunks,
            file_endstring=file_endstring,
        )

    def weight_ensemble_and_draw_write_config(self, file_endstring=None):
        """
        Weight the ensemble, draw samples, and write the configurations

        Weights the ensemble, draws samples, and writes the configurations
        using the specified file end string.

        This method calls the `weight_ensemble_and_draw` function
        with the appropriate configuration parameters.
        If no file end string is provided, it defaults to the instance's date string.


        Parameters
        ----------
        file_endstring : str, optional
            The string to append to the output file name.
            If None, uses the instance's `datestr` attribute.

        Returns
        -------
        None
        """
        if file_endstring is None:
            file_endstring = ""  # self.datestr
        print(self.configs["constraint_configs"])
        weight_ensemble_and_draw(
            constraint_config=self.configs["constraint_configs"],
            file_endstring=file_endstring,
            output_ensemble_size=self.configs["meta_configs"]["output_ensemble_size"],
        )

    def run_full_calibration_pipeline(self):
        """
        Run the complete calibration pipeline,

        Runs the complete calibration pipeline
        including prior ensemble generation, distribution pruning,
        and ensemble weighting/drawing.

        This method executes the full sequence of calibration steps in order:
        1. Runs the prior ensemble.
        2. Prunes the resulting distribution.
        3. Weights the ensemble, draws samples, and writes the configuration.

        Notes
        -----
        This method performs each calibration step sequentially.
        """
        self._run_prior_ensemble()
        self.prune_distribution()
        self.weight_ensemble_and_draw_write_config()
