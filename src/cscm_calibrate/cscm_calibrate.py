import json
import sys

import numpy as np
from datetime import date

from .set_up_calibration_configs_and_run import define_scendata_for_scm
from .run_prior_ensemble import run_prior_ensemble
from .prune_distribution_to_timeseries import prune_all_chunks
from .weigth_ensemble_from_constraints_and_draw import weight_ensemble_and_draw

try:
    from pandas.core.common import SettingWithCopyWarning
except:
    from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", message=".*Parameter.*")

cscm_path = os.path.join("..", "..", "..", "ciceroscm")

sys.path.insert(0, os.path.join(cscm_path, "src"))

from ciceroscm.parallel._configdistro import _ConfigDistro


class CSCMCalibrationPipeline:

    def __init__(
        self, config_file, data_directory, data_name_prefix, optional_arg_dict=None
    ):
        # Initialise with the parameters and ranges to calibrate on
        # Pass the constraints to fit to
        # Possibly a pruning timeseries of data
        # Also get the path to the correct version of cscm-code to use
        # should possibly include some setup and cloning from tag and include environment or at least
        # version check written to metadata...?
        self.read_in_configs(config_file=config_file)
        self.datestr = f"_{date.today().strftime('%Y%m%d')}"

    def read_in_configs(self, config_file):
        with open(config_file, "r") as json_config:
            configs_raw = json.loads(json_config)
        self.configs = configs_raw

    def _run_prior_ensemble(self):
        prior_cfgs = self.configs["prior_configs"]
        testconfig = _ConfigDistro(
            distro_dict=prior_cfgs["prior_distro_dict"],
            set_values=prior_cfgs["set_values"],
        )
        scenariodata = define_scendata_for_scm(
            test_data_dir=prior_cfgs["input_dir"],
            gaspam=prior_cfgs["gases"],
            df_nat_ch4=prior_cfgs["nat_ch4"],
            df_nat_n2o=prior_cfgs["nat_n2o"],
            df_conc=prior_cfgs["gases"],
            df_emis=prior_cfgs["emis"],
            nystart=prior_cfgs["nystart"],
            emstart=prior_cfgs["emstart"],
            nyend=prior_cfgs["nyend"],
        )

        run_prior_ensemble(
            testconfig=testconfig,
            scenariodata=scenariodata,
            calibdata=self.configs["constraing_configs"],
            prunecfgs=self.configs["prune_configs"],
            distnums=self.configs.get("distnums", 6000000),
            chunk_size=self.configs.get("chunk_size", 10000),
            startdate=self.datestr,
        )

    def prune_distribution(self, file_endstring=None):
        if file_endstring is None:
            file_endstring = self.datestr
        tot_samples = self.configs.get("distnums", 6000000)
        num_chunks = int(np.ceil(tot_samples / self.configs.get("chunk_size", 10000)))
        prune_lists = []
        for varname, varinfo in self.configs["prune_configs"].items():
            prune_lists.append(
                [
                    varname,
                    f"{self.configs['prior_configs']['input_dir']}{varinfo[1]}",
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
        if file_endstring is None:
            file_endstring = self.datestr
        weight_ensemble_and_draw(
            constraint_config=self.configs["constraint_configs"],
            file_endstring=file_endstring,
            output_ensemble_size=self.configs["meta_configs"]["output_ensemble_size"],
        )

    def run_full_calibration_pipeline(self):
        self.run_full_calibration_pipeline()
        self.prune_distribution()
        self.weight_ensemble_and_draw_write_config()
