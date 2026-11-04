import runpy
import unittest
from unittest.mock import patch
from pathlib import Path
from mvesuvio.util import handle_config
from shutil import copytree, rmtree
from mantid.simpleapi import LoadAscii, CompareWorkspaces, AnalysisDataService
import mvesuvio
from mvesuvio import ConfigArgInputs
from mvesuvio.util.files_manager import FilesManager


class TestHRatioRoutine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FilesManager._experiment_dir = None
        handle_config.refresh_config_dir_and_contents()
        handle_config.set_default_config_vars()
        mvesuvio.main(ConfigArgInputs(experiment_dir="", ip_dir=""))
        cls.benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "h_ratio" / "hydrogen_intensity_ratios_estimates"
        cls.result_path = handle_config.USER_CONFIG_PATH / "experiment_template" / "hydrogen_intensity_ratios_estimates"
        reduction_inputs = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "reduction"
        copytree(
            reduction_inputs,
            handle_config.USER_CONFIG_PATH / "experiment_template" / "reduction_inputs",
            dirs_exist_ok=True
            )

    def setUp(self):
        if self.result_path.is_dir():
            rmtree(self.result_path, ignore_errors=True)
        else:
            self.result_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        AnalysisDataService.clear()
        FilesManager._experiment_dir = None

    def test_h_ratio_routine(self):
        reduction_script = handle_config.USER_CONFIG_PATH / "experiment_template" / "run_reduction.py"
        namespace = runpy.run_path(str(reduction_script), run_name="test_h_ratio_run_reduction")
        namespace["BackwardAnalysisInputs"].run_this_scattering_type = True
        namespace["ForwardAnalysisInputs"].run_this_scattering_type = True
        namespace["BackwardAnalysisInputs"].intensity_ratio_of_hydrogen_to_chosen_mass = 0
        namespace["BackwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        namespace["ForwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        namespace["BackwardAnalysisInputs"].chosen_mass_index = 0
        namespace["BackwardAnalysisInputs"].intensity_ratio_of_hydrogen_to_chosen_mass = 0
        namespace["BackwardAnalysisInputs"].constraints = ()
        namespace["ForwardAnalysisInputs"].constraints = ()
        namespace["BackwardAnalysisInputs"].do_symmetrisation = False
        namespace["ForwardAnalysisInputs"].do_symmetrisation = False
        namespace["BackwardAnalysisInputs"].mask_time_of_flight_range = None
        namespace["ForwardAnalysisInputs"].mask_time_of_flight_range = None
        with patch("builtins.input", return_value=""):
            namespace["run_reduction"]()

        bench_name = "bench_h_ratios"
        result_name = "result_h_ratios"

        LoadAscii(str(self.benchmark_path), Separator="CSV", OutputWorkspace=bench_name)
        LoadAscii(str(self.result_path), Separator="CSV", OutputWorkspace=result_name)
        result, _ = CompareWorkspaces(bench_name, result_name, Tolerance=1e-3)
        self.assertTrue(result)