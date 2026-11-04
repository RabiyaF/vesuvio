
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from mvesuvio.util.files_manager import FilesManager

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

from mvesuvio.util import handle_config
from mvesuvio.globals import FitModels
from shutil import copytree, rmtree
from mantid.simpleapi import mtd, LoadAscii, AnalysisDataService, CompareWorkspaces, Load

TESTS_ROOT = Path(__file__).resolve().parents[2]
FITTING_INPUTS_PATH = TESTS_ROOT / "data" / "analysis" / "inputs" / "fitting"


class TestFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FilesManager._experiment_dir = None
        handle_config.refresh_config_dir_and_contents()
        handle_config.set_default_config_vars()
        cls.benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "fitting" / "gauss_fit"
        cls.results_path = handle_config.USER_CONFIG_PATH / "experiment_template" / "fitting_outputs" / "gauss_fit"
        rmtree(cls.results_path, ignore_errors=True)
        copytree(
            FITTING_INPUTS_PATH,
            handle_config.USER_CONFIG_PATH / "experiment_template" / "fitting_inputs",
            dirs_exist_ok=True
            )

    def setUp(self):
        rmtree(self.results_path, ignore_errors=True)

    def tearDown(self) -> None:
        AnalysisDataService.clear()
        FilesManager._experiment_dir = None

    def test_fitting_routine(self):
        fitting_script = handle_config.USER_CONFIG_PATH / "experiment_template" / "run_fitting.py"
        namespace = runpy.run_path(str(fitting_script), run_name="test_fitting_run_fitting")
        namespace["BackwardFittingInputs"].run_this_fitting_type = False
        namespace["ForwardFittingInputs"].run_this_fitting_type = True
        namespace["ForwardFittingInputs"].fitting_model = FitModels.GAUSS
        namespace["BackwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        namespace["ForwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        namespace["BackwardAnalysisInputs"].do_symmetrisation = False
        namespace["ForwardAnalysisInputs"].do_symmetrisation = False
        namespace["BackwardAnalysisInputs"].mask_time_of_flight_range = None
        namespace["ForwardAnalysisInputs"].mask_time_of_flight_range = None
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"), patch("matplotlib.figure.Figure.savefig"):
            namespace["run_fitting"]()

        AnalysisDataService.clear()

        for prefix, path in zip(("bench", "result"), (self.benchmark_path, self.results_path)):
            for p in path.iterdir():
                if p.is_dir():
                    continue
                if p.name.endswith("nxs"):
                    Load(str(p), OutputWorkspace=prefix+"_"+p.stem)
                    continue
                LoadAscii(str(p), Separator="CSV", OutputWorkspace=prefix+"_"+p.stem)

        for ws_name in mtd.getObjectNames():
            if ws_name.startswith('bench'):
                tol = 1e-3
                result, _ = CompareWorkspaces(ws_name, ws_name.replace("bench", "result"), Tolerance=tol)
                self.assertTrue(result, f"Comparison failed for workspace: {ws_name}")